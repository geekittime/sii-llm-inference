#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimized_inference.py
======================
Qwen2.5-14B-Instruct 批量优化推理引擎

优化:
  1. Batch 并行生成 — 所有 prompt 打 batch 一次推理
  2. Triton 融合 RMSNorm / SiLU×gate 算子
  3. Flash Attention (SDPA)
  4. KV-Cache 批量贪心解码
  5. 按长度排序分组 — 减少 padding 浪费
"""

import os
import sys
import time
import math
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================================
# 全局配置
# ============================================================================
DEVICE         = "cuda:0"
DTYPE          = torch.float16
MAX_NEW_TOKENS = 256
BATCH_SIZE     = 32
SEED           = 42

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True


# ============================================================================
# 自动修复 Triton 编译环境
# ============================================================================

def _fix_triton_env():
    """
    Triton 编译 cuda_utils.c 时需要 Python.h。
    自动把 Python include 路径加到 C_INCLUDE_PATH / CPATH。
    """
    import sysconfig
    inc = sysconfig.get_path("include")
    if inc and os.path.isfile(os.path.join(inc, "Python.h")):
        # 把路径加入环境变量，让 gcc 能找到 Python.h
        for var in ("CPATH", "C_INCLUDE_PATH"):
            old = os.environ.get(var, "")
            if inc not in old:
                os.environ[var] = f"{inc}:{old}" if old else inc
        return True
    return False


def _try_install_python_dev():
    """尝试 apt 安装 python3-dev（需要 root）"""
    try:
        subprocess.check_call(
            ["apt-get", "install", "-y", "python3-dev"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=60,
        )
        return True
    except Exception:
        return False


# ============================================================================
# Triton 导入 + 运行时探测
# ============================================================================

_TRITON_IMPORTED = False
HAS_TRITON       = False

try:
    # 先修复环境
    if not _fix_triton_env():
        _try_install_python_dev()
        _fix_triton_env()

    import triton
    import triton.language as tl
    _TRITON_IMPORTED = True
except ImportError:
    pass


def _probe_triton():
    """运行时探测 Triton 是否真正能编译 + 执行内核"""
    global HAS_TRITON
    if not _TRITON_IMPORTED:
        print("[INFO] Triton 未安装，使用 PyTorch 原生优化")
        HAS_TRITON = False
        return

    try:
        @triton.jit
        def _test_kernel(X, BLOCK: tl.constexpr):
            idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            tl.store(X + idx, tl.load(X + idx) + 1.0)

        x = torch.zeros(128, device=DEVICE, dtype=torch.float32)
        _test_kernel[(1,)](x, BLOCK=128)
        torch.cuda.synchronize(DEVICE)
        if x.sum().item() == 128.0:
            HAS_TRITON = True
            print(f"[INFO] Triton {triton.__version__} 探测成功，启用融合算子")
        else:
            raise RuntimeError("Triton 探测结果不正确")
    except Exception as e:
        HAS_TRITON = False
        print(f"[WARN] Triton 运行时不可用: {e}")
        print("[INFO] 回退到 PyTorch 原生融合实现")


# ============================================================================
# Triton 融合算子
# ============================================================================

if _TRITON_IMPORTED:
    # ─── Triton 融合 RMSNorm ───
    @triton.jit
    def _rms_norm_kernel(
        X, W, Y,
        stride_x, stride_y,
        N, eps,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        x_off = X + row * stride_x
        y_off = Y + row * stride_y
        _acc = tl.zeros([BLOCK], dtype=tl.float32)
        for b in range(0, N, BLOCK):
            cols = b + tl.arange(0, BLOCK)
            mask = cols < N
            xv = tl.load(x_off + cols, mask=mask, other=0.0).to(tl.float32)
            _acc += xv * xv
        var = tl.sum(_acc, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        for b in range(0, N, BLOCK):
            cols = b + tl.arange(0, BLOCK)
            mask = cols < N
            xv = tl.load(x_off + cols, mask=mask, other=0.0).to(tl.float32)
            wv = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
            tl.store(y_off + cols, xv * rstd * wv, mask=mask)

    def _triton_rms_norm(x, weight, eps=1e-6):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        M, N = x2.shape
        y = torch.empty_like(x2)
        BLK = min(triton.next_power_of_2(N), 4096)
        _rms_norm_kernel[(M,)](
            x2, weight, y,
            x2.stride(0), y.stride(0),
            N, eps, BLOCK=BLK,
        )
        return y.reshape(shape)

    # ─── Triton 融合 SiLU × gate (SwiGLU) ───
    @triton.jit
    def _silu_mul_kernel(
        G, U, O,
        stride_g, stride_u, stride_o,
        N,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        g_off = G + row * stride_g
        u_off = U + row * stride_u
        o_off = O + row * stride_o
        for b in range(0, N, BLOCK):
            cols = b + tl.arange(0, BLOCK)
            mask = cols < N
            gv = tl.load(g_off + cols, mask=mask, other=0.0).to(tl.float32)
            uv = tl.load(u_off + cols, mask=mask, other=0.0).to(tl.float32)
            sv = gv * tl.sigmoid(gv) * uv
            tl.store(o_off + cols, sv, mask=mask)

    def _triton_silu_mul(gate, up):
        shape = gate.shape
        g2 = gate.reshape(-1, shape[-1]).contiguous()
        u2 = up.reshape(-1, shape[-1]).contiguous()
        M, N = g2.shape
        o = torch.empty_like(g2)
        BLK = min(triton.next_power_of_2(N), 4096)
        _silu_mul_kernel[(M,)](
            g2, u2, o,
            g2.stride(0), u2.stride(0), o.stride(0),
            N, BLOCK=BLK,
        )
        return o.reshape(shape)


# ============================================================================
# PyTorch 原生实现（回退）
# ============================================================================

def _pt_rms_norm(x, weight, eps=1e-6):
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps) * weight.to(torch.float32)).to(x.dtype)


def _pt_silu_mul(gate, up):
    return F.silu(gate) * up


# ============================================================================
# 统一调度
# ============================================================================

def fused_rms_norm(x, w, eps=1e-6):
    if HAS_TRITON and x.is_cuda:
        return _triton_rms_norm(x, w, eps)
    return _pt_rms_norm(x, w, eps)


def fused_silu_mul(gate, up):
    if HAS_TRITON and gate.is_cuda:
        return _triton_silu_mul(gate, up)
    return _pt_silu_mul(gate, up)


# ============================================================================
# Monkey-patch 模型层
# ============================================================================

def _make_rmsnorm_fwd(mod):
    w = mod.weight
    eps = getattr(mod, "variance_epsilon", getattr(mod, "eps", 1e-6))
    def fwd(x):
        return fused_rms_norm(x, w, eps)
    return fwd


def _make_mlp_fwd(mod):
    gp, up, dp = mod.gate_proj, mod.up_proj, mod.down_proj
    def fwd(x):
        return dp(fused_silu_mul(gp(x), up(x)))
    return fwd


def apply_optimizations(model):
    n_rms = n_mlp = 0
    for _, m in model.named_modules():
        cn = type(m).__name__
        if "RMSNorm" in cn:
            m.forward = _make_rmsnorm_fwd(m)
            n_rms += 1
        if "MLP" in cn and hasattr(m, "gate_proj"):
            m.forward = _make_mlp_fwd(m)
            n_mlp += 1
    tag = "Triton" if HAS_TRITON else "PyTorch"
    print(f"[OPT] {tag} 融合 RMSNorm×{n_rms}, SwiGLU×{n_mlp}")


# ============================================================================
# 加载模型
# ============================================================================

def load_model(model_path: str):
    print(f"[INFO] 加载模型: {model_path}")
    print(f"[INFO] 设备={DEVICE}  精度={DTYPE}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 尝试各种 attention 实现
    model = None
    for attn in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=DTYPE,
                device_map=DEVICE,
                trust_remote_code=True,
                attn_implementation=attn,
            )
            print(f"[OPT] Attention: {attn}")
            break
        except Exception:
            continue
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=DTYPE, device_map=DEVICE,
            trust_remote_code=True,
        )
        print("[OPT] Attention: default")
    model.eval()

    # 探测 Triton
    _probe_triton()

    # 替换层
    apply_optimizations(model)

    # 预热
    print("[INFO] 预热推理...")
    warm_ids = tokenizer("hello world", return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        for _ in range(3):
            model(**warm_ids, use_cache=True)
    torch.cuda.synchronize(DEVICE)

    n_p = sum(p.numel() for p in model.parameters()) / 1e9
    vram = torch.cuda.memory_allocated(DEVICE) / 1e9
    print(f"[INFO] 就绪 | {n_p:.1f}B params | VRAM {vram:.2f} GB")
    return tokenizer, model


# ============================================================================
# EOS token 集合
# ============================================================================

def _eos_ids(tokenizer):
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    for s in ["<|endoftext|>", "<|im_end|>", "<|end|>"]:
        try:
            t = tokenizer.convert_tokens_to_ids(s)
            if t is not None and t != getattr(tokenizer, "unk_token_id", -1):
                ids.add(t)
        except Exception:
            pass
    return ids if ids else {tokenizer.eos_token_id or 0}


# ============================================================================
# 核心：批量贪心生成 + KV-Cache
# ============================================================================

@torch.inference_mode()
def batch_generate(
    model, tokenizer,
    prompts: list,
    max_new_tokens: int = MAX_NEW_TOKENS,
):
    """
    批量贪心 decode + KV-Cache。

    Returns:
        texts, out_lengths, in_lengths, ttft_ms, total_ms
    """
    device  = torch.device(DEVICE)
    pad_id  = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_ids = _eos_ids(tokenizer)
    B       = len(prompts)

    # tokenize (left-padding)
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)
    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    input_lengths  = attention_mask.sum(dim=1).tolist()

    # 状态
    unfinished     = torch.ones(B, dtype=torch.bool, device=device)
    generated      = []
    sample_lengths = torch.zeros(B, dtype=torch.long, device=device)
    past           = None
    cur_ids        = input_ids
    cur_mask       = attention_mask

    # EOS 向量化检测
    eos_t = torch.tensor(sorted(eos_ids), dtype=torch.long, device=device)

    torch.cuda.synchronize(device)
    t0   = time.perf_counter()
    ttft = None

    for step in range(max_new_tokens):
        out = model(
            input_ids=cur_ids,
            attention_mask=cur_mask,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        logits = out.logits[:, -1, :]
        past   = out.past_key_values

        next_tok = logits.argmax(dim=-1)      # (B,)

        if step == 0:
            torch.cuda.synchronize(device)
            ttft = (time.perf_counter() - t0) * 1000.0

        # 已结束 → 填 pad
        next_tok = torch.where(unfinished, next_tok, torch.full_like(next_tok, pad_id))

        # EOS 检测
        is_eos = (next_tok.unsqueeze(1) == eos_t.unsqueeze(0)).any(dim=1)

        # 有效长度 += 未结束 & 非EOS
        sample_lengths += (unfinished & ~is_eos).long()
        generated.append(next_tok.unsqueeze(1))

        unfinished = unfinished & ~is_eos
        if not unfinished.any():
            break

        cur_ids  = next_tok.unsqueeze(1)
        cur_mask = torch.cat(
            [cur_mask, torch.ones(B, 1, device=device, dtype=cur_mask.dtype)],
            dim=1,
        )

    torch.cuda.synchronize(device)
    total_ms = (time.perf_counter() - t0) * 1000.0
    if ttft is None:
        ttft = total_ms

    # decode
    if generated:
        gen_ids = torch.cat(generated, dim=1)
    else:
        gen_ids = torch.zeros(B, 0, dtype=torch.long, device=device)

    lengths = sample_lengths.tolist()
    texts   = []
    for i in range(B):
        L = min(max(lengths[i], 0), gen_ids.shape[1])
        ids_list = gen_ids[i, :L].tolist() if L > 0 else []
        texts.append(tokenizer.decode(ids_list, skip_special_tokens=True))

    return texts, lengths, input_lengths, ttft, total_ms


# ============================================================================
# 高层接口：按长度排序分 batch 推理
# ============================================================================

def infer_all(
    tokenizer, model,
    prompts: list,
    batch_size: int = BATCH_SIZE,
    max_new_tokens: int = MAX_NEW_TOKENS,
    show_progress: bool = True,
):
    n = len(prompts)
    if n == 0:
        return []

    # 按长度排序
    enc_lens = []
    for p in prompts:
        enc_lens.append(len(tokenizer.encode(p, add_special_tokens=False)))
    sorted_idx = sorted(range(n), key=lambda i: enc_lens[i])

    all_results = [None] * n
    num_batches = math.ceil(n / batch_size)

    for b in range(num_batches):
        s = b * batch_size
        e = min(s + batch_size, n)
        idx_b = sorted_idx[s:e]
        p_b   = [prompts[i] for i in idx_b]

        texts, out_lens, in_lens, ttft, total = batch_generate(
            model, tokenizer, p_b, max_new_tokens,
        )

        for j in range(len(p_b)):
            oi = idx_b[j]
            tps = (out_lens[j] / total * 1000.0) if (total > 0 and out_lens[j] > 0) else 0.0
            all_results[oi] = {
                "prompt":           prompts[oi],
                "output":           texts[j],
                "input_tokens":     in_lens[j],
                "output_tokens":    out_lens[j],
                "total_latency_ms": round(total, 2),
                "ttft_ms":          round(ttft, 2),
                "throughput_tps":   round(tps, 2),
            }

        if show_progress:
            print(
                f"  [batch {b+1}/{num_batches}]  "
                f"bs={len(p_b)}  ttft={ttft:.0f}ms  total={total:.0f}ms  "
                f"out_tok={sum(out_lens)}  ({e}/{n} done)"
            )

    return all_results


def infer_single(tokenizer, model, prompt: str) -> dict:
    return infer_all(tokenizer, model, [prompt], batch_size=1, show_progress=False)[0]


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("--model_path", type=str, required=True)
    pa.add_argument("--prompt", type=str, default="请用三句话解释KV Cache的作用。")
    args = pa.parse_args()

    tok, mdl = load_model(args.model_path)
    r = infer_single(tok, mdl, args.prompt)

    print(f"\n{'='*60}")
    print(f"  输出: {r['output'][:300]}")
    print(f"  in={r['input_tokens']} out={r['output_tokens']}")
    print(f"  延迟={r['total_latency_ms']:.1f}ms  TTFT={r['ttft_ms']:.1f}ms")
    print(f"  吞吐={r['throughput_tps']:.1f} tok/s")
    print(f"  峰值显存={torch.cuda.max_memory_allocated(DEVICE)/1e9:.2f} GB")
    print(f"{'='*60}")