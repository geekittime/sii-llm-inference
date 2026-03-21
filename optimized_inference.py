#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimized_inference.py
======================
Qwen2.5-14B-Instruct 批量优化推理引擎

核心优化:
  1. Batch并行生成 — 所有prompt一次性打batch推理
  2. Triton融合RMSNorm / SiLU×gate算子
  3. Flash Attention (SDPA)
  4. KV-Cache批量生成循环
  5. 按长度排序分组 — 减少padding浪费
"""

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("[WARN] Triton 未安装，回退到 PyTorch 原生实现")

from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================================
# 全局配置
# ============================================================================
DEVICE         = "cuda:0"
DTYPE          = torch.float16
MAX_NEW_TOKENS = 256
BATCH_SIZE     = 32          # 默认batch大小，可根据显存调整
SEED           = 42

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True


# ============================================================================
# Triton 融合算子
# ============================================================================

if HAS_TRITON:
    # ---- Triton 融合 RMSNorm ----
    @triton.jit
    def _rms_norm_kernel(
        X, W, Y,
        stride_x, stride_y,
        N, eps,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        x_ptr = X + row * stride_x
        y_ptr = Y + row * stride_y
        _acc = tl.zeros([BLOCK], dtype=tl.float32)
        for off in range(0, N, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < N
            x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            _acc += x * x
        var = tl.sum(_acc, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        for off in range(0, N, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < N
            x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
            tl.store(y_ptr + cols, x * rstd * w, mask=mask)

    def triton_rms_norm(x, weight, eps=1e-6):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        M, N = x2.shape
        y = torch.empty_like(x2)
        BLK = min(triton.next_power_of_2(N), 4096)
        _rms_norm_kernel[(M,)](x2, weight, y, x2.stride(0), y.stride(0), N, eps, BLOCK=BLK)
        return y.reshape(shape)

    # ---- Triton 融合 SiLU × gate (SwiGLU) ----
    @triton.jit
    def _silu_mul_kernel(
        G, U, O,
        stride_g, stride_u, stride_o,
        N,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        for off in range(0, N, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < N
            g = tl.load(G + row * stride_g + cols, mask=mask, other=0.0).to(tl.float32)
            u = tl.load(U + row * stride_u + cols, mask=mask, other=0.0).to(tl.float32)
            tl.store(O + row * stride_o + cols, g * tl.sigmoid(g) * u, mask=mask)

    def triton_silu_mul(gate, up):
        shape = gate.shape
        g2 = gate.reshape(-1, shape[-1])
        u2 = up.reshape(-1, shape[-1])
        M, N = g2.shape
        o = torch.empty_like(g2)
        BLK = min(triton.next_power_of_2(N), 4096)
        _silu_mul_kernel[(M,)](g2, u2, o, g2.stride(0), u2.stride(0), o.stride(0), N, BLOCK=BLK)
        return o.reshape(shape)


# ---- 统一入口（自动回退） ----
def fused_rms_norm(x, w, eps=1e-6):
    if HAS_TRITON and x.is_cuda:
        return triton_rms_norm(x, w, eps)
    v = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(v + eps) * w).to(x.dtype)

def fused_silu_mul(gate, up):
    if HAS_TRITON and gate.is_cuda:
        return triton_silu_mul(gate, up)
    return F.silu(gate) * up


# ============================================================================
# Monkey-patch 模型层
# ============================================================================

def _patch_rmsnorm(mod):
    w, eps = mod.weight, getattr(mod, "variance_epsilon", getattr(mod, "eps", 1e-6))
    def fwd(x):
        return fused_rms_norm(x, w, eps)
    return fwd

def _patch_mlp(mod):
    gp, up, dp = mod.gate_proj, mod.up_proj, mod.down_proj
    def fwd(x):
        return dp(fused_silu_mul(gp(x), up(x)))
    return fwd

def apply_triton_optimizations(model):
    n_rms = n_mlp = 0
    for _, m in model.named_modules():
        cn = type(m).__name__
        if "RMSNorm" in cn:
            m.forward = _patch_rmsnorm(m)
            n_rms += 1
        if "MLP" in cn and hasattr(m, "gate_proj"):
            m.forward = _patch_mlp(m)
            n_mlp += 1
    print(f"[OPT] Triton RMSNorm×{n_rms}, SwiGLU×{n_mlp}")


# ============================================================================
# 加载模型
# ============================================================================

def load_model(model_path: str):
    """加载模型 + tokenizer + 全部优化，返回 (tokenizer, model)"""
    print(f"[INFO] 加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 尝试 flash_attention_2 > sdpa > eager
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
            model_path, torch_dtype=DTYPE, device_map=DEVICE, trust_remote_code=True,
        )
    model.eval()

    if HAS_TRITON:
        apply_triton_optimizations(model)

    # 预热 2 次（触发 CUDA lazy init + Triton JIT）
    print("[INFO] 预热...")
    _ids = tokenizer("warmup", return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        for _ in range(2):
            model(**_ids, use_cache=True)
    torch.cuda.synchronize(DEVICE)

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[INFO] 就绪 | {n_params:.1f}B params | VRAM {torch.cuda.memory_allocated(DEVICE)/1e9:.2f} GB")
    return tokenizer, model


# ============================================================================
# EOS token 集合
# ============================================================================

def _eos_ids(tokenizer):
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    for s in ["<|endoftext|>", "<|im_end|>", "<|end|>"]:
        t = tokenizer.convert_tokens_to_ids(s)
        if t is not None and t != getattr(tokenizer, "unk_token_id", -1):
            ids.add(t)
    eid = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eid, (list, tuple)):
        ids.update(eid)
    return ids if ids else {tokenizer.eos_token_id or 0}


# ============================================================================
# 核心：批量生成 (Batch Greedy Decode + KV-Cache)
# ============================================================================

@torch.inference_mode()
def batch_generate(
    model, tokenizer,
    prompts: list,
    max_new_tokens: int = MAX_NEW_TOKENS,
):
    """
    批量贪心生成。

    Args:
        prompts: list[str]，一批 prompt 文本

    Returns:
        outputs      : list[str]  — 每条 prompt 的生成文本
        out_lengths  : list[int]  — 每条的输出 token 数
        ttft_ms      : float      — 首 token 时间 (整个 batch 共享)
        total_ms     : float      — 端到端时间
    """
    device     = torch.device(DEVICE)
    pad_id     = tokenizer.pad_token_id or 0
    eos_ids    = _eos_ids(tokenizer)
    batch_size = len(prompts)

    # ---- tokenize (left-padding) ----
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)
    input_ids      = enc["input_ids"]                       # (B, S)
    attention_mask = enc["attention_mask"]                   # (B, S)
    input_lengths  = attention_mask.sum(dim=1).tolist()      # 每条的真实输入长度

    # ---- 生成状态 ----
    unfinished     = torch.ones(batch_size, dtype=torch.bool, device=device)
    generated      = []                                      # list of (B,1)
    sample_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    past           = None
    cur_ids        = input_ids
    cur_mask       = attention_mask

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    ttft = None

    for step in range(max_new_tokens):
        out = model(
            input_ids=cur_ids,
            attention_mask=cur_mask,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        logits = out.logits[:, -1, :]          # (B, V)
        past   = out.past_key_values

        next_tok = logits.argmax(dim=-1)       # (B,)

        # TTFT — 仅首步同步计时
        if step == 0:
            torch.cuda.synchronize(device)
            ttft = (time.perf_counter() - t0) * 1000.0

        # 已完成样本填 pad
        next_tok = torch.where(unfinished, next_tok, torch.full_like(next_tok, pad_id))

        # 检查 EOS
        is_eos = torch.zeros_like(unfinished)
        for eid in eos_ids:
            is_eos |= (next_tok == eid)

        # 有效 token（未完成 & 非 EOS）计入长度
        sample_lengths += (unfinished & ~is_eos).long()
        generated.append(next_tok.unsqueeze(1))

        # 更新完成状态
        unfinished = unfinished & ~is_eos
        if not unfinished.any():
            break

        # 准备下一步
        cur_ids  = next_tok.unsqueeze(1)
        cur_mask = torch.cat(
            [cur_mask, torch.ones(batch_size, 1, device=device, dtype=cur_mask.dtype)],
            dim=1,
        )

    torch.cuda.synchronize(device)
    total_ms = (time.perf_counter() - t0) * 1000.0
    if ttft is None:
        ttft = total_ms

    # ---- 拼接 & decode ----
    if generated:
        gen_ids = torch.cat(generated, dim=1)              # (B, steps)
    else:
        gen_ids = torch.zeros(batch_size, 0, dtype=torch.long, device=device)

    lengths = sample_lengths.tolist()
    texts   = []
    for i in range(batch_size):
        tok_ids = gen_ids[i, :lengths[i]].tolist()
        texts.append(tokenizer.decode(tok_ids, skip_special_tokens=True))

    return texts, lengths, input_lengths, ttft, total_ms


# ============================================================================
# 高层接口：分 batch 推理全部 prompts
# ============================================================================

def infer_all(
    tokenizer, model,
    prompts: list,
    batch_size: int = BATCH_SIZE,
    max_new_tokens: int = MAX_NEW_TOKENS,
    show_progress: bool = True,
):
    """
    对所有 prompt 做批量推理（按长度排序 → 分 batch → 推理 → 还原顺序）。

    Returns:
        results: list[dict] — 每条 prompt 一个 dict，包含:
            prompt, output, input_tokens, output_tokens,
            total_latency_ms, ttft_ms, throughput_tps
    """
    n = len(prompts)
    if n == 0:
        return []

    # 1) 按 token 长度排序（减少 padding 浪费）
    prompt_lens = []
    for p in prompts:
        ids = tokenizer.encode(p, add_special_tokens=False)
        prompt_lens.append(len(ids))
    sorted_idx = sorted(range(n), key=lambda i: prompt_lens[i])

    # 2) 分 batch
    all_results = [None] * n
    num_batches = math.ceil(n / batch_size)

    for b in range(num_batches):
        start = b * batch_size
        end   = min(start + batch_size, n)
        idx_batch     = sorted_idx[start:end]
        prompt_batch  = [prompts[i] for i in idx_batch]

        texts, out_lens, in_lens, ttft, total = batch_generate(
            model, tokenizer, prompt_batch, max_new_tokens,
        )

        bs = len(prompt_batch)
        for j in range(bs):
            orig_i = idx_batch[j]
            tps = out_lens[j] / total * 1000.0 if total > 0 and out_lens[j] > 0 else 0.0
            all_results[orig_i] = {
                "prompt":           prompts[orig_i],
                "output":           texts[j],
                "input_tokens":     in_lens[j],
                "output_tokens":    out_lens[j],
                "total_latency_ms": round(total, 2),
                "ttft_ms":          round(ttft, 2),
                "throughput_tps":   round(tps, 2),
            }

        if show_progress:
            done = end
            print(
                f"  [batch {b+1}/{num_batches}] "
                f"size={bs}  ttft={ttft:.0f}ms  total={total:.0f}ms  "
                f"tokens={sum(out_lens)}  ({done}/{n} done)"
            )

    return all_results


# ============================================================================
# 兼容接口：单条推理
# ============================================================================

def infer_single(tokenizer, model, prompt: str) -> dict:
    results = infer_all(tokenizer, model, [prompt], batch_size=1, show_progress=False)
    return results[0]


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="请用三句话解释KV Cache的作用。")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_path)
    res = infer_single(tokenizer, model, args.prompt)

    print(f"\n{'='*60}")
    print(f"  输出: {res['output'][:300]}")
    print(f"  输入tokens: {res['input_tokens']}  输出tokens: {res['output_tokens']}")
    print(f"  延迟: {res['total_latency_ms']:.1f}ms  TTFT: {res['ttft_ms']:.1f}ms")
    print(f"  吞吐: {res['throughput_tps']:.1f} tok/s")
    print(f"  峰值显存: {torch.cuda.max_memory_allocated(DEVICE)/1e9:.2f} GB")
    print(f"{'='*60}")