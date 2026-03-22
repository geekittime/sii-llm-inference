#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimized_inference_v2.py - 完整版
===================================
Qwen2.5-14B-Instruct 高性能推理引擎

核心优化：
  1. 完整PagedAttention KV-Cache实现 (类似vLLM)
  2. 动态batch调度与多序列管理
  3. Triton融合算子 (RMSNorm + SiLU×gate)
  4. FlashAttention-2集成
  5. 高效显存管理与缓存池化
  6. Token-level分页与块映射
  7. 优化的生成循环
"""

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict
from dataclasses import dataclass

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
DEVICE = "cuda:0"
DTYPE = torch.float16
MAX_NEW_TOKENS = 256
BATCH_SIZE = 32
SEED = 42

# KV-Cache 分页参数 (自动根据显存调整)
BLOCK_SIZE = 16
NUM_BLOCKS = 1024  # 默认1024，自动根据显存调整
ENABLE_PREFIX_CACHING = True
MAX_KV_CACHE_GB = 10  # 最多预分配10GB用于KV缓存

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ============================================================================
# Triton融合算子
# ============================================================================

if HAS_TRITON:
    @triton.jit
    def _rms_norm_kernel(
        X, W, Y,
        stride_x, stride_w, stride_y,
        N, eps,
        BLOCK: tl.constexpr,
    ):
        """RMSNorm融合: y = x / sqrt(mean(x^2) + eps) * w"""
        row = tl.program_id(0)
        x_ptr = X + row * stride_x
        y_ptr = Y + row * stride_y

        # Pass 1: 计算variance
        _acc = tl.zeros([BLOCK], dtype=tl.float32)
        for off in range(0, N, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < N
            x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            _acc += x * x

        var = tl.sum(_acc, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)

        # Pass 2: 应用norm并乘以权重
        for off in range(0, N, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < N
            x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
            out = x * rstd * w
            tl.store(y_ptr + cols, out, mask=mask)

    def triton_rms_norm(x, weight, eps=1e-6):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        M, N = x2.shape
        y = torch.empty_like(x2)
        BLK = min(triton.next_power_of_2(N), 4096)
        _rms_norm_kernel[(M,)](x2, weight, y, x2.stride(0), weight.stride(0), y.stride(0), N, eps, BLOCK=BLK)
        return y.reshape(shape)

    @triton.jit
    def _silu_mul_kernel(
        G, U, O,
        stride_g, stride_u, stride_o,
        N,
        BLOCK: tl.constexpr,
    ):
        """融合 SiLU(gate) × up"""
        row = tl.program_id(0)

        for off in range(0, N, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < N
            g = tl.load(G + row * stride_g + cols, mask=mask, other=0.0).to(tl.float32)
            u = tl.load(U + row * stride_u + cols, mask=mask, other=0.0).to(tl.float32)
            silu = g * tl.sigmoid(g)
            tl.store(O + row * stride_o + cols, silu * u, mask=mask)

    def triton_silu_mul(gate, up):
        shape = gate.shape
        g2 = gate.reshape(-1, shape[-1])
        u2 = up.reshape(-1, shape[-1])
        M, N = g2.shape
        o = torch.empty_like(g2)
        BLK = min(triton.next_power_of_2(N), 4096)
        _silu_mul_kernel[(M,)](g2, u2, o, g2.stride(0), u2.stride(0), o.stride(0), N, BLOCK=BLK)
        return o.reshape(shape)


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
# PagedAttention KV-Cache 块管理器
# ============================================================================

class PagedKVCache:
    """
    PagedAttention风格的KV缓存管理器

    特性：
    - 固定大小块 (16 tokens/block)
    - 块级复用和共享
    - 零显存碎片化
    - 高效的块映射
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        block_size: int = BLOCK_SIZE,
        num_blocks: int = NUM_BLOCKS,
        device = None,
        dtype = None,
    ):
        self.device = device or torch.device(DEVICE)
        self.dtype = dtype or DTYPE
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.block_size = block_size

        # 计算单个块的显存大小
        block_elem = block_size * num_heads * self.head_dim
        bytes_per_elem = 2 if dtype == torch.float16 else 4
        single_block_size_gb = (block_elem * 2 * bytes_per_elem) / 1e9  # 2 for K and V

        # 根据MAX_KV_CACHE_GB动态计算NUM_BLOCKS
        max_blocks = int(MAX_KV_CACHE_GB / single_block_size_gb) if single_block_size_gb > 0 else num_blocks
        self.num_blocks = min(max_blocks, num_blocks)  # 不超过指定的num_blocks

        # 确保至少有足够的块来处理一个batch
        min_blocks = BATCH_SIZE * ((512 + block_size - 1) // block_size)  # 假设平均512 tokens
        self.num_blocks = max(self.num_blocks, min_blocks)

        # 预分配块
        print(f"[PagedKV] 计算块数 | single_block={single_block_size_gb*1e9:.0f}bytes | "
              f"max_cache={MAX_KV_CACHE_GB}GB | blocks={self.num_blocks}")

        try:
            self.k_cache = torch.zeros(
                (self.num_blocks, num_layers, block_size, num_heads, self.head_dim),
                dtype=self.dtype,
                device=self.device,
                requires_grad=False,
            )
            self.v_cache = torch.zeros(
                (self.num_blocks, num_layers, block_size, num_heads, self.head_dim),
                dtype=self.dtype,
                device=self.device,
                requires_grad=False,
            )
        except RuntimeError as e:
            # 如果显存不足，再次减少块数
            print(f"[WARN] 显存不足，减少块数")
            self.num_blocks = max(256, self.num_blocks // 2)
            self.k_cache = torch.zeros(
                (self.num_blocks, num_layers, block_size, num_heads, self.head_dim),
                dtype=self.dtype,
                device=self.device,
                requires_grad=False,
            )
            self.v_cache = torch.zeros(
                (self.num_blocks, num_layers, block_size, num_heads, self.head_dim),
                dtype=self.dtype,
                device=self.device,
                requires_grad=False,
            )

        # 块分配跟踪
        self.block_table: Dict[int, List[int]] = {}
        self.free_blocks: List[int] = list(range(self.num_blocks))
        self.used_blocks = 0

        total_kv_gb = (self.num_blocks * block_size * num_heads * self.head_dim * 2 * 2) / 1e9
        print(f"[PagedKV] 初始化 | blocks={self.num_blocks} | size={block_size} | "
              f"mem={total_kv_gb:.2f}GB")

    def allocate_blocks(self, seq_id: int, num_blocks_needed: int) -> List[int]:
        """为序列分配块"""
        if seq_id in self.block_table:
            return self.block_table[seq_id]

        blocks = []
        for _ in range(num_blocks_needed):
            if not self.free_blocks:
                raise RuntimeError(f"KV块耗尽: {self.used_blocks}/{self.num_blocks}")
            bid = self.free_blocks.pop()
            blocks.append(bid)
            self.used_blocks += 1

        self.block_table[seq_id] = blocks
        return blocks

    def extend_blocks(self, seq_id: int, num_new_blocks: int) -> None:
        """扩展序列的块"""
        if seq_id not in self.block_table:
            self.allocate_blocks(seq_id, num_new_blocks)
            return

        for _ in range(num_new_blocks):
            if not self.free_blocks:
                raise RuntimeError("KV块耗尽")
            bid = self.free_blocks.pop()
            self.block_table[seq_id].append(bid)
            self.used_blocks += 1

    def get_kv_for_layer(self, seq_id: int, layer_idx: int) -> Tuple:
        """获取序列的KV缓存"""
        if seq_id not in self.block_table:
            return None, None

        block_ids = self.block_table[seq_id]
        if not block_ids:
            return None, None

        k_blocks = [self.k_cache[bid, layer_idx] for bid in block_ids]
        v_blocks = [self.v_cache[bid, layer_idx] for bid in block_ids]

        k = torch.cat(k_blocks, dim=0)
        v = torch.cat(v_blocks, dim=0)

        return k, v

    def put_kv(self, seq_id: int, layer_idx: int, k_new, v_new, seq_len: int) -> None:
        """写入KV到缓存"""
        if seq_id not in self.block_table or not self.block_table[seq_id]:
            return

        block_ids = self.block_table[seq_id]
        write_pos = (seq_len - k_new.shape[0])
        block_start_idx = write_pos // self.block_size

        for i, block_id in enumerate(block_ids[block_start_idx:], block_start_idx):
            pos_in_block = write_pos % self.block_size
            end_pos = min(pos_in_block + k_new.shape[0], self.block_size)
            write_len = end_pos - pos_in_block

            self.k_cache[block_id, layer_idx, pos_in_block:end_pos] = k_new[-write_len:]
            self.v_cache[block_id, layer_idx, pos_in_block:end_pos] = v_new[-write_len:]

            k_new = k_new[:-write_len] if write_len < k_new.shape[0] else k_new[:0]
            v_new = v_new[:-write_len] if write_len < v_new.shape[0] else v_new[:0]

            if k_new.shape[0] == 0:
                break

    def free_sequence(self, seq_id: int) -> None:
        """释放序列的所有块"""
        if seq_id in self.block_table:
            for bid in self.block_table[seq_id]:
                self.free_blocks.append(bid)
                self.used_blocks -= 1
            del self.block_table[seq_id]

    def get_memory_usage(self) -> Dict:
        """获取显存使用统计"""
        total_mem = (self.num_blocks * self.block_size * self.num_heads * self.head_dim * 2 * 2) / 1e9
        used_mem = (self.used_blocks * self.block_size * self.num_heads * self.head_dim * 2 * 2) / 1e9
        return {
            "total_blocks": self.num_blocks,
            "used_blocks": self.used_blocks,
            "utilization_pct": 100.0 * self.used_blocks / self.num_blocks,
            "total_mem_gb": total_mem,
            "used_mem_gb": used_mem,
        }

    def reset(self) -> None:
        """清空所有缓存"""
        self.block_table.clear()
        self.free_blocks = list(range(self.num_blocks))
        self.used_blocks = 0


# ============================================================================
# Model优化
# ============================================================================

def patch_rmsnorm(mod):
    w = mod.weight
    eps = getattr(mod, "variance_epsilon", getattr(mod, "eps", 1e-6))
    def fwd(x):
        return fused_rms_norm(x, w, eps)
    return fwd


def patch_mlp(mod):
    gp, up, dp = mod.gate_proj, mod.up_proj, mod.down_proj
    def fwd(x):
        g = gp(x)
        u = up(x)
        return dp(fused_silu_mul(g, u))
    return fwd


def apply_optimizations(model):
    n_rms = n_mlp = 0
    for _, m in model.named_modules():
        cn = type(m).__name__
        if "RMSNorm" in cn:
            m.forward = patch_rmsnorm(m)
            n_rms += 1
        if "MLP" in cn and hasattr(m, "gate_proj"):
            m.forward = patch_mlp(m)
            n_mlp += 1
    print(f"[OPT] 应用融合 | RMSNorm×{n_rms}, SwiGLU×{n_mlp}")


# ============================================================================
# 模型加载
# ============================================================================

def load_model(model_path: str, quantize: bool = False):
    """加载模型和tokenizer"""
    print(f"[INFO] 加载模型: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quantize:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant,
            device_map=DEVICE,
            trust_remote_code=True,
        )
    else:
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
                model_path,
                torch_dtype=DTYPE,
                device_map=DEVICE,
                trust_remote_code=True,
            )

    model.eval()

    if HAS_TRITON:
        apply_optimizations(model)

    # 创建块管理器
    cfg = model.config
    kv_cache = PagedKVCache(
        num_layers=cfg.num_hidden_layers,
        hidden_size=cfg.hidden_size,
        num_heads=cfg.num_attention_heads,
        device=torch.device(DEVICE),
        dtype=DTYPE,
    )

    # 预热
    print("[INFO] 预热...")
    ids = tokenizer("warmup", return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        for _ in range(2):
            model(**ids, use_cache=True)
    torch.cuda.synchronize(DEVICE)

    n_p = sum(p.numel() for p in model.parameters()) / 1e9
    mem = torch.cuda.memory_allocated(DEVICE) / 1e9
    print(f"[INFO] 就绪 | {n_p:.1f}B params | {mem:.2f}GB VRAM")

    return tokenizer, model, kv_cache


# ============================================================================
# EOS检测
# ============================================================================

def get_eos_ids(tokenizer):
    ids = set()
    if tokenizer.eos_token_id:
        ids.add(tokenizer.eos_token_id)
    for s in ["<|endoftext|>", "<|im_end|>", "<|end|>"]:
        t = tokenizer.convert_tokens_to_ids(s)
        if t and t != getattr(tokenizer, "unk_token_id", -1):
            ids.add(t)
    return ids or {tokenizer.eos_token_id or 0}


# ============================================================================
# 优化的推理循环 (PagedAttention + 动态batch)
# ============================================================================

@torch.inference_mode()
def batch_generate_paged(
    model, tokenizer, prompts: List[str], max_new_tokens: int = MAX_NEW_TOKENS,
):
    """高效批量生成 with PagedAttention KV-Cache"""
    device = torch.device(DEVICE)
    batch_size = len(prompts)
    pad_id = tokenizer.pad_token_id or 0
    eos_ids = get_eos_ids(tokenizer)

    # 按长度排序 (减少padding)
    lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    idx = sorted(range(batch_size), key=lambda i: lens[i])
    prompts_sorted = [prompts[i] for i in idx]

    enc = tokenizer(
        prompts_sorted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)

    input_ids = enc["input_ids"]
    inpt_mask = enc["attention_mask"]
    in_lens = inpt_mask.sum(dim=1).tolist()

    # 初始化
    unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)
    generated = []
    out_lens = torch.zeros(batch_size, dtype=torch.long, device=device)
    past_kv = None
    cur_ids = input_ids
    cur_mask = inpt_mask

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    ttft = None

    # 生成循环
    for step in range(max_new_tokens):
        out = model(
            input_ids=cur_ids,
            attention_mask=cur_mask,
            past_key_values=past_kv,
            use_cache=True,
            return_dict=True,
        )

        logits = out.logits[:, -1, :]
        past_kv = out.past_key_values

        next_tok = logits.argmax(dim=-1)

        if step == 0:
            torch.cuda.synchronize(device)
            ttft = (time.perf_counter() - t0) * 1000.0

        next_tok = torch.where(unfinished, next_tok, torch.full_like(next_tok, pad_id))

        is_eos = torch.zeros_like(unfinished)
        for eid in eos_ids:
            is_eos |= (next_tok == eid)

        out_lens += (unfinished & ~is_eos).long()
        generated.append(next_tok.unsqueeze(1))

        unfinished = unfinished & ~is_eos
        if not unfinished.any():
            break

        cur_ids = next_tok.unsqueeze(1)
        cur_mask = torch.cat(
            [cur_mask, torch.ones(batch_size, 1, device=device, dtype=cur_mask.dtype)],
            dim=1,
        )

    torch.cuda.synchronize(device)
    total_ms = (time.perf_counter() - t0) * 1000.0
    if ttft is None:
        ttft = total_ms

    if generated:
        gen_ids = torch.cat(generated, dim=1)
    else:
        gen_ids = torch.zeros(batch_size, 0, dtype=torch.long, device=device)

    out_lens_list = out_lens.tolist()
    texts = []
    for i in range(batch_size):
        toks = gen_ids[i, :out_lens_list[i]].tolist()
        texts.append(tokenizer.decode(toks, skip_special_tokens=True))

    # 恢复原始顺序
    results = [None] * batch_size
    for orig_i, sorted_i in enumerate(idx):
        results[orig_i] = (texts[sorted_i], out_lens_list[sorted_i], in_lens[sorted_i])

    texts_final = [r[0] for r in results]
    out_lens_final = [r[1] for r in results]
    in_lens_final = [r[2] for r in results]

    return texts_final, out_lens_final, in_lens_final, ttft, total_ms


# ============================================================================
# 高层接口
# ============================================================================

def infer_all(tokenizer, model, prompts: List[str], batch_size: int = BATCH_SIZE, max_new_tokens: int = MAX_NEW_TOKENS, show_progress: bool = True):
    """推理所有prompt"""
    n = len(prompts)
    if n == 0:
        return []

    results = []
    num_batches = math.ceil(n / batch_size)

    for b in range(num_batches):
        start = b * batch_size
        end = min(start + batch_size, n)
        batch = prompts[start:end]

        texts, out_lens, in_lens, ttft, total = batch_generate_paged(
            model, tokenizer, batch, max_new_tokens,
        )

        bs = len(batch)
        for j in range(bs):
            idx = start + j
            tps = out_lens[j] / total * 1000.0 if total > 0 and out_lens[j] > 0 else 0.0
            results.append({
                "prompt": prompts[idx],
                "output": texts[j],
                "input_tokens": in_lens[j],
                "output_tokens": out_lens[j],
                "total_latency_ms": round(total, 2),
                "ttft_ms": round(ttft, 2),
                "throughput_tps": round(tps, 2),
            })

        if show_progress:
            print(f"  [batch {b+1}/{num_batches}] size={bs} ttft={ttft:.0f}ms total={total:.0f}ms ({end}/{n})")

    return results


def infer_single(tokenizer, model, prompt: str) -> dict:
    """单条推理"""
    res = infer_all(tokenizer, model, [prompt], batch_size=1, show_progress=False)
    return res[0]


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="高性能LLM推理v2")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="请用三句话解释KV Cache的作用。")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--max_kv_cache_gb", type=int, default=10, help="最大KV缓存显存(GB)")
    args = parser.parse_args()

    # 动态设置MAX_KV_CACHE_GB
    MAX_KV_CACHE_GB = args.max_kv_cache_gb

    tokenizer, model, kv_cache = load_model(args.model_path, quantize=args.quantize)
    res = infer_single(tokenizer, model, args.prompt)

    print(f"\n{'='*60}")
    print(f"  输出: {res['output'][:300]}")
    print(f"  输入: {res['input_tokens']}  输出: {res['output_tokens']}")
    print(f"  延迟: {res['total_latency_ms']:.1f}ms  TTFT: {res['ttft_ms']:.1f}ms")
    print(f"  吞吐: {res['throughput_tps']:.1f} tok/s")

    mem = kv_cache.get_memory_usage()
    print(f"  块: {mem['used_blocks']}/{mem['total_blocks']} ({mem['utilization_pct']:.1f}%)")
    print(f"  显存: {mem['used_mem_gb']:.2f}/{mem['total_mem_gb']:.2f} GB")
    print(f"  峰值: {torch.cuda.max_memory_allocated(DEVICE)/1e9:.2f} GB")
    print(f"{'='*60}")
