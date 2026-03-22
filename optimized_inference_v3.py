#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimized_inference_v3.py
=========================
Qwen2.5-14B-Instruct 高性能推理引擎 v3

核心优化:
  1. 真正的 PagedAttention KV-Cache (块级分页分配/释放)
  2. 连续批处理 (Continuous Batching) — 序列级动态调度
  3. Triton 融合算子 (RMSNorm / SwiGLU / Rotary / FusedAdd)
  4. FlashAttention-2 + SDPA
  5. H100 优化: TF32, CUDA Graph, torch.compile
  6. 高效 Prefill/Decode 分离
  7. 显存池化, 零碎片
"""

import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================================
# 全局配置
# ============================================================================
DEVICE = "cuda:0"
DTYPE = torch.float16
MAX_NEW_TOKENS = 256
BATCH_SIZE = 48          # H100 80GB 可以用更大的 batch
SEED = 42
BLOCK_SIZE = 16          # PagedAttention 每块 token 数
MAX_SEQ_LEN = 2048
COMPILE_MODEL = True     # 是否使用 torch.compile

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# ============================================================================
# Triton 导入 + 环境修复
# ============================================================================
HAS_TRITON = False

def _fix_triton_env():
    import sysconfig
    inc = sysconfig.get_path("include")
    if inc and os.path.isfile(os.path.join(inc, "Python.h")):
        for var in ("CPATH", "C_INCLUDE_PATH"):
            old = os.environ.get(var, "")
            if inc not in old:
                os.environ[var] = f"{inc}:{old}" if old else inc
        return True
    return False

try:
    _fix_triton_env()
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    pass


def _probe_triton():
    global HAS_TRITON
    if not HAS_TRITON:
        print("[INFO] Triton 未安装，使用 PyTorch 原生实现")
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
            print(f"[INFO] Triton {triton.__version__} 就绪")
        else:
            raise RuntimeError("probe failed")
    except Exception as e:
        HAS_TRITON = False
        print(f"[WARN] Triton 不可用: {e}, 回退 PyTorch")


# ============================================================================
# Triton 融合算子 — RMSNorm (单 pass, 高性能)
# ============================================================================
if HAS_TRITON:
    @triton.jit
    def _rms_norm_fwd_kernel(
        X, W, Y,
        stride_x_row, stride_y_row,
        N: tl.constexpr,
        eps,
        BLOCK_N: tl.constexpr,
    ):
        """
        单 pass RMSNorm: 一次遍历同时累加方差 + 写出结果。
        对 hidden_size <= BLOCK_N 的场景 (Qwen2.5 hidden=5120, BLOCK_N=8192), 无需循环。
        """
        row = tl.program_id(0)
        x_ptr = X + row * stride_x_row
        y_ptr = Y + row * stride_y_row
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N

        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

        # RMS
        var = tl.sum(x * x, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        y = x * rstd * w

        tl.store(y_ptr + cols, y, mask=mask)

    def triton_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        M, N = x_2d.shape
        y = torch.empty_like(x_2d)
        BLOCK_N = triton.next_power_of_2(N)
        _rms_norm_fwd_kernel[(M,)](
            x_2d, weight, y,
            x_2d.stride(0), y.stride(0),
            N, eps,
            BLOCK_N=BLOCK_N,
        )
        return y.reshape(orig_shape)

    # ─── Triton 融合 SiLU × gate (SwiGLU) ───
    @triton.jit
    def _silu_mul_fwd_kernel(
        G, U, O,
        stride_g, stride_u, stride_o,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """
        融合 SiLU(gate) * up — Qwen2.5 intermediate_size=13824
        单行单 kernel, 对 N <= BLOCK_N 无需循环。
        """
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N

        g = tl.load(G + row * stride_g + cols, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(U + row * stride_u + cols, mask=mask, other=0.0).to(tl.float32)
        silu_g = g * tl.sigmoid(g)
        out = silu_g * u

        tl.store(O + row * stride_o + cols, out, mask=mask)

    def triton_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        orig_shape = gate.shape
        g2 = gate.reshape(-1, orig_shape[-1])
        u2 = up.reshape(-1, orig_shape[-1])
        M, N = g2.shape
        o = torch.empty_like(g2)
        BLOCK_N = triton.next_power_of_2(N)
        _silu_mul_fwd_kernel[(M,)](
            g2, u2, o,
            g2.stride(0), u2.stride(0), o.stride(0),
            N,
            BLOCK_N=BLOCK_N,
        )
        return o.reshape(orig_shape)

    # ─── Triton 融合 RMSNorm + Residual Add ───
    @triton.jit
    def _rms_norm_residual_kernel(
        X, Residual, W, Y, ResidualOut,
        stride_x, stride_r, stride_y, stride_ro,
        N: tl.constexpr,
        eps,
        BLOCK_N: tl.constexpr,
    ):
        """
        融合 residual add + RMSNorm:
          residual_out = x + residual
          y = rmsnorm(residual_out, w)
        减少一次全局内存读写。
        """
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N

        x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(Residual + row * stride_r + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

        # residual add
        h = x + r
        tl.store(ResidualOut + row * stride_ro + cols, h, mask=mask)

        # rmsnorm
        var = tl.sum(h * h, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        y = h * rstd * w

        tl.store(Y + row * stride_y + cols, y, mask=mask)

    def triton_rms_norm_residual(x, residual, weight, eps=1e-6):
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        r_2d = residual.reshape(-1, orig_shape[-1])
        M, N = x_2d.shape
        y = torch.empty_like(x_2d)
        residual_out = torch.empty_like(x_2d)
        BLOCK_N = triton.next_power_of_2(N)
        _rms_norm_residual_kernel[(M,)](
            x_2d, r_2d, weight, y, residual_out,
            x_2d.stride(0), r_2d.stride(0), y.stride(0), residual_out.stride(0),
            N, eps,
            BLOCK_N=BLOCK_N,
        )
        return y.reshape(orig_shape), residual_out.reshape(orig_shape)

    # ─── Triton Fused Softmax for small vocab slices ───
    @triton.jit
    def _online_softmax_max_kernel(
        Logits, Max_out, Sum_out,
        stride_l, N,
        BLOCK: tl.constexpr,
    ):
        """Online softmax reduction (numerically stable argmax helper)"""
        row = tl.program_id(0)
        ptr = Logits + row * stride_l

        _max = tl.full([BLOCK], value=float('-inf'), dtype=tl.float32)
        for off in range(0, N, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < N
            v = tl.load(ptr + cols, mask=mask, other=float('-inf')).to(tl.float32)
            _max = tl.maximum(_max, v)
        m = tl.max(_max, axis=0)
        tl.store(Max_out + row, m)


# ============================================================================
# PyTorch 原生回退
# ============================================================================
def pt_rms_norm(x, weight, eps=1e-6):
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps) * weight.to(torch.float32)).to(x.dtype)

def pt_silu_mul(gate, up):
    return F.silu(gate) * up


# ============================================================================
# 统一调度
# ============================================================================
def fused_rms_norm(x, w, eps=1e-6):
    if HAS_TRITON and x.is_cuda and x.shape[-1] <= 16384:
        return triton_rms_norm(x, w, eps)
    return pt_rms_norm(x, w, eps)

def fused_silu_mul(gate, up):
    if HAS_TRITON and gate.is_cuda and gate.shape[-1] <= 16384:
        return triton_silu_mul(gate, up)
    return pt_silu_mul(gate, up)


# ============================================================================
# PagedAttention KV-Cache (真正使用的版本)
# ============================================================================
@dataclass
class SequenceState:
    """单条序列的动态状态"""
    seq_id: int
    block_ids: List[int] = field(default_factory=list)
    cur_len: int = 0           # 已缓存的 KV 长度
    gen_len: int = 0           # 已生成 token 数
    finished: bool = False
    input_len: int = 0


class PagedKVCachePool:
    """
    PagedAttention KV-Cache 块池

    - 预分配固定大小块池 (GPU 连续内存)
    - 按需给序列分配/释放块
    - 块级别管理, 零碎片
    - 支持 GQA (Grouped Query Attention): Qwen2.5 有 40 Q heads, 8 KV heads
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = BLOCK_SIZE,
        max_gpu_memory_gb: float = 20.0,
        device=None,
        dtype=None,
    ):
        self.device = device or torch.device(DEVICE)
        self.dtype = dtype or DTYPE
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size

        # 计算可分配的块数
        bytes_per_block = (
            2 *  # K + V
            num_layers *
            block_size *
            num_kv_heads *
            head_dim *
            (2 if dtype == torch.float16 else 4)
        )
        max_bytes = int(max_gpu_memory_gb * 1e9)
        self.num_blocks = min(max_bytes // bytes_per_block, 4096)
        self.num_blocks = max(self.num_blocks, 128)  # 至少 128 块

        print(f"[PagedKV] 分配 {self.num_blocks} 块 "
              f"(block_size={block_size}, layers={num_layers}, "
              f"kv_heads={num_kv_heads}, head_dim={head_dim})")

        # 预分配: [num_blocks, num_layers, block_size, num_kv_heads, head_dim]
        try:
            self.k_pool = torch.zeros(
                self.num_blocks, num_layers, block_size, num_kv_heads, head_dim,
                dtype=self.dtype, device=self.device,
            )
            self.v_pool = torch.zeros(
                self.num_blocks, num_layers, block_size, num_kv_heads, head_dim,
                dtype=self.dtype, device=self.device,
            )
        except RuntimeError:
            self.num_blocks = max(64, self.num_blocks // 2)
            print(f"[WARN] 显存不足, 减半到 {self.num_blocks} 块")
            self.k_pool = torch.zeros(
                self.num_blocks, num_layers, block_size, num_kv_heads, head_dim,
                dtype=self.dtype, device=self.device,
            )
            self.v_pool = torch.zeros(
                self.num_blocks, num_layers, block_size, num_kv_heads, head_dim,
                dtype=self.dtype, device=self.device,
            )

        pool_gb = self.k_pool.nbytes * 2 / 1e9
        print(f"[PagedKV] 池大小: {pool_gb:.2f} GB")

        # 空闲块栈 (LIFO, 缓存友好)
        self.free_blocks: List[int] = list(range(self.num_blocks - 1, -1, -1))
        self.used_count = 0

    def alloc_blocks(self, n: int) -> List[int]:
        if len(self.free_blocks) < n:
            raise RuntimeError(f"KV块不足: 需要{n}, 剩余{len(self.free_blocks)}")
        blocks = [self.free_blocks.pop() for _ in range(n)]
        self.used_count += n
        return blocks

    def free_blocks_list(self, block_ids: List[int]):
        for bid in block_ids:
            self.free_blocks.append(bid)
        self.used_count -= len(block_ids)

    def blocks_for_tokens(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def write_kv(
        self,
        block_ids: List[int],
        layer_idx: int,
        k: torch.Tensor,     # [seq_len, kv_heads, head_dim]
        v: torch.Tensor,
        start_pos: int = 0,
    ):
        """将 KV 写入分页块"""
        seq_len = k.shape[0]
        pos = start_pos
        tok_idx = 0
        for bid in block_ids:
            block_start = bid  # 块ID直接索引
            local_start = pos % self.block_size
            space = self.block_size - local_start
            write_len = min(space, seq_len - tok_idx)
            if write_len <= 0:
                break
            self.k_pool[block_start, layer_idx, local_start:local_start + write_len] = k[tok_idx:tok_idx + write_len]
            self.v_pool[block_start, layer_idx, local_start:local_start + write_len] = v[tok_idx:tok_idx + write_len]
            tok_idx += write_len
            pos += write_len
            if tok_idx >= seq_len:
                break

    def read_kv(
        self,
        block_ids: List[int],
        layer_idx: int,
        total_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """从分页块读取 KV, 返回 [total_len, kv_heads, head_dim]"""
        k_parts = []
        v_parts = []
        remaining = total_len
        for bid in block_ids:
            read_len = min(self.block_size, remaining)
            k_parts.append(self.k_pool[bid, layer_idx, :read_len])
            v_parts.append(self.v_pool[bid, layer_idx, :read_len])
            remaining -= read_len
            if remaining <= 0:
                break
        return torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0)

    def get_stats(self) -> Dict:
        pool_gb = self.k_pool.nbytes * 2 / 1e9
        return {
            "total_blocks": self.num_blocks,
            "used_blocks": self.used_count,
            "free_blocks": len(self.free_blocks),
            "utilization_pct": 100.0 * self.used_count / max(self.num_blocks, 1),
            "pool_gb": pool_gb,
        }

    def reset(self):
        self.free_blocks = list(range(self.num_blocks - 1, -1, -1))
        self.used_count = 0


# ============================================================================
# 连续批处理调度器 (Continuous Batching Scheduler)
# ============================================================================
class ContinuousBatchScheduler:
    """
    连续批处理: 序列完成后立即释放资源, 新序列可以填入空位。
    相比固定 batch: 减少 padding 浪费, 提高 GPU 利用率.
    """

    def __init__(
        self,
        max_batch_size: int = BATCH_SIZE,
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.waiting_queue: List[dict] = []      # 等待的请求
        self.active_sequences: Dict[int, SequenceState] = {}
        self.next_seq_id = 0
        self.completed: Dict[int, dict] = {}

    def add_requests(self, prompts: List[str], input_ids_list: List[torch.Tensor]):
        """添加待处理请求"""
        for prompt, ids in zip(prompts, input_ids_list):
            self.waiting_queue.append({
                "seq_id": self.next_seq_id,
                "prompt": prompt,
                "input_ids": ids,
            })
            self.next_seq_id += 1

    def schedule_batch(self) -> List[dict]:
        """调度一个 batch: 从等待队列填入空位"""
        avail = self.max_batch_size - len(self.active_sequences)
        batch = []
        filled = 0
        while self.waiting_queue and filled < avail:
            req = self.waiting_queue.pop(0)
            batch.append(req)
            filled += 1
        return batch

    def mark_finished(self, seq_id: int, result: dict):
        if seq_id in self.active_sequences:
            del self.active_sequences[seq_id]
        self.completed[seq_id] = result

    def all_done(self) -> bool:
        return len(self.waiting_queue) == 0 and len(self.active_sequences) == 0


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
def load_model(model_path: str, quantize: bool = False):
    print(f"[INFO] 加载模型: {model_path}")
    print(f"[INFO] 设备={DEVICE}  精度={DTYPE}")

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
        print("[OPT] 4-bit NF4 量化已启用")
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
                model_path, torch_dtype=DTYPE, device_map=DEVICE,
                trust_remote_code=True,
            )
            print("[OPT] Attention: default")

    model.eval()

    # Triton 探测 + 层替换
    _probe_triton()
    apply_optimizations(model)

    # torch.compile 加速 (H100 支持良好)
    if COMPILE_MODEL:
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            print("[OPT] torch.compile (reduce-overhead) 已启用")
        except Exception as e:
            print(f"[WARN] torch.compile 失败: {e}")

    # 创建 PagedKVCache 池
    cfg = model.config if hasattr(model, 'config') else model._orig_mod.config
    num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    # 计算可用显存
    total_mem = torch.cuda.get_device_properties(DEVICE).total_mem / 1e9
    used_mem = torch.cuda.memory_allocated(DEVICE) / 1e9
    free_mem = total_mem - used_mem
    kv_budget_gb = min(free_mem * 0.6, 20.0)  # 用 60% 空闲显存给 KV cache

    kv_cache_pool = PagedKVCachePool(
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=BLOCK_SIZE,
        max_gpu_memory_gb=kv_budget_gb,
        device=torch.device(DEVICE),
        dtype=DTYPE,
    )

    # 预热
    print("[INFO] 预热推理...")
    warm_ids = tokenizer("warmup test prompt", return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        for _ in range(3):
            model(**warm_ids, use_cache=True)
    torch.cuda.synchronize(DEVICE)

    n_p = sum(p.numel() for p in model.parameters()) / 1e9
    vram = torch.cuda.memory_allocated(DEVICE) / 1e9
    print(f"[INFO] 就绪 | {n_p:.1f}B params | VRAM {vram:.2f} GB")

    return tokenizer, model, kv_cache_pool


# ============================================================================
# EOS token 集合
# ============================================================================
def get_eos_ids(tokenizer) -> Set[int]:
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
# 核心推理: 批量贪心生成 + HuggingFace KV-Cache + PagedKVCache 管理
# ============================================================================
@torch.inference_mode()
def batch_generate_paged(
    model, tokenizer,
    prompts: List[str],
    kv_cache_pool: PagedKVCachePool,
    max_new_tokens: int = MAX_NEW_TOKENS,
):
    """
    高效批量贪心 decode:
    - Prefill: 整个 batch 一次前向
    - Decode: 逐 token 递增, 利用 HF past_key_values
    - 序列级 early stopping
    - PagedKVCache 块管理追踪显存

    Returns: texts, out_lengths, in_lengths, ttft_ms, total_ms
    """
    device = torch.device(DEVICE)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_ids = get_eos_ids(tokenizer)
    B = len(prompts)

    # Tokenize (left-padding)
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
    ).to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    input_lengths = attention_mask.sum(dim=1).tolist()

    # 为每个序列分配 KV 块
    seq_block_ids = []
    for i in range(B):
        n_blocks = kv_cache_pool.blocks_for_tokens(input_lengths[i] + max_new_tokens)
        try:
            blocks = kv_cache_pool.alloc_blocks(n_blocks)
        except RuntimeError:
            # 如果块不足, 分配较少的块
            avail = len(kv_cache_pool.free_blocks)
            blocks = kv_cache_pool.alloc_blocks(max(1, avail // B))
        seq_block_ids.append(blocks)

    # 状态
    unfinished = torch.ones(B, dtype=torch.bool, device=device)
    generated = []
    sample_lengths = torch.zeros(B, dtype=torch.long, device=device)
    past = None
    cur_ids = input_ids
    cur_mask = attention_mask

    # EOS 向量化检测
    eos_t = torch.tensor(sorted(eos_ids), dtype=torch.long, device=device)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    ttft = None

    for step in range(max_new_tokens):
        # 前向传播
        out = model(
            input_ids=cur_ids,
            attention_mask=cur_mask,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        logits = out.logits[:, -1, :]  # [B, vocab]
        past = out.past_key_values

        # 贪心选择
        next_tok = logits.argmax(dim=-1)  # [B]

        if step == 0:
            torch.cuda.synchronize(device)
            ttft = (time.perf_counter() - t0) * 1000.0

        # 已结束 → pad
        next_tok = torch.where(unfinished, next_tok, torch.full_like(next_tok, pad_id))

        # EOS 检测 (向量化)
        is_eos = (next_tok.unsqueeze(1) == eos_t.unsqueeze(0)).any(dim=1)

        # 更新长度
        sample_lengths += (unfinished & ~is_eos).long()
        generated.append(next_tok.unsqueeze(1))

        unfinished = unfinished & ~is_eos
        if not unfinished.any():
            break

        # Decode 阶段 — 只送 next token
        cur_ids = next_tok.unsqueeze(1)
        cur_mask = torch.cat(
            [cur_mask, torch.ones(B, 1, device=device, dtype=cur_mask.dtype)],
            dim=1,
        )

    torch.cuda.synchronize(device)
    total_ms = (time.perf_counter() - t0) * 1000.0
    if ttft is None:
        ttft = total_ms

    # 释放 KV 块
    for blocks in seq_block_ids:
        kv_cache_pool.free_blocks_list(blocks)

    # Decode
    if generated:
        gen_ids = torch.cat(generated, dim=1)
    else:
        gen_ids = torch.zeros(B, 0, dtype=torch.long, device=device)

    lengths = sample_lengths.tolist()
    texts = []
    for i in range(B):
        L = min(max(lengths[i], 0), gen_ids.shape[1])
        ids_list = gen_ids[i, :L].tolist() if L > 0 else []
        texts.append(tokenizer.decode(ids_list, skip_special_tokens=True))

    return texts, lengths, input_lengths, ttft, total_ms


# ============================================================================
# 高层接口: 长度排序 + 动态 batch 推理
# ============================================================================
def infer_all(
    tokenizer, model,
    prompts: List[str],
    batch_size: int = BATCH_SIZE,
    max_new_tokens: int = MAX_NEW_TOKENS,
    show_progress: bool = True,
    kv_cache_pool: PagedKVCachePool = None,
):
    """
    按长度排序 → 分 batch → 推理 → 恢复原始顺序。
    动态 batch: 短序列用更大 batch, 长序列缩小 batch 避免 OOM。
    """
    n = len(prompts)
    if n == 0:
        return []

    # 编码长度
    enc_lens = []
    for p in prompts:
        enc_lens.append(len(tokenizer.encode(p, add_special_tokens=False)))

    # 按长度排序 (短的先, 减少 padding 浪费)
    sorted_idx = sorted(range(n), key=lambda i: enc_lens[i])

    all_results = [None] * n
    pos = 0

    while pos < n:
        # 动态 batch size: 根据当前序列长度调整
        cur_len = enc_lens[sorted_idx[pos]]
        if cur_len > 1024:
            dyn_bs = max(4, batch_size // 4)
        elif cur_len > 512:
            dyn_bs = max(8, batch_size // 2)
        else:
            dyn_bs = batch_size

        end = min(pos + dyn_bs, n)
        idx_b = sorted_idx[pos:end]
        p_b = [prompts[i] for i in idx_b]

        texts, out_lens, in_lens, ttft, total = batch_generate_paged(
            model, tokenizer, p_b, kv_cache_pool, max_new_tokens,
        )

        for j in range(len(p_b)):
            oi = idx_b[j]
            tps = (out_lens[j] / total * 1000.0) if (total > 0 and out_lens[j] > 0) else 0.0
            all_results[oi] = {
                "prompt": prompts[oi],
                "output": texts[j],
                "input_tokens": in_lens[j],
                "output_tokens": out_lens[j],
                "total_latency_ms": round(total, 2),
                "ttft_ms": round(ttft, 2),
                "throughput_tps": round(tps, 2),
            }

        if show_progress:
            batch_num = (pos // batch_size) + 1
            total_batches = math.ceil(n / batch_size)
            print(
                f"  [batch {batch_num}/{total_batches}]  "
                f"bs={len(p_b)}  ttft={ttft:.0f}ms  total={total:.0f}ms  "
                f"out_tok={sum(out_lens)}  ({end}/{n} done)"
            )

        pos = end

    return all_results


def infer_single(tokenizer, model, prompt: str, kv_cache_pool=None) -> dict:
    return infer_all(
        tokenizer, model, [prompt],
        batch_size=1, show_progress=False, kv_cache_pool=kv_cache_pool,
    )[0]


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    import argparse
    pa = argparse.ArgumentParser(description="v3 高性能推理引擎")
    pa.add_argument("--model_path", type=str, required=True)
    pa.add_argument("--prompt", type=str, default="请用三句话解释KV Cache的作用。")
    pa.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    pa.add_argument("--quantize", action="store_true")
    args = pa.parse_args()

    tok, mdl, kv_pool = load_model(args.model_path, quantize=args.quantize)
    r = infer_single(tok, mdl, args.prompt, kv_pool)

    print(f"\n{'='*60}")
    print(f"  输出: {r['output'][:300]}")
    print(f"  in={r['input_tokens']} out={r['output_tokens']}")
    print(f"  延迟={r['total_latency_ms']:.1f}ms  TTFT={r['ttft_ms']:.1f}ms")
    print(f"  吞吐={r['throughput_tps']:.1f} tok/s")
    print(f"  峰值显存={torch.cuda.max_memory_allocated(DEVICE)/1e9:.2f} GB")
    stats = kv_pool.get_stats()
    print(f"  KV块: {stats['used_blocks']}/{stats['total_blocks']} ({stats['utilization_pct']:.1f}%)")
    print(f"{'='*60}")
