#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimized_inference.py
======================
Qwen2.5-14B-Instruct 高性能推理引擎

核心优化:
  1. PagedAttention — 基于 minivllm 的 paged KV-Cache，按需分配物理 block，零碎片
  2. Triton 融合算子 — store_kvcache / paged_attention_decode / RMSNorm / SwiGLU
  3. Prefill 用 HF SDPA → 拷贝 KV 到 paged cache → Decode 用 Triton PagedAttention
  4. 自适应 Batch Size — 自动探测显存上限
  5. 按长度排序分组 — 减少 padding 浪费
"""

import os, sys, time, math, sysconfig
from typing import List, Dict, Set
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================================
# 全局配置
# ============================================================================
DEVICE          = "cuda:0"
DTYPE           = torch.float16
MAX_NEW_TOKENS  = 256
BATCH_SIZE      = 32
SEED            = 42
PAGE_BLOCK_SIZE = 16  # 每个物理 block 存放的 token 数（同 minivllm）

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

# ============================================================================
# 修复 Triton 编译环境
# ============================================================================
def _inject_include(d):
    for v in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
        old = os.environ.get(v, "")
        if d not in old:
            os.environ[v] = f"{d}:{old}" if old else d

def _ensure_python_headers():
    sd = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(sd, "python3_include")
    if os.path.isfile(os.path.join(p, "Python.h")):
        _inject_include(p); return True
    inc = sysconfig.get_path("include")
    if inc and os.path.isfile(os.path.join(inc, "Python.h")):
        _inject_include(inc); return True
    for v in range(14, 6, -1):
        for pfx in ("/usr/include", "/usr/local/include"):
            d = f"{pfx}/python3.{v}"
            if os.path.isfile(os.path.join(d, "Python.h")):
                _inject_include(d); return True
    return False

_ensure_python_headers()

# ============================================================================
# Triton 导入 & 探测
# ============================================================================
_TRITON_IMPORTED = False
HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    _TRITON_IMPORTED = True
except ImportError:
    pass

def _probe_triton():
    global HAS_TRITON
    if not _TRITON_IMPORTED:
        HAS_TRITON = False; return
    try:
        @triton.jit
        def _t(X, B: tl.constexpr):
            i = tl.program_id(0) * B + tl.arange(0, B)
            tl.store(X + i, tl.load(X + i) + 1.0)
        x = torch.zeros(128, device=DEVICE, dtype=torch.float32)
        _t[(1,)](x, B=128); torch.cuda.synchronize(DEVICE)
        assert abs(x.sum().item() - 128.0) < 1e-3
        HAS_TRITON = True
        print(f"[INFO] Triton {triton.__version__} 可用，启用融合算子")
    except Exception as e:
        HAS_TRITON = False
        print(f"[WARN] Triton 不可用: {e}")

# ============================================================================
# Triton 融合算子
# ============================================================================
if _TRITON_IMPORTED:

    # ─── RMSNorm ───────────────────────────────────────────────────
    @triton.jit
    def _rms_norm_k(X, W, Y, sx, sy, N, eps, BLOCK: tl.constexpr):
        r = tl.program_id(0)
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for b in range(0, N, BLOCK):
            c = b + tl.arange(0, BLOCK); m = c < N
            xv = tl.load(X + r * sx + c, mask=m, other=0.0).to(tl.float32)
            acc += xv * xv
        rstd = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / N + eps)
        for b in range(0, N, BLOCK):
            c = b + tl.arange(0, BLOCK); m = c < N
            xv = tl.load(X + r * sx + c, mask=m, other=0.0).to(tl.float32)
            wv = tl.load(W + c, mask=m, other=1.0).to(tl.float32)
            tl.store(Y + r * sy + c, xv * rstd * wv, mask=m)

    def triton_rms_norm(x, w, eps=1e-6):
        s = x.shape; x2 = x.reshape(-1, s[-1]).contiguous()
        M, N = x2.shape; y = torch.empty_like(x2)
        BLK = min(triton.next_power_of_2(N), 4096)
        _rms_norm_k[(M,)](x2, w, y, x2.stride(0), y.stride(0), N, eps, BLOCK=BLK)
        return y.reshape(s)

    # ─── SiLU×Mul ──────────────────────────────────────────────────
    @triton.jit
    def _silu_mul_k(G, U, O, sg, su, so, N, BLOCK: tl.constexpr):
        r = tl.program_id(0)
        for b in range(0, N, BLOCK):
            c = b + tl.arange(0, BLOCK); m = c < N
            gv = tl.load(G + r * sg + c, mask=m, other=0.0).to(tl.float32)
            uv = tl.load(U + r * su + c, mask=m, other=0.0).to(tl.float32)
            tl.store(O + r * so + c, gv * tl.sigmoid(gv) * uv, mask=m)

    def triton_silu_mul(gate, up):
        s = gate.shape
        g2 = gate.reshape(-1, s[-1]).contiguous()
        u2 = up.reshape(-1, s[-1]).contiguous()
        M, N = g2.shape; o = torch.empty_like(g2)
        BLK = min(triton.next_power_of_2(N), 4096)
        _silu_mul_k[(M,)](g2, u2, o, g2.stride(0), u2.stride(0), o.stride(0), N, BLOCK=BLK)
        return o.reshape(s)

    # ─── Store KV Cache (来自 minivllm/attention.py) ───────────────
    @triton.jit
    def _store_kvcache_kernel(
        key_ptr, value_ptr,
        k_cache_ptr, v_cache_ptr,
        slot_mapping_ptr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
    ):
        """
        将 K/V 写入 paged cache。
        Grid: (num_tokens, num_kv_heads)
        Input:  K/V shape (num_tokens, num_kv_heads, head_dim)
        Cache:  shape (num_blocks, block_size, num_kv_heads, head_dim)
        """
        token_idx = tl.program_id(0)
        head_idx  = tl.program_id(1)

        slot_idx = tl.load(slot_mapping_ptr + token_idx)
        if slot_idx == -1:
            return

        block_idx    = slot_idx // block_size
        block_offset = slot_idx % block_size

        head_offsets = tl.arange(0, head_dim)

        # Input offset: (num_tokens, num_kv_heads, head_dim)
        input_offset = (token_idx * num_kv_heads * head_dim +
                        head_idx * head_dim +
                        head_offsets)

        # Cache offset: (num_blocks, block_size, num_kv_heads, head_dim)
        cache_offset = (block_idx * block_size * num_kv_heads * head_dim +
                        block_offset * num_kv_heads * head_dim +
                        head_idx * head_dim +
                        head_offsets)

        key   = tl.load(key_ptr + input_offset)
        value = tl.load(value_ptr + input_offset)
        tl.store(k_cache_ptr + cache_offset, key)
        tl.store(v_cache_ptr + cache_offset, value)

    # ─── Paged Attention Decode (来自 minivllm/attention.py) ───────
    @triton.jit
    def _paged_attn_decode_kernel(
        output_ptr, query_ptr,
        k_cache_ptr, v_cache_ptr,
        block_tables_ptr, context_lens_ptr,
        scale: tl.constexpr,
        num_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
        max_num_blocks: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """
        Decode 阶段 paged attention。
        Grid: (batch_size, num_heads)
        Query: (batch_size, num_heads, head_dim)
        Cache: (num_blocks, block_size, num_kv_heads, head_dim)
        """
        batch_idx = tl.program_id(0)
        head_idx  = tl.program_id(1)
        kv_head_idx = head_idx // (num_heads // num_kv_heads)

        context_len = tl.load(context_lens_ptr + batch_idx)

        offs_d = tl.arange(0, head_dim)
        q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
        q = tl.load(query_ptr + q_offset)

        acc = tl.zeros([head_dim], dtype=tl.float32)
        l_i = 0.0
        m_i = -1e10

        max_chunks = tl.cdiv(max_num_blocks * block_size, BLOCK_N)

        for chunk_idx in range(max_chunks):
            token_start = chunk_idx * BLOCK_N
            if token_start < context_len:
                offs_n = token_start + tl.arange(0, BLOCK_N)
                mask_n = offs_n < context_len

                qk = tl.zeros([BLOCK_N], dtype=tl.float32) - 1e10

                for i in range(BLOCK_N):
                    token_idx = token_start + i
                    if token_idx < context_len:
                        block_num    = token_idx // block_size
                        block_offset = token_idx % block_size
                        if block_num < max_num_blocks:
                            bt_offset = batch_idx * max_num_blocks + block_num
                            physical_block_idx = tl.load(block_tables_ptr + bt_offset)
                            if physical_block_idx != -1:
                                k_offset = (physical_block_idx * block_size * num_kv_heads * head_dim +
                                            block_offset * num_kv_heads * head_dim +
                                            kv_head_idx * head_dim + offs_d)
                                k_vec = tl.load(k_cache_ptr + k_offset)
                                score = tl.sum(q * k_vec) * scale
                                mask_i = tl.arange(0, BLOCK_N) == i
                                qk = tl.where(mask_i, score, qk)

                qk = tl.where(mask_n, qk, -1e10)

                m_ij = tl.max(qk)
                m_i_new = tl.maximum(m_i, m_ij)
                alpha = tl.exp(m_i - m_i_new)
                p = tl.exp(qk - m_i_new)

                acc = acc * alpha
                l_i = l_i * alpha

                for i in range(BLOCK_N):
                    token_idx = token_start + i
                    if token_idx < context_len:
                        block_num    = token_idx // block_size
                        block_offset = token_idx % block_size
                        if block_num < max_num_blocks:
                            bt_offset = batch_idx * max_num_blocks + block_num
                            physical_block_idx = tl.load(block_tables_ptr + bt_offset)
                            if physical_block_idx != -1:
                                v_offset = (physical_block_idx * block_size * num_kv_heads * head_dim +
                                            block_offset * num_kv_heads * head_dim +
                                            kv_head_idx * head_dim + offs_d)
                                v_vec = tl.load(v_cache_ptr + v_offset)
                                mask_i = tl.arange(0, BLOCK_N) == i
                                weight = tl.sum(tl.where(mask_i, p, 0.0))
                                acc = acc + weight * v_vec
                                l_i = l_i + weight

                m_i = m_i_new

        output = acc / l_i
        output_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
        tl.store(output_ptr + output_offset, output)


# ============================================================================
# PyTorch 原生实现（Triton 不可用时的 fallback）
# ============================================================================
def pt_rms_norm(x, w, eps=1e-6):
    xf = x.to(torch.float32)
    return ((xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)) * w.float()).to(x.dtype)

def pt_silu_mul(g, u):
    return F.silu(g) * u

def fused_rms_norm(x, w, eps=1e-6):
    return triton_rms_norm(x, w, eps) if (HAS_TRITON and x.is_cuda) else pt_rms_norm(x, w, eps)

def fused_silu_mul(g, u):
    return triton_silu_mul(g, u) if (HAS_TRITON and g.is_cuda) else pt_silu_mul(g, u)


# ============================================================================
# store_kvcache — Triton / PyTorch 双路径
# ============================================================================
def store_kvcache(key, value, k_cache, v_cache, slot_mapping, block_size):
    """
    将 K/V 写入 paged cache。
    key, value:    (num_tokens, num_kv_heads, head_dim)
    k/v_cache:     (num_blocks, block_size, num_kv_heads, head_dim)
    slot_mapping:  (num_tokens,) — 每个 token 在 cache 中的 flat 位置
    """
    num_tokens, num_kv_heads, head_dim = key.shape

    if HAS_TRITON and key.is_cuda:
        key   = key.contiguous()
        value = value.contiguous()
        grid  = (num_tokens, num_kv_heads)
        _store_kvcache_kernel[grid](
            key, value, k_cache, v_cache, slot_mapping,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
        )
    else:
        # PyTorch fallback
        for t in range(num_tokens):
            slot = slot_mapping[t].item()
            if slot == -1:
                continue
            blk = slot // block_size
            off = slot % block_size
            k_cache[blk, off, :, :] = key[t]
            v_cache[blk, off, :, :] = value[t]


# ============================================================================
# paged_attention_decode — Triton / PyTorch 双路径
# ============================================================================
def paged_attention_decode(query, k_cache, v_cache, block_tables, context_lens,
                           scale, num_heads, num_kv_heads, head_dim, block_size):
    """
    Decode 阶段 paged attention。
    query:        (batch_size, num_heads, head_dim)
    k/v_cache:    (num_blocks, block_size, num_kv_heads, head_dim)
    block_tables: (batch_size, max_num_blocks)  int32
    context_lens: (batch_size,)  int/long
    """
    batch_size = query.shape[0]
    max_num_blocks = block_tables.shape[1]

    if HAS_TRITON and query.is_cuda:
        query  = query.contiguous()
        output = torch.empty_like(query, dtype=torch.float32)
        BLOCK_N = 64 if head_dim <= 128 else 32
        grid = (batch_size, num_heads)
        _paged_attn_decode_kernel[grid](
            output, query, k_cache, v_cache,
            block_tables, context_lens,
            scale=scale,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            max_num_blocks=max_num_blocks,
            BLOCK_N=BLOCK_N,
        )
        return output.to(query.dtype)
    else:
        return _pt_paged_attn_decode(query, k_cache, v_cache, block_tables,
                                     context_lens, scale, num_heads,
                                     num_kv_heads, head_dim, block_size)


def _pt_paged_attn_decode(query, k_cache, v_cache, block_tables, context_lens,
                           scale, num_heads, num_kv_heads, head_dim, block_size):
    """PyTorch 参考实现"""
    B = query.shape[0]
    num_q_per_kv = num_heads // num_kv_heads
    o = torch.zeros_like(query, dtype=torch.float32)
    for i in range(B):
        ctx = context_lens[i].item()
        n_blks = math.ceil(ctx / block_size)
        k_list, v_list = [], []
        for b in range(n_blks):
            pb = block_tables[i, b].long().item()
            if pb == -1:
                continue
            start = b * block_size
            length = min(block_size, ctx - start)
            # cache: (num_blocks, block_size, num_kv_heads, head_dim)
            k_list.append(k_cache[pb, :length, :, :])  # (length, nkvh, hd)
            v_list.append(v_cache[pb, :length, :, :])
        k_seq = torch.cat(k_list, dim=0)  # (ctx, nkvh, hd)
        v_seq = torch.cat(v_list, dim=0)
        # → (nkvh, ctx, hd)
        k_seq = k_seq.permute(1, 0, 2)
        v_seq = v_seq.permute(1, 0, 2)
        # GQA expand
        if num_q_per_kv > 1:
            k_seq = k_seq.unsqueeze(1).expand(-1, num_q_per_kv, -1, -1).reshape(num_heads, ctx, head_dim)
            v_seq = v_seq.unsqueeze(1).expand(-1, num_q_per_kv, -1, -1).reshape(num_heads, ctx, head_dim)
        qi = query[i].float()  # (num_heads, head_dim)
        scores = torch.einsum('hd,hsd->hs', qi, k_seq.float()) * scale
        o[i] = torch.einsum('hs,hsd->hd', F.softmax(scores, dim=-1), v_seq.float())
    return o.to(query.dtype)


# ============================================================================
# PagedKVCache — 物理 block 池管理器
# ============================================================================
class PagedKVCache:
    """
    基于 minivllm/block_manager.py 的 KV cache 管理。

    物理存储 (每层独立):
      k_pool: (max_blocks, block_size, num_kv_heads, head_dim)
      v_pool: (max_blocks, block_size, num_kv_heads, head_dim)

    逻辑映射:
      page_tables[seq_idx] = [phys_block_0, phys_block_1, ...]
      seq_lens[seq_idx]    = 已存储的 token 总数
    """

    def __init__(self, num_layers, num_kv_heads, head_dim, block_size,
                 max_blocks, device, dtype):
        self.num_layers   = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim     = head_dim
        self.block_size   = block_size
        self.max_blocks   = max_blocks
        self.device       = device
        self.dtype        = dtype

        # 预分配物理 block 池 — 同 minivllm/model_runner.py allocate_kv_cache
        pool_shape = (max_blocks, block_size, num_kv_heads, head_dim)
        self.k_pools = [torch.zeros(pool_shape, dtype=dtype, device=device)
                        for _ in range(num_layers)]
        self.v_pools = [torch.zeros(pool_shape, dtype=dtype, device=device)
                        for _ in range(num_layers)]

        # 空闲 block 栈 — 同 minivllm/block_manager.py
        self.free_blocks: deque = deque(range(max_blocks))
        self.page_tables: Dict[int, List[int]] = {}
        self.seq_lens: Dict[int, int] = {}

    @property
    def num_free(self):
        return len(self.free_blocks)

    def _alloc_block(self) -> int:
        if not self.free_blocks:
            raise RuntimeError(f"PagedKVCache: 物理 block 用尽！"
                               f"(max={self.max_blocks}, bs={self.block_size})")
        return self.free_blocks.popleft()

    def allocate_seq(self, seq_idx: int, num_tokens: int):
        """为序列分配足够的 block（同 minivllm block_manager.allocate）"""
        n_blks = max(1, math.ceil(num_tokens / self.block_size))
        blocks = [self._alloc_block() for _ in range(n_blks)]
        self.page_tables[seq_idx] = blocks
        self.seq_lens[seq_idx] = 0

    def ensure_slot(self, seq_idx: int):
        """
        确保当前 seq_lens[seq_idx] 位置有可用 slot。
        如果需要新 block 则分配（同 minivllm block_manager.append）。
        返回 slot index。
        """
        pos = self.seq_lens[seq_idx]
        blk_idx = pos // self.block_size
        offset  = pos % self.block_size
        if blk_idx >= len(self.page_tables[seq_idx]):
            self.page_tables[seq_idx].append(self._alloc_block())
        pb = self.page_tables[seq_idx][blk_idx]
        return pb * self.block_size + offset

    def increment_seq_len(self, seq_idx: int):
        self.seq_lens[seq_idx] += 1

    def free_seq(self, seq_idx: int):
        if seq_idx in self.page_tables:
            self.free_blocks.extend(self.page_tables[seq_idx])
            del self.page_tables[seq_idx]
            del self.seq_lens[seq_idx]

    def get_block_table_tensor(self, seq_ids: List[int]) -> torch.Tensor:
        max_blks = max(len(self.page_tables[s]) for s in seq_ids)
        tbl = torch.full((len(seq_ids), max_blks), -1, dtype=torch.int32, device=self.device)
        for i, s in enumerate(seq_ids):
            pt = self.page_tables[s]
            tbl[i, :len(pt)] = torch.tensor(pt, dtype=torch.int32)
        return tbl

    def get_context_lens_tensor(self, seq_ids: List[int]) -> torch.Tensor:
        return torch.tensor([self.seq_lens[s] for s in seq_ids],
                            dtype=torch.long, device=self.device)


# ============================================================================
# Rotary Position Embedding 辅助函数
# ============================================================================
def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos + _rotate_half(q) * sin,
            k * cos + _rotate_half(k) * sin)


# ============================================================================
# Prefill KV → Paged Cache 拷贝（使用 store_kvcache Triton kernel）
# ============================================================================
def _copy_prefill_kv_to_paged(hf_past, paged_cache: PagedKVCache,
                               input_lengths: List[int], padded_len: int):
    """
    将 HF past_key_values 拷贝到 PagedKVCache。
    利用 store_kvcache Triton kernel 做高效并行写入。
    """
    B = len(input_lengths)
    device = paged_cache.device
    block_size = paged_cache.block_size

    for layer_idx in range(paged_cache.num_layers):
        # HF KV: (B, num_kv_heads, padded_len, head_dim)
        K_layer = hf_past[layer_idx][0]
        V_layer = hf_past[layer_idx][1]

        # 收集所有 token 的 KV 和 slot_mapping
        all_k, all_v, all_slots = [], [], []
        for i in range(B):
            actual_len = input_lengths[i]
            # left-padding: 实际内容在后面
            k_actual = K_layer[i, :, padded_len - actual_len:, :]  # (nkvh, actual_len, hd)
            v_actual = V_layer[i, :, padded_len - actual_len:, :]

            # → (actual_len, nkvh, hd)  匹配 store_kvcache 的输入格式
            all_k.append(k_actual.permute(1, 0, 2).contiguous())
            all_v.append(v_actual.permute(1, 0, 2).contiguous())

            # 计算 slot_mapping
            pt = paged_cache.page_tables[i]
            for j in range(actual_len):
                blk_idx = j // block_size
                offset  = j % block_size
                slot = pt[blk_idx] * block_size + offset
                all_slots.append(slot)

        # 拼接后走一次 store_kvcache
        cat_k = torch.cat(all_k, dim=0)  # (total_tokens, nkvh, hd)
        cat_v = torch.cat(all_v, dim=0)
        slots = torch.tensor(all_slots, dtype=torch.long, device=device)

        store_kvcache(cat_k, cat_v,
                      paged_cache.k_pools[layer_idx],
                      paged_cache.v_pools[layer_idx],
                      slots, block_size)

    # 更新 seq_lens
    for i in range(B):
        paged_cache.seq_lens[i] = input_lengths[i]


# ============================================================================
# 自定义 Decode Step — PagedAttention
# ============================================================================
@torch.inference_mode()
def paged_decode_step(
    model, token_ids: torch.Tensor, position_ids: torch.Tensor,
    paged_cache: PagedKVCache, seq_ids: List[int],
    num_q_heads: int, num_kv_heads: int, head_dim: int,
):
    """
    用 PagedAttention 执行一步 decode。
    流程同 minivllm Attention.forward decode 分支。

    token_ids:    (B, 1)
    position_ids: (B, 1)
    返回:         logits (B, vocab_size)
    """
    B = token_ids.shape[0]
    device = token_ids.device
    scale = 1.0 / math.sqrt(head_dim)
    block_size = paged_cache.block_size

    # ─── Step 1: 计算 slot_mapping（新 token 要写入的位置）───
    slot_list = []
    for sid in seq_ids:
        slot_list.append(paged_cache.ensure_slot(sid))
    slot_mapping = torch.tensor(slot_list, dtype=torch.long, device=device)

    # ─── Step 2: Embedding ───
    hidden = model.model.embed_tokens(token_ids)  # (B, 1, hidden_dim)

    # ─── Step 3: 逐层 Transformer ───
    for layer_idx, layer in enumerate(model.model.layers):
        residual = hidden

        # Input LayerNorm
        h = layer.input_layernorm(hidden)

        # QKV Projection
        q = layer.self_attn.q_proj(h)
        k = layer.self_attn.k_proj(h)
        v = layer.self_attn.v_proj(h)

        # Reshape: (B, 1, dim) → (B, num_heads, 1, head_dim)
        q = q.view(B, 1, num_q_heads,  head_dim).transpose(1, 2)
        k = k.view(B, 1, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(B, 1, num_kv_heads, head_dim).transpose(1, 2)

        # RoPE
        cos, sin = model.model.rotary_emb(q, position_ids)
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        # ─── Store K/V into Paged Cache（同 minivllm Attention.forward 中的 store_kvcache）───
        # k: (B, nkvh, 1, hd) → squeeze → (B, nkvh, hd) = (num_tokens, nkvh, hd)
        k_store = k.squeeze(2).contiguous()
        v_store = v.squeeze(2).contiguous()
        store_kvcache(k_store, v_store,
                      paged_cache.k_pools[layer_idx],
                      paged_cache.v_pools[layer_idx],
                      slot_mapping, block_size)

        # ─── Paged Attention Decode（同 minivllm Attention.forward 中的 paged_attention_decode）───
        # context_lens = 已存储 token 数 + 1（刚写入的）
        ctx_lens = paged_cache.get_context_lens_tensor(seq_ids) + 1
        blk_tables = paged_cache.get_block_table_tensor(seq_ids)

        q_attn = q.squeeze(2)  # (B, num_q_heads, head_dim)
        attn_out = paged_attention_decode(
            q_attn,
            paged_cache.k_pools[layer_idx],
            paged_cache.v_pools[layer_idx],
            blk_tables, ctx_lens,
            scale, num_q_heads, num_kv_heads, head_dim, block_size,
        )  # (B, num_q_heads, head_dim)

        # → (B, 1, num_q_heads * head_dim) for output projection
        attn_out = attn_out.unsqueeze(1)
        attn_out = attn_out.reshape(B, 1, num_q_heads * head_dim)
        attn_out = layer.self_attn.o_proj(attn_out)

        hidden = residual + attn_out

        # Post-attention LayerNorm + MLP
        residual = hidden
        h = layer.post_attention_layernorm(hidden)
        hidden = residual + layer.mlp(h)

    # ─── Step 4: 所有层完成，推进 seq_lens ───
    for sid in seq_ids:
        paged_cache.increment_seq_len(sid)

    # ─── Step 5: Final Norm + LM Head ───
    hidden = model.model.norm(hidden)
    logits = model.lm_head(hidden[:, -1, :])  # (B, vocab)
    return logits


# ============================================================================
# EOS Token 集合
# ============================================================================
def _eos_ids(tokenizer) -> Set[int]:
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
# 批量生成 — PagedAttention 版本
# ============================================================================
@torch.inference_mode()
def batch_generate_paged(model, tokenizer, prompts: List[str],
                         max_new_tokens: int = MAX_NEW_TOKENS):
    """
    完整生成流水线:
      Phase 1: HF 模型 prefill → past_key_values（SDPA 高效处理长 prompt）
      Phase 2: 拷贝 KV 到 PagedKVCache（Triton store_kvcache kernel）
      Phase 3: 自定义 Decode 循环（Triton PagedAttention kernel）
    """
    device  = torch.device(DEVICE)
    pad_id  = tokenizer.pad_token_id or 0
    eos_ids = _eos_ids(tokenizer)
    eos_t   = torch.tensor(sorted(eos_ids), dtype=torch.long, device=device)
    B       = len(prompts)

    # 模型配置
    cfg = model.config
    num_layers   = cfg.num_hidden_layers
    num_q_heads  = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, 'num_key_value_heads', num_q_heads)
    head_dim     = cfg.hidden_size // num_q_heads

    # ─── Tokenize（left-padding）───
    enc = tokenizer(prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=2048).to(device)
    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    input_lengths  = attention_mask.sum(dim=1).tolist()
    padded_len     = input_ids.shape[1]

    # ─── Phase 1: Prefill（HF SDPA）───
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )

    torch.cuda.synchronize(device)
    ttft = (time.perf_counter() - t0) * 1000.0

    # ─── Phase 2: 初始化 PagedKVCache & 拷贝 ───
    max_total_tokens = max(input_lengths) + max_new_tokens + 8
    max_blocks_per_seq = math.ceil(max_total_tokens / PAGE_BLOCK_SIZE) + 2
    total_max_blocks = max_blocks_per_seq * B + 64

    paged_cache = PagedKVCache(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=PAGE_BLOCK_SIZE,
        max_blocks=total_max_blocks,
        device=device,
        dtype=DTYPE,
    )

    seq_ids = list(range(B))
    for i in range(B):
        paged_cache.allocate_seq(i, input_lengths[i])

    # 用 Triton store_kvcache 高效拷贝 HF KV → paged cache
    _copy_prefill_kv_to_paged(out.past_key_values, paged_cache,
                               input_lengths, padded_len)

    # 释放 HF KV cache，节省显存
    first_logits = out.logits[:, -1, :].clone()
    del out
    torch.cuda.empty_cache()

    # ─── Phase 3: Decode 循环（PagedAttention）───
    first_tokens = first_logits.argmax(dim=-1)  # (B,)
    del first_logits

    unfinished     = torch.ones(B, dtype=torch.bool, device=device)
    generated      = [first_tokens.unsqueeze(1)]
    sample_lengths = torch.zeros(B, dtype=torch.long, device=device)
    cur_tokens     = first_tokens

    # 检查第一个 token 是否是 EOS
    is_eos = (cur_tokens.unsqueeze(1) == eos_t.unsqueeze(0)).any(dim=1)
    sample_lengths += (unfinished & ~is_eos).long()
    unfinished = unfinished & ~is_eos

    # 位置追踪：下一个 decode 的 position
    positions = torch.tensor(input_lengths, dtype=torch.long, device=device)

    for step in range(1, max_new_tokens):
        if not unfinished.any():
            break

        pos_ids = positions.unsqueeze(1)  # (B, 1)

        logits = paged_decode_step(
            model, cur_tokens.unsqueeze(1), pos_ids,
            paged_cache, seq_ids,
            num_q_heads, num_kv_heads, head_dim,
        )

        next_tokens = logits.argmax(dim=-1)
        next_tokens = torch.where(unfinished, next_tokens,
                                  torch.full_like(next_tokens, pad_id))

        is_eos = (next_tokens.unsqueeze(1) == eos_t.unsqueeze(0)).any(dim=1)
        sample_lengths += (unfinished & ~is_eos).long()
        generated.append(next_tokens.unsqueeze(1))

        unfinished = unfinished & ~is_eos
        positions += 1
        cur_tokens = next_tokens

    torch.cuda.synchronize(device)
    total_ms = (time.perf_counter() - t0) * 1000.0

    # ─── 收集结果 ───
    gen_ids = torch.cat(generated, dim=1)
    lengths = sample_lengths.tolist()
    texts = []
    for i in range(B):
        L = min(max(lengths[i], 0), gen_ids.shape[1])
        texts.append(
            tokenizer.decode(gen_ids[i, :L].tolist(), skip_special_tokens=True)
            if L > 0 else ""
        )

    # 释放 paged cache
    for i in range(B):
        paged_cache.free_seq(i)

    return texts, lengths, input_lengths, ttft, total_ms


# ============================================================================
# 标准批量生成（备选，不用 PagedAttention）
# ============================================================================
@torch.inference_mode()
def batch_generate_standard(model, tokenizer, prompts, max_new_tokens=MAX_NEW_TOKENS):
    device  = torch.device(DEVICE)
    pad_id  = tokenizer.pad_token_id or 0
    eos_ids = _eos_ids(tokenizer)
    eos_t   = torch.tensor(sorted(eos_ids), dtype=torch.long, device=device)
    B       = len(prompts)

    enc = tokenizer(prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=2048).to(device)
    input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]
    input_lengths = attention_mask.sum(dim=1).tolist()

    unfinished = torch.ones(B, dtype=torch.bool, device=device)
    generated, sample_lengths = [], torch.zeros(B, dtype=torch.long, device=device)
    past, cur_ids, cur_mask = None, input_ids, attention_mask

    torch.cuda.synchronize(device); t0 = time.perf_counter(); ttft = None
    for step in range(max_new_tokens):
        out = model(input_ids=cur_ids, attention_mask=cur_mask,
                    past_key_values=past, use_cache=True, return_dict=True)
        logits = out.logits[:, -1, :]; past = out.past_key_values
        next_tok = logits.argmax(dim=-1)
        if step == 0:
            torch.cuda.synchronize(device)
            ttft = (time.perf_counter() - t0) * 1000.0
        next_tok = torch.where(unfinished, next_tok, torch.full_like(next_tok, pad_id))
        is_eos = (next_tok.unsqueeze(1) == eos_t.unsqueeze(0)).any(dim=1)
        sample_lengths += (unfinished & ~is_eos).long()
        generated.append(next_tok.unsqueeze(1))
        unfinished = unfinished & ~is_eos
        if not unfinished.any():
            break
        cur_ids = next_tok.unsqueeze(1)
        cur_mask = torch.cat([cur_mask, torch.ones(B, 1, device=device, dtype=cur_mask.dtype)], dim=1)

    torch.cuda.synchronize(device)
    total_ms = (time.perf_counter() - t0) * 1000.0
    if ttft is None:
        ttft = total_ms
    gen_ids = torch.cat(generated, dim=1) if generated else torch.zeros(B, 0, dtype=torch.long, device=device)
    lengths = sample_lengths.tolist()
    texts = []
    for i in range(B):
        L = min(max(lengths[i], 0), gen_ids.shape[1])
        texts.append(tokenizer.decode(gen_ids[i, :L].tolist(), skip_special_tokens=True) if L > 0 else "")
    return texts, lengths, input_lengths, ttft, total_ms


# ============================================================================
# 自适应 Batch Size
# ============================================================================
def _auto_batch_size(model, tokenizer, initial_bs=64):
    device = torch.device(DEVICE)
    dummy = "你好世界，请简要回答。"
    best = 1; lo, hi = 1, initial_bs
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            torch.cuda.empty_cache()
            enc = tokenizer([dummy] * mid, return_tensors="pt", padding=True,
                            truncation=True, max_length=128).to(device)
            with torch.inference_mode():
                o = model(input_ids=enc["input_ids"],
                          attention_mask=enc["attention_mask"],
                          use_cache=True, return_dict=True)
                p = o.past_key_values
                ci = o.logits[:, -1, :].argmax(-1).unsqueeze(1)
                cm = torch.cat([enc["attention_mask"],
                                torch.ones(mid, 1, device=device, dtype=torch.long)], dim=1)
                for _ in range(8):
                    o2 = model(input_ids=ci, attention_mask=cm,
                               past_key_values=p, use_cache=True, return_dict=True)
                    p = o2.past_key_values
                    ci = o2.logits[:, -1, :].argmax(-1).unsqueeze(1)
                    cm = torch.cat([cm, torch.ones(mid, 1, device=device, dtype=torch.long)], dim=1)
            del o, o2, p, ci, cm
            torch.cuda.synchronize(device)
            best = mid; lo = mid + 1
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache(); hi = mid - 1
        except Exception:
            hi = mid - 1
    torch.cuda.empty_cache()
    return max(1, int(best * 0.85))


# ============================================================================
# Monkey-patch（RMSNorm + SwiGLU Triton 融合）
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
    nr = nm = 0
    for _, m in model.named_modules():
        cn = type(m).__name__
        if "RMSNorm" in cn:
            m.forward = _make_rmsnorm_fwd(m); nr += 1
        if "MLP" in cn and hasattr(m, "gate_proj"):
            m.forward = _make_mlp_fwd(m); nm += 1
    tag = "Triton" if HAS_TRITON else "PyTorch"
    print(f"[OPT] {tag} 融合 RMSNorm×{nr}, SwiGLU×{nm}")


# ============================================================================
# 加载模型
# ============================================================================
def load_model(model_path: str):
    print(f"[INFO] 加载模型: {model_path}")
    print(f"[INFO] 设备={DEVICE}  精度={DTYPE}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None
    for attn in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=DTYPE, device_map=DEVICE,
                trust_remote_code=True, attn_implementation=attn)
            print(f"[OPT] Attention: {attn}")
            break
        except Exception:
            continue
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=DTYPE, device_map=DEVICE, trust_remote_code=True)
    model.eval()

    _probe_triton()
    apply_optimizations(model)

    print("[INFO] 预热推理...")
    w = tokenizer("hello world", return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        for _ in range(3):
            model(**w, use_cache=True)
    torch.cuda.synchronize(DEVICE)

    n_p = sum(p.numel() for p in model.parameters()) / 1e9
    vram = torch.cuda.memory_allocated(DEVICE) / 1e9
    print(f"[INFO] 就绪 | {n_p:.1f}B params | VRAM {vram:.2f} GB")
    return tokenizer, model


# ============================================================================
# 高层 API
# ============================================================================
def infer_all(tokenizer, model, prompts: list,
              batch_size: int = BATCH_SIZE,
              max_new_tokens: int = MAX_NEW_TOKENS,
              show_progress: bool = True,
              use_paged: bool = True):
    n = len(prompts)
    if n == 0:
        return []

    # 按 prompt 长度排序，减少 padding
    enc_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    sorted_idx = sorted(range(n), key=lambda i: enc_lens[i])

    all_results = [None] * n
    num_batches = math.ceil(n / batch_size)
    gen_fn = batch_generate_paged if use_paged else batch_generate_standard

    for b in range(num_batches):
        s = b * batch_size
        e = min(s + batch_size, n)
        idx_b = sorted_idx[s:e]
        p_b = [prompts[i] for i in idx_b]

        texts, out_lens, in_lens, ttft, total = gen_fn(
            model, tokenizer, p_b, max_new_tokens)

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
            print(f"  [batch {b+1}/{num_batches}]  bs={len(p_b)}  ttft={ttft:.0f}ms  "
                  f"total={total:.0f}ms  out_tok={sum(out_lens)}  ({e}/{n} done)")

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
    pa.add_argument("--batch_size", type=int, default=-1)
    pa.add_argument("--no_paged", action="store_true")
    args = pa.parse_args()

    tok, mdl = load_model(args.model_path)
    if args.batch_size <= 0:
        args.batch_size = _auto_batch_size(mdl, tok)
        print(f"[INFO] auto batch_size = {args.batch_size}")

    r = infer_single(tok, mdl, args.prompt)
    print(f"\n{'='*60}")
    print(f"  输出: {r['output'][:300]}")
    print(f"  in={r['input_tokens']} out={r['output_tokens']}")
    print(f"  延迟={r['total_latency_ms']:.1f}ms  TTFT={r['ttft_ms']:.1f}ms")
    print(f"  吞吐={r['throughput_tps']:.1f} tok/s")
    print(f"  峰值显存={torch.cuda.max_memory_allocated(DEVICE)/1e9:.2f} GB")
    print(f"{'='*60}")