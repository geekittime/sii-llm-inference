#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimized_inference.py
======================
Qwen2.5-14B-Instruct 统一 PagedAttention 推理引擎

特性:
  1. Prefill 与 Decode 都接入 paged KV cache
  2. Prefill 采用 chunked extend:
       - prefix paged attention
       - local causal attention
       - LSE merge
  3. Decode 使用 paged attention
  4. Triton 融合 RMSNorm / SwiGLU / store_kvcache / paged_decode
  5. 按长度排序分组
  6. 先保证结构正确，方便后续继续把 prefix prefill Triton 化
"""

import os
import sys
import time
import math
import sysconfig
from typing import List, Dict, Set
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================================
# 全局配置
# ============================================================================
DEVICE = "cuda:0"
DTYPE = torch.float16
MAX_NEW_TOKENS = 256
BATCH_SIZE = 32
SEED = 42
PAGE_BLOCK_SIZE = 16
PREFILL_CHUNK_SIZE = 128

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


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
        _inject_include(p)
        return True

    inc = sysconfig.get_path("include")
    if inc and os.path.isfile(os.path.join(inc, "Python.h")):
        _inject_include(inc)
        return True

    for v in range(14, 6, -1):
        for pfx in ("/usr/include", "/usr/local/include"):
            d = f"{pfx}/python3.{v}"
            if os.path.isfile(os.path.join(d, "Python.h")):
                _inject_include(d)
                return True
    return False


_ensure_python_headers()

# ============================================================================
# Triton 导入
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
        HAS_TRITON = False
        print("[INFO] Triton 未安装，使用 PyTorch fallback")
        return

    try:
        @triton.jit
        def _t(X, B: tl.constexpr):
            i = tl.program_id(0) * B + tl.arange(0, B)
            tl.store(X + i, tl.load(X + i) + 1.0)

        x = torch.zeros(128, device=DEVICE, dtype=torch.float32)
        _t[(1,)](x, B=128)
        torch.cuda.synchronize(DEVICE)
        assert abs(x.sum().item() - 128.0) < 1e-3
        HAS_TRITON = True
        print(f"[INFO] Triton {triton.__version__} 可用")
    except Exception as e:
        HAS_TRITON = False
        print(f"[WARN] Triton 不可用: {e}")
        print("[INFO] 回退到 PyTorch fallback")


# ============================================================================
# Triton 融合算子
# ============================================================================
if _TRITON_IMPORTED:
    @triton.jit
    def _rms_norm_k(X, W, Y, sx, sy, N, eps, BLOCK: tl.constexpr):
        r = tl.program_id(0)
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for b in range(0, N, BLOCK):
            c = b + tl.arange(0, BLOCK)
            m = c < N
            xv = tl.load(X + r * sx + c, mask=m, other=0.0).to(tl.float32)
            acc += xv * xv
        rstd = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / N + eps)
        for b in range(0, N, BLOCK):
            c = b + tl.arange(0, BLOCK)
            m = c < N
            xv = tl.load(X + r * sx + c, mask=m, other=0.0).to(tl.float32)
            wv = tl.load(W + c, mask=m, other=1.0).to(tl.float32)
            tl.store(Y + r * sy + c, xv * rstd * wv, mask=m)

    def triton_rms_norm(x, w, eps=1e-6):
        s = x.shape
        x2 = x.reshape(-1, s[-1]).contiguous()
        M, N = x2.shape
        y = torch.empty_like(x2)
        BLK = min(triton.next_power_of_2(N), 4096)
        _rms_norm_k[(M,)](x2, w, y, x2.stride(0), y.stride(0), N, eps, BLOCK=BLK)
        return y.reshape(s)

    @triton.jit
    def _silu_mul_k(G, U, O, sg, su, so, N, BLOCK: tl.constexpr):
        r = tl.program_id(0)
        for b in range(0, N, BLOCK):
            c = b + tl.arange(0, BLOCK)
            m = c < N
            gv = tl.load(G + r * sg + c, mask=m, other=0.0).to(tl.float32)
            uv = tl.load(U + r * su + c, mask=m, other=0.0).to(tl.float32)
            tl.store(O + r * so + c, gv * tl.sigmoid(gv) * uv, mask=m)

    def triton_silu_mul(gate, up):
        s = gate.shape
        g2 = gate.reshape(-1, s[-1]).contiguous()
        u2 = up.reshape(-1, s[-1]).contiguous()
        M, N = g2.shape
        o = torch.empty_like(g2)
        BLK = min(triton.next_power_of_2(N), 4096)
        _silu_mul_k[(M,)](g2, u2, o, g2.stride(0), u2.stride(0), o.stride(0), N, BLOCK=BLK)
        return o.reshape(s)

    @triton.jit
    def _store_kvcache_kernel(
        key_ptr, value_ptr,
        k_cache_ptr, v_cache_ptr,
        slot_mapping_ptr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
    ):
        token_idx = tl.program_id(0)
        head_idx = tl.program_id(1)

        slot_idx = tl.load(slot_mapping_ptr + token_idx)
        if slot_idx == -1:
            return

        block_idx = slot_idx // block_size
        block_offset = slot_idx % block_size
        head_offsets = tl.arange(0, head_dim)

        input_offset = (
            token_idx * num_kv_heads * head_dim +
            head_idx * head_dim +
            head_offsets
        )

        cache_offset = (
            block_idx * block_size * num_kv_heads * head_dim +
            block_offset * num_kv_heads * head_dim +
            head_idx * head_dim +
            head_offsets
        )

        key = tl.load(key_ptr + input_offset)
        value = tl.load(value_ptr + input_offset)
        tl.store(k_cache_ptr + cache_offset, key)
        tl.store(v_cache_ptr + cache_offset, value)

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
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
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
                        block_num = token_idx // block_size
                        block_offset = token_idx % block_size
                        if block_num < max_num_blocks:
                            bt_offset = batch_idx * max_num_blocks + block_num
                            physical_block_idx = tl.load(block_tables_ptr + bt_offset)
                            if physical_block_idx != -1:
                                k_offset = (
                                    physical_block_idx * block_size * num_kv_heads * head_dim +
                                    block_offset * num_kv_heads * head_dim +
                                    kv_head_idx * head_dim + offs_d
                                )
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
                        block_num = token_idx // block_size
                        block_offset = token_idx % block_size
                        if block_num < max_num_blocks:
                            bt_offset = batch_idx * max_num_blocks + block_num
                            physical_block_idx = tl.load(block_tables_ptr + bt_offset)
                            if physical_block_idx != -1:
                                v_offset = (
                                    physical_block_idx * block_size * num_kv_heads * head_dim +
                                    block_offset * num_kv_heads * head_dim +
                                    kv_head_idx * head_dim + offs_d
                                )
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
# PyTorch fallback
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
# Monkey-patch
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
            m.forward = _make_rmsnorm_fwd(m)
            nr += 1
        if "MLP" in cn and hasattr(m, "gate_proj"):
            m.forward = _make_mlp_fwd(m)
            nm += 1
    tag = "Triton" if HAS_TRITON else "PyTorch"
    print(f"[OPT] {tag} 融合 RMSNorm×{nr}, SwiGLU×{nm}")


# ============================================================================
# EOS 集合
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
# Paged KV Cache
# ============================================================================
class PagedKVCache:
    def __init__(self, num_layers, num_kv_heads, head_dim, block_size,
                 max_blocks, device, dtype):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.device = device
        self.dtype = dtype

        pool_shape = (max_blocks, block_size, num_kv_heads, head_dim)
        self.k_pools = [torch.zeros(pool_shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self.v_pools = [torch.zeros(pool_shape, dtype=dtype, device=device) for _ in range(num_layers)]

        self.free_blocks: deque = deque(range(max_blocks))
        self.page_tables: Dict[int, List[int]] = {}
        self.seq_lens: Dict[int, int] = {}

    def _alloc_block(self):
        if not self.free_blocks:
            raise RuntimeError("PagedKVCache: 物理 block 用尽")
        return self.free_blocks.popleft()

    def allocate_seq(self, seq_idx: int, init_tokens: int = 0):
        n_blks = max(1, math.ceil(max(init_tokens, 1) / self.block_size))
        self.page_tables[seq_idx] = [self._alloc_block() for _ in range(n_blks)]
        self.seq_lens[seq_idx] = 0

    def ensure_slots_for_append(self, seq_idx: int, num_new_tokens: int):
        need_total = self.seq_lens[seq_idx] + num_new_tokens
        need_blocks = math.ceil(max(need_total, 1) / self.block_size)
        while len(self.page_tables[seq_idx]) < need_blocks:
            self.page_tables[seq_idx].append(self._alloc_block())

    def get_slot_mapping_for_append(self, seq_idx: int, num_new_tokens: int):
        self.ensure_slots_for_append(seq_idx, num_new_tokens)
        start = self.seq_lens[seq_idx]
        slots = []
        for j in range(num_new_tokens):
            pos = start + j
            blk_idx = pos // self.block_size
            blk_off = pos % self.block_size
            pb = self.page_tables[seq_idx][blk_idx]
            slots.append(pb * self.block_size + blk_off)
        return slots

    def append_len(self, seq_idx: int, n: int):
        self.seq_lens[seq_idx] += n

    def get_block_table_tensor(self, seq_ids: List[int]):
        max_blks = max(len(self.page_tables[s]) for s in seq_ids)
        tbl = torch.full((len(seq_ids), max_blks), -1, dtype=torch.int32, device=self.device)
        for i, s in enumerate(seq_ids):
            pt = self.page_tables[s]
            tbl[i, :len(pt)] = torch.tensor(pt, dtype=torch.int32, device=self.device)
        return tbl

    def get_context_lens_tensor(self, seq_ids: List[int]):
        return torch.tensor([self.seq_lens[s] for s in seq_ids], dtype=torch.long, device=self.device)

    def free_seq(self, seq_idx: int):
        if seq_idx in self.page_tables:
            self.free_blocks.extend(self.page_tables[seq_idx])
            del self.page_tables[seq_idx]
            del self.seq_lens[seq_idx]


# ============================================================================
# Store KV Cache
# ============================================================================
def store_kvcache(key, value, k_cache, v_cache, slot_mapping, block_size):
    num_tokens, num_kv_heads, head_dim = key.shape

    if HAS_TRITON and key.is_cuda:
        key = key.contiguous()
        value = value.contiguous()
        slot_mapping = slot_mapping.contiguous()
        grid = (num_tokens, num_kv_heads)
        _store_kvcache_kernel[grid](
            key, value, k_cache, v_cache, slot_mapping,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
        )
    else:
        for t in range(num_tokens):
            slot = int(slot_mapping[t].item())
            if slot == -1:
                continue
            blk = slot // block_size
            off = slot % block_size
            k_cache[blk, off, :, :] = key[t]
            v_cache[blk, off, :, :] = value[t]


# ============================================================================
# Decode PagedAttention
# ============================================================================
def paged_attention_decode(query, k_cache, v_cache, block_tables, context_lens,
                           scale, num_heads, num_kv_heads, head_dim, block_size):
    batch_size = query.shape[0]
    max_num_blocks = block_tables.shape[1]

    if HAS_TRITON and query.is_cuda:
        query = query.contiguous()
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
        return _pt_paged_attn_decode(
            query, k_cache, v_cache, block_tables,
            context_lens, scale, num_heads,
            num_kv_heads, head_dim, block_size
        )


def _pt_paged_attn_decode(query, k_cache, v_cache, block_tables, context_lens,
                          scale, num_heads, num_kv_heads, head_dim, block_size):
    B = query.shape[0]
    num_q_per_kv = num_heads // num_kv_heads
    o = torch.zeros_like(query, dtype=torch.float32)

    for i in range(B):
        ctx = int(context_lens[i].item())
        n_blks = math.ceil(ctx / block_size)
        k_list, v_list = [], []
        for b in range(n_blks):
            pb = int(block_tables[i, b].item())
            if pb == -1:
                continue
            start = b * block_size
            length = min(block_size, ctx - start)
            k_list.append(k_cache[pb, :length, :, :])
            v_list.append(v_cache[pb, :length, :, :])

        k_seq = torch.cat(k_list, dim=0)
        v_seq = torch.cat(v_list, dim=0)

        k_seq = k_seq.permute(1, 0, 2)
        v_seq = v_seq.permute(1, 0, 2)

        if num_q_per_kv > 1:
            k_seq = k_seq.unsqueeze(1).expand(-1, num_q_per_kv, -1, -1).reshape(num_heads, ctx, head_dim)
            v_seq = v_seq.unsqueeze(1).expand(-1, num_q_per_kv, -1, -1).reshape(num_heads, ctx, head_dim)

        qi = query[i].float()
        scores = torch.einsum('hd,hsd->hs', qi, k_seq.float()) * scale
        o[i] = torch.einsum('hs,hsd->hd', F.softmax(scores, dim=-1), v_seq.float())

    return o.to(query.dtype)


# ============================================================================
# Rotary
# ============================================================================
def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos + _rotate_half(q) * sin,
            k * cos + _rotate_half(k) * sin)


def _apply_qwen_rope(model, q, k, position_ids):
    """
    q: (B, Hq, T, D)
    k: (B, Hkv, T, D)

    这部分不同 transformers/Qwen 版本实现细节可能略有差异。
    当前写法兼容较常见的 Qwen2.5 HF 路径。
    """
    try:
        cos, sin = model.model.rotary_emb(k, position_ids)
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)
        return q, k
    except Exception:
        try:
            cos, sin = model.model.rotary_emb(q, position_ids)
            q, k = _apply_rotary_pos_emb(q, k, cos, sin)
            return q, k
        except Exception as e:
            raise RuntimeError(
                "RoPE 接口与当前本地 transformers/Qwen 实现不匹配，请按本地模型代码微调 _apply_qwen_rope"
            ) from e


# ============================================================================
# Prefill: Prefix paged attention (PyTorch reference)
# ============================================================================
def paged_prefix_attention_pt(
    q,                # (B, T, Hq, D)
    k_cache,          # (num_blocks, block_size, Hkv, D)
    v_cache,          # (num_blocks, block_size, Hkv, D)
    block_tables,     # (B, max_blocks)
    prefix_lens,      # (B,)
    num_q_heads,
    num_kv_heads,
    head_dim,
    block_size,
    scale,
):
    B, T, _, D = q.shape
    num_q_per_kv = num_q_heads // num_kv_heads

    out = torch.zeros((B, T, num_q_heads, D), device=q.device, dtype=torch.float32)
    lse = torch.full((B, T, num_q_heads), float("-inf"), device=q.device, dtype=torch.float32)

    for b in range(B):
        ctx = int(prefix_lens[b].item())
        if ctx == 0:
            out[b].zero_()
            lse[b].fill_(float("-inf"))
            continue

        n_blks = math.ceil(ctx / block_size)
        k_list, v_list = [], []
        for bi in range(n_blks):
            pb = int(block_tables[b, bi].item())
            if pb == -1:
                continue
            start = bi * block_size
            length = min(block_size, ctx - start)
            k_list.append(k_cache[pb, :length, :, :])
            v_list.append(v_cache[pb, :length, :, :])

        k_seq = torch.cat(k_list, dim=0)
        v_seq = torch.cat(v_list, dim=0)

        k_seq = k_seq.permute(1, 0, 2).contiguous()
        v_seq = v_seq.permute(1, 0, 2).contiguous()

        if num_q_per_kv > 1:
            k_seq = k_seq.unsqueeze(1).expand(-1, num_q_per_kv, -1, -1).reshape(num_q_heads, ctx, D)
            v_seq = v_seq.unsqueeze(1).expand(-1, num_q_per_kv, -1, -1).reshape(num_q_heads, ctx, D)
        else:
            k_seq = k_seq.reshape(num_q_heads, ctx, D)
            v_seq = v_seq.reshape(num_q_heads, ctx, D)

        qb = q[b].float()
        scores = torch.einsum("thd,hsd->ths", qb, k_seq.float()) * scale
        max_score = scores.max(dim=-1).values
        exp_scores = torch.exp(scores - max_score.unsqueeze(-1))
        denom = exp_scores.sum(dim=-1)
        probs = exp_scores / denom.unsqueeze(-1)
        ob = torch.einsum("ths,hsd->thd", probs, v_seq.float())

        out[b] = ob
        lse[b] = max_score + torch.log(denom)

    return out.to(q.dtype), lse


# ============================================================================
# Prefill: local causal attention
# ============================================================================
def local_causal_attention_sdpa(q, k, v, num_q_heads, num_kv_heads):
    """
    q: (B, T, Hq, D)
    k: (B, T, Hkv, D)
    v: (B, T, Hkv, D)
    """
    B, T, Hq, D = q.shape
    Hkv = num_kv_heads
    num_q_per_kv = Hq // Hkv

    q2 = q.permute(0, 2, 1, 3).contiguous()
    k2 = k.permute(0, 2, 1, 3).contiguous()
    v2 = v.permute(0, 2, 1, 3).contiguous()

    if num_q_per_kv > 1:
        k2 = k2.unsqueeze(2).expand(B, Hkv, num_q_per_kv, T, D).reshape(B, Hq, T, D)
        v2 = v2.unsqueeze(2).expand(B, Hkv, num_q_per_kv, T, D).reshape(B, Hq, T, D)

    out = F.scaled_dot_product_attention(
        q2.float(), k2.float(), v2.float(),
        attn_mask=None, dropout_p=0.0, is_causal=True
    )

    scale = 1.0 / math.sqrt(D)
    scores = torch.einsum("bhtd,bhsd->bhts", q2.float(), k2.float()) * scale
    causal = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))
    max_score = scores.max(dim=-1).values
    exp_scores = torch.exp(scores - max_score.unsqueeze(-1))
    denom = exp_scores.sum(dim=-1)
    lse = max_score + torch.log(denom)

    out = out.permute(0, 2, 1, 3).contiguous()
    lse = lse.permute(0, 2, 1).contiguous()
    return out.to(q.dtype), lse


# ============================================================================
# Merge prefix/local attention outputs
# ============================================================================
def merge_attn_outputs(prefix_out, prefix_lse, local_out, local_lse):
    m = torch.maximum(prefix_lse, local_lse)
    wp = torch.exp(prefix_lse - m)
    wl = torch.exp(local_lse - m)
    denom = wp + wl
    return (prefix_out * wp.unsqueeze(-1) + local_out * wl.unsqueeze(-1)) / denom.unsqueeze(-1)


# ============================================================================
# 单层/整块 prefill-extend
# ============================================================================
@torch.inference_mode()
def paged_extend_step(
    model,
    input_ids,         # (B, T)
    position_ids,      # (B, T)
    paged_cache: PagedKVCache,
    seq_ids: List[int],
    valid_lens: List[int],  # 每条样本在这个 chunk 中的真实长度
):
    device = input_ids.device
    cfg = model.config
    num_q_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_q_heads)
    head_dim = cfg.hidden_size // num_q_heads
    block_size = paged_cache.block_size
    scale = 1.0 / math.sqrt(head_dim)

    B, T = input_ids.shape
    hidden = model.model.embed_tokens(input_ids)

    prefix_lens = paged_cache.get_context_lens_tensor(seq_ids)
    block_tables = paged_cache.get_block_table_tensor(seq_ids)

    # 为本 chunk 的真实 token 预分配 slot
    flat_slots = []
    real_token_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
    for i, sid in enumerate(seq_ids):
        L = valid_lens[i]
        real_token_mask[i, :L] = True
        flat_slots.extend(paged_cache.get_slot_mapping_for_append(sid, L))

    flat_slots = torch.tensor(flat_slots, dtype=torch.long, device=device)

    for layer_idx, layer in enumerate(model.model.layers):
        residual = hidden
        h = layer.input_layernorm(hidden)

        q = layer.self_attn.q_proj(h)
        k = layer.self_attn.k_proj(h)
        v = layer.self_attn.v_proj(h)

        q = q.view(B, T, num_q_heads, head_dim)
        k = k.view(B, T, num_kv_heads, head_dim)
        v = v.view(B, T, num_kv_heads, head_dim)

        q_rope = q.transpose(1, 2).contiguous()
        k_rope = k.transpose(1, 2).contiguous()
        q_rope, k_rope = _apply_qwen_rope(model, q_rope, k_rope, position_ids)
        q = q_rope.transpose(1, 2).contiguous()
        k = k_rope.transpose(1, 2).contiguous()

        prefix_out, prefix_lse = paged_prefix_attention_pt(
            q,
            paged_cache.k_pools[layer_idx],
            paged_cache.v_pools[layer_idx],
            block_tables,
            prefix_lens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            block_size,
            scale,
        )

        local_out, local_lse = local_causal_attention_sdpa(
            q, k, v, num_q_heads, num_kv_heads
        )

        attn_out = merge_attn_outputs(
            prefix_out.float(), prefix_lse.float(),
            local_out.float(), local_lse.float()
        ).to(hidden.dtype)

        # 对 chunk 内 pad 位置清零
        attn_out = attn_out * real_token_mask.unsqueeze(-1).unsqueeze(-1)

        attn_out = attn_out.reshape(B, T, num_q_heads * head_dim)
        attn_out = layer.self_attn.o_proj(attn_out)
        hidden = residual + attn_out

        residual = hidden
        h = layer.post_attention_layernorm(hidden)
        mlp_out = layer.mlp(h)
        mlp_out = mlp_out * real_token_mask.unsqueeze(-1)
        hidden = residual + mlp_out

        # 写入本层真实 token 的 KV 到 paged cache
        k_real = []
        v_real = []
        for i in range(B):
            L = valid_lens[i]
            if L > 0:
                k_real.append(k[i, :L])
                v_real.append(v[i, :L])

        if len(k_real) > 0:
            k_store = torch.cat(k_real, dim=0).reshape(-1, num_kv_heads, head_dim).contiguous()
            v_store = torch.cat(v_real, dim=0).reshape(-1, num_kv_heads, head_dim).contiguous()
            store_kvcache(
                k_store,
                v_store,
                paged_cache.k_pools[layer_idx],
                paged_cache.v_pools[layer_idx],
                flat_slots,
                block_size,
            )

    for i, sid in enumerate(seq_ids):
        paged_cache.append_len(sid, valid_lens[i])

    hidden = model.model.norm(hidden)
    logits = model.lm_head(hidden)
    return logits


# ============================================================================
# Decode step
# ============================================================================
@torch.inference_mode()
def paged_decode_step(
    model,
    token_ids,         # (B, 1)
    position_ids,      # (B, 1)
    paged_cache: PagedKVCache,
    seq_ids: List[int],
):
    device = token_ids.device
    cfg = model.config
    num_q_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_q_heads)
    head_dim = cfg.hidden_size // num_q_heads
    block_size = paged_cache.block_size
    scale = 1.0 / math.sqrt(head_dim)
    B = token_ids.shape[0]

    hidden = model.model.embed_tokens(token_ids)

    slot_list = []
    for sid in seq_ids:
        slot_list.extend(paged_cache.get_slot_mapping_for_append(sid, 1))
    slot_mapping = torch.tensor(slot_list, dtype=torch.long, device=device)

    for layer_idx, layer in enumerate(model.model.layers):
        residual = hidden
        h = layer.input_layernorm(hidden)

        q = layer.self_attn.q_proj(h)
        k = layer.self_attn.k_proj(h)
        v = layer.self_attn.v_proj(h)

        q = q.view(B, 1, num_q_heads, head_dim)
        k = k.view(B, 1, num_kv_heads, head_dim)
        v = v.view(B, 1, num_kv_heads, head_dim)

        q_rope = q.transpose(1, 2).contiguous()
        k_rope = k.transpose(1, 2).contiguous()
        q_rope, k_rope = _apply_qwen_rope(model, q_rope, k_rope, position_ids)

        q = q_rope.transpose(1, 2).contiguous()
        k = k_rope.transpose(1, 2).contiguous()

        k_store = k.reshape(B, num_kv_heads, head_dim).contiguous()
        v_store = v.reshape(B, num_kv_heads, head_dim).contiguous()
        store_kvcache(
            k_store, v_store,
            paged_cache.k_pools[layer_idx],
            paged_cache.v_pools[layer_idx],
            slot_mapping,
            block_size,
        )

        ctx_lens = paged_cache.get_context_lens_tensor(seq_ids) + 1
        blk_tables = paged_cache.get_block_table_tensor(seq_ids)

        q_attn = q.reshape(B, num_q_heads, head_dim)
        attn_out = paged_attention_decode(
            q_attn,
            paged_cache.k_pools[layer_idx],
            paged_cache.v_pools[layer_idx],
            blk_tables,
            ctx_lens,
            scale,
            num_q_heads,
            num_kv_heads,
            head_dim,
            block_size,
        )

        attn_out = attn_out.reshape(B, 1, num_q_heads * head_dim).to(hidden.dtype)
        attn_out = layer.self_attn.o_proj(attn_out)
        hidden = residual + attn_out

        residual = hidden
        h = layer.post_attention_layernorm(hidden)
        hidden = residual + layer.mlp(h)

    for sid in seq_ids:
        paged_cache.append_len(sid, 1)

    hidden = model.model.norm(hidden)
    logits = model.lm_head(hidden[:, -1, :])
    return logits


# ============================================================================
# 统一生成：Prefill + Decode 都走 paged cache
# ============================================================================
@torch.inference_mode()
def batch_generate_paged(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = MAX_NEW_TOKENS,
    prefill_chunk_size: int = PREFILL_CHUNK_SIZE,
):
    device = torch.device(DEVICE)
    pad_id = tokenizer.pad_token_id or 0
    eos_ids = _eos_ids(tokenizer)
    eos_t = torch.tensor(sorted(eos_ids), dtype=torch.long, device=device)
    B = len(prompts)

    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_q_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_q_heads)
    head_dim = cfg.hidden_size // num_q_heads

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    input_lengths = attention_mask.sum(dim=1).tolist()
    padded_len = input_ids.shape[1]

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
        paged_cache.allocate_seq(i, init_tokens=input_lengths[i])

    # 去掉 left padding，拿到每条真实 token
    real_token_lists = []
    for i in range(B):
        L = input_lengths[i]
        real_ids = input_ids[i, padded_len - L:padded_len]
        real_token_lists.append(real_ids)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    # -------------------------
    # Prefill: chunked extend
    # -------------------------
    last_logits_per_seq = [None] * B
    max_input_len = max(input_lengths)

    for start in range(0, max_input_len, prefill_chunk_size):
        end = min(start + prefill_chunk_size, max_input_len)

        active_seq_ids = []
        chunk_list = []
        pos_list = []
        valid_lens = []

        for sid in range(B):
            seq = real_token_lists[sid]
            if start < seq.shape[0]:
                sub = seq[start:end]
                active_seq_ids.append(sid)
                chunk_list.append(sub)
                valid_lens.append(sub.shape[0])
                pos_list.append(torch.arange(start, start + sub.shape[0], device=device, dtype=torch.long))

        if len(active_seq_ids) == 0:
            continue

        T = max(valid_lens)
        chunk_ids = torch.full((len(active_seq_ids), T), pad_id, dtype=torch.long, device=device)
        chunk_pos = torch.zeros((len(active_seq_ids), T), dtype=torch.long, device=device)

        for i in range(len(active_seq_ids)):
            L = valid_lens[i]
            chunk_ids[i, :L] = chunk_list[i]
            chunk_pos[i, :L] = pos_list[i]

        logits = paged_extend_step(
            model,
            chunk_ids,
            chunk_pos,
            paged_cache,
            active_seq_ids,
            valid_lens,
        )

        for i, sid in enumerate(active_seq_ids):
            last_logits_per_seq[sid] = logits[i, valid_lens[i] - 1, :].float()

    torch.cuda.synchronize(device)
    ttft = (time.perf_counter() - t0) * 1000.0

    first_logits = torch.stack(last_logits_per_seq, dim=0)
    first_tokens = first_logits.argmax(dim=-1)

    # -------------------------
    # Decode
    # -------------------------
    unfinished = torch.ones(B, dtype=torch.bool, device=device)
    generated = [first_tokens.unsqueeze(1)]
    sample_lengths = torch.zeros(B, dtype=torch.long, device=device)
    cur_tokens = first_tokens

    is_eos = (cur_tokens.unsqueeze(1) == eos_t.unsqueeze(0)).any(dim=1)
    sample_lengths += (unfinished & ~is_eos).long()
    unfinished = unfinished & ~is_eos

    positions = torch.tensor(input_lengths, dtype=torch.long, device=device)

    for _ in range(1, max_new_tokens):
        if not unfinished.any():
            break

        pos_ids = positions.unsqueeze(1)

        logits = paged_decode_step(
            model,
            cur_tokens.unsqueeze(1),
            pos_ids,
            paged_cache,
            seq_ids,
        )

        next_tokens = logits.argmax(dim=-1)
        next_tokens = torch.where(
            unfinished,
            next_tokens,
            torch.full_like(next_tokens, pad_id),
        )

        is_eos = (next_tokens.unsqueeze(1) == eos_t.unsqueeze(0)).any(dim=1)
        sample_lengths += (unfinished & ~is_eos).long()
        generated.append(next_tokens.unsqueeze(1))

        unfinished = unfinished & ~is_eos
        positions += 1
        cur_tokens = next_tokens

    torch.cuda.synchronize(device)
    total_ms = (time.perf_counter() - t0) * 1000.0

    gen_ids = torch.cat(generated, dim=1)
    lengths = sample_lengths.tolist()
    texts = []
    for i in range(B):
        L = min(max(lengths[i], 0), gen_ids.shape[1])
        texts.append(
            tokenizer.decode(gen_ids[i, :L].tolist(), skip_special_tokens=True)
            if L > 0 else ""
        )

    for i in range(B):
        paged_cache.free_seq(i)

    return texts, lengths, input_lengths, ttft, total_ms


# ============================================================================
# 标准 HF 版本（对照/回退）
# ============================================================================
@torch.inference_mode()
def batch_generate_standard(
    model,
    tokenizer,
    prompts,
    max_new_tokens: int = MAX_NEW_TOKENS,
):
    device = torch.device(DEVICE)
    pad_id = tokenizer.pad_token_id or 0
    eos_ids = _eos_ids(tokenizer)
    eos_t = torch.tensor(sorted(eos_ids), dtype=torch.long, device=device)
    B = len(prompts)

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    input_lengths = attention_mask.sum(dim=1).tolist()

    unfinished = torch.ones(B, dtype=torch.bool, device=device)
    generated = []
    sample_lengths = torch.zeros(B, dtype=torch.long, device=device)
    past = None
    cur_ids = input_ids
    cur_mask = attention_mask

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
        logits = out.logits[:, -1, :]
        past = out.past_key_values

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
        cur_mask = torch.cat(
            [cur_mask, torch.ones(B, 1, device=device, dtype=cur_mask.dtype)],
            dim=1,
        )

    torch.cuda.synchronize(device)
    total_ms = (time.perf_counter() - t0) * 1000.0
    if ttft is None:
        ttft = total_ms

    gen_ids = torch.cat(generated, dim=1) if generated else torch.zeros(B, 0, dtype=torch.long, device=device)
    lengths = sample_lengths.tolist()
    texts = []
    for i in range(B):
        L = min(max(lengths[i], 0), gen_ids.shape[1])
        texts.append(
            tokenizer.decode(gen_ids[i, :L].tolist(), skip_special_tokens=True)
            if L > 0 else ""
        )

    return texts, lengths, input_lengths, ttft, total_ms


# ============================================================================
# 自动 batch size
# ============================================================================
def _auto_batch_size(model, tokenizer, initial_bs=64):
    device = torch.device(DEVICE)
    dummy = "你好世界，请简要回答。"
    best = 1
    lo, hi = 1, initial_bs
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            torch.cuda.empty_cache()
            enc = tokenizer([dummy] * mid, return_tensors="pt", padding=True,
                            truncation=True, max_length=128).to(device)
            with torch.inference_mode():
                _ = model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    use_cache=True,
                    return_dict=True,
                )
            torch.cuda.synchronize(device)
            best = mid
            lo = mid + 1
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            hi = mid - 1
        except Exception:
            hi = mid - 1
    torch.cuda.empty_cache()
    return max(1, int(best * 0.85))


# ============================================================================
# 加载模型
# ============================================================================
def load_model(model_path: str):
    print(f"[INFO] 加载模型: {model_path}")
    print(f"[INFO] 设备={DEVICE}  精度={DTYPE}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
            print(f"[OPT] Attention backend: {attn}")
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
        print("[OPT] Attention backend: default")

    model.eval()

    _probe_triton()
    apply_optimizations(model)

    print("[INFO] 预热推理...")
    warm = tokenizer("hello world", return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        for _ in range(3):
            model(**warm, use_cache=True)
    torch.cuda.synchronize(DEVICE)

    n_p = sum(p.numel() for p in model.parameters()) / 1e9
    vram = torch.cuda.memory_allocated(DEVICE) / 1e9
    print(f"[INFO] 就绪 | {n_p:.1f}B params | VRAM {vram:.2f} GB")
    return tokenizer, model


# ============================================================================
# 高层接口
# ============================================================================
def infer_all(
    tokenizer,
    model,
    prompts: list,
    batch_size: int = BATCH_SIZE,
    max_new_tokens: int = MAX_NEW_TOKENS,
    show_progress: bool = True,
    use_paged: bool = True,
    prefill_chunk_size: int = PREFILL_CHUNK_SIZE,
):
    n = len(prompts)
    if n == 0:
        return []

    enc_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    sorted_idx = sorted(range(n), key=lambda i: enc_lens[i])

    all_results = [None] * n
    num_batches = math.ceil(n / batch_size)

    for b in range(num_batches):
        s = b * batch_size
        e = min(s + batch_size, n)
        idx_b = sorted_idx[s:e]
        p_b = [prompts[i] for i in idx_b]

        if use_paged:
            texts, out_lens, in_lens, ttft, total = batch_generate_paged(
                model, tokenizer, p_b,
                max_new_tokens=max_new_tokens,
                prefill_chunk_size=prefill_chunk_size,
            )
        else:
            texts, out_lens, in_lens, ttft, total = batch_generate_standard(
                model, tokenizer, p_b,
                max_new_tokens=max_new_tokens,
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
            mode = "paged" if use_paged else "standard"
            print(
                f"  [batch {b+1}/{num_batches}] mode={mode} "
                f"bs={len(p_b)} ttft={ttft:.0f}ms total={total:.0f}ms "
                f"out_tok={sum(out_lens)} ({e}/{n} done)"
            )

    return all_results


def infer_single(tokenizer, model, prompt: str, use_paged: bool = True) -> dict:
    return infer_all(
        tokenizer,
        model,
        [prompt],
        batch_size=1,
        show_progress=False,
        use_paged=use_paged,
    )[0]


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    import argparse

    pa = argparse.ArgumentParser()
    pa.add_argument("--model_path", type=str, required=True)
    pa.add_argument("--prompt", type=str, default="请用三句话解释 KV Cache 的作用。")
    pa.add_argument("--batch_size", type=int, default=-1)
    pa.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    pa.add_argument("--prefill_chunk_size", type=int, default=PREFILL_CHUNK_SIZE)
    pa.add_argument("--no_paged", action="store_true")
    args = pa.parse_args()

    tok, mdl = load_model(args.model_path)

    if args.batch_size <= 0:
        args.batch_size = _auto_batch_size(mdl, tok)
        print(f"[INFO] auto batch_size = {args.batch_size}")

    r = infer_single(tok, mdl, args.prompt, use_paged=not args.no_paged)

    print(f"\n{'=' * 60}")
    print(f"输出: {r['output'][:300]}")
    print(f"in={r['input_tokens']} out={r['output_tokens']}")
    print(f"延迟={r['total_latency_ms']:.1f}ms  TTFT={r['ttft_ms']:.1f}ms")
    print(f"吞吐={r['throughput_tps']:.1f} tok/s")
    print(f"峰值显存={torch.cuda.max_memory_allocated(DEVICE)/1e9:.2f} GB")
    print(f"{'=' * 60}")