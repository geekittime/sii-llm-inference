#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimized_inference.py
======================
Qwen2.5-14B-Instruct 工业级 PagedAttention 推理引擎

核心改进 (v2 — 工业级 paged attention):

  KV Cache 布局:
    K cache: (max_blocks, num_kv_heads, head_dim, block_size)  — K 转置存储
    V cache: (max_blocks, num_kv_heads, block_size, head_dim)  — V 正常存储
    → attention 计算时 K^T 和 V 均为连续内存读取

  Prefill — chunked extend:
    ① prefix paged attention:  Q_tile(BLOCK_M, D) × K^T_block(D, BLOCK_N) via tl.dot
       → 利用 Tensor Core, 一个 program 处理 BLOCK_M 个 query
    ② local  causal attention: Q_chunk × K_chunk (causal) → out_l, lse_l
    ③ online-softmax merge:    (out_p, lse_p) ⊕ (out_l, lse_l) → final
    ④ store K^T, V → paged cache
    ⑤ advance seq_lens

  Decode — Flash-Decoding (Split-K):
    ① store new K^T, V → paged cache
    ② Split-K: 将 KV 序列分为 NUM_SPLITS 段, 各段并行计算 partial attn
    ③ reduction kernel: 合并各段 partial output + partial LSE
    ④ advance seq_lens

  Triton 融合算子 (均有 PyTorch fallback):
    RMSNorm / SwiGLU / store_kvcache / paged_decode (split-k) /
    prefix_paged_prefill (tiled tl.dot)
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

# Flash-Decoding split-K 配置
DECODE_NUM_SPLITS = 8          # decode 阶段 KV 序列分段数
PREFILL_BLOCK_M = 64           # prefill prefix kernel query tile 大小

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

    # ---- RMSNorm (保持不变) ----
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
        _rms_norm_k[(M,)](x2, w, y, x2.stride(0), y.stride(0), N, eps,
                          BLOCK=BLK)
        return y.reshape(s)

    # ---- SiLU×Mul (保持不变) ----
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
        _silu_mul_k[(M,)](g2, u2, o, g2.stride(0), u2.stride(0),
                          o.stride(0), N, BLOCK=BLK)
        return o.reshape(s)

    # ================================================================
    # Store KV Cache — K 转置存储
    #   K cache: (max_blocks, num_kv_heads, head_dim, block_size)
    #   V cache: (max_blocks, num_kv_heads, block_size, head_dim)
    #   input key/value: (num_tokens, num_kv_heads, head_dim)
    # ================================================================
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

        offs_d = tl.arange(0, head_dim)

        # ---- 输入偏移: (num_tokens, num_kv_heads, head_dim) ----
        in_off = (token_idx * num_kv_heads * head_dim
                  + head_idx * head_dim + offs_d)

        key_vec = tl.load(key_ptr + in_off)
        val_vec = tl.load(value_ptr + in_off)

        # ---- K cache 转置布局: (blocks, heads, head_dim, block_size) ----
        # k_cache[block_idx, head_idx, d, block_offset] = key[d]
        # stride: block_idx * (nkv * D * BS) + head * (D * BS) + d * BS + offset
        k_off = (block_idx * num_kv_heads * head_dim * block_size
                 + head_idx * head_dim * block_size
                 + offs_d * block_size + block_offset)
        tl.store(k_cache_ptr + k_off, key_vec)

        # ---- V cache 正常布局: (blocks, heads, block_size, head_dim) ----
        # v_cache[block_idx, head_idx, block_offset, d] = val[d]
        v_off = (block_idx * num_kv_heads * block_size * head_dim
                 + head_idx * block_size * head_dim
                 + block_offset * head_dim + offs_d)
        tl.store(v_cache_ptr + v_off, val_vec)

    # ================================================================
    # Paged Attention Decode — 向量化 + Online Softmax
    # Grid: (batch, num_heads)
    # 每个 program 处理一个 (batch, head) 对的完整 KV 序列
    # ================================================================
    @triton.jit
    def _paged_attn_decode_kernel_v2(
        output_ptr,           # (B, num_heads, head_dim)
        query_ptr,            # (B, num_heads, head_dim)
        k_cache_ptr,          # (max_blocks, num_kv_heads, head_dim, block_size)
        v_cache_ptr,          # (max_blocks, num_kv_heads, block_size, head_dim)
        block_tables_ptr,     # (B, max_num_blocks) int32
        context_lens_ptr,     # (B,) int64
        scale,
        max_num_blocks,
        stride_qt_b, stride_qt_h,   # query strides
        stride_kt_blk, stride_kt_h,  # k_cache strides
        stride_vt_blk, stride_vt_h,  # v_cache strides
        stride_bt_b,                  # block_tables stride
        NUM_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        kv_head_idx = head_idx // (NUM_HEADS // NUM_KV_HEADS)

        ctx_len = tl.load(context_lens_ptr + batch_idx)

        # ---- 加载 Q: (HEAD_DIM,) ----
        offs_d = tl.arange(0, HEAD_DIM)
        q = tl.load(query_ptr + batch_idx * stride_qt_b
                     + head_idx * stride_qt_h + offs_d).to(tl.float32)

        # ---- Online softmax 状态 ----
        m_i = float("-inf")
        l_i = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        # ---- 迭代所有 page block ----
        num_blocks_seq = tl.cdiv(ctx_len, BLOCK_SIZE)
        for block_num in range(max_num_blocks):
            cur_start = block_num * BLOCK_SIZE
            still_valid = cur_start < ctx_len
            if still_valid:
                # 查物理 block
                phys_block = tl.load(block_tables_ptr
                                     + batch_idx * stride_bt_b
                                     + block_num)

                # ---- 加载 K^T 块: (HEAD_DIM, BLOCK_SIZE) ----
                # k_cache[phys, kv_head, :, :] 连续
                offs_bs = tl.arange(0, BLOCK_SIZE)
                k_base = (phys_block * stride_kt_blk
                          + kv_head_idx * stride_kt_h)
                # shape: (HEAD_DIM, BLOCK_SIZE)
                kt_block = tl.load(k_cache_ptr + k_base
                                   + offs_d[:, None] * BLOCK_SIZE
                                   + offs_bs[None, :]).to(tl.float32)

                # ---- 计算 scores: q @ K^T → (BLOCK_SIZE,) ----
                # q: (HEAD_DIM,) → (HEAD_DIM, 1) broadcast multiply
                scores = tl.sum(q[:, None] * kt_block, axis=0) * scale

                # mask 尾部无效位置
                valid_mask = (cur_start + offs_bs) < ctx_len
                scores = tl.where(valid_mask, scores, float("-inf"))

                # ---- Online softmax 更新 ----
                m_ij = tl.max(scores)
                m_new = tl.maximum(m_i, m_ij)
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(scores - m_new)

                l_i = l_i * alpha + tl.sum(p)
                acc = acc * alpha

                # ---- 加载 V 块: (BLOCK_SIZE, HEAD_DIM) ----
                v_base = (phys_block * stride_vt_blk
                          + kv_head_idx * stride_vt_h)
                v_block = tl.load(v_cache_ptr + v_base
                                  + offs_bs[:, None] * HEAD_DIM
                                  + offs_d[None, :]).to(tl.float32)

                # 加权累加: p[:, None] * V → sum over BLOCK_SIZE → (HEAD_DIM,)
                # 等效于 p^T @ V
                acc += tl.sum(p[:, None] * v_block, axis=0)
                m_i = m_new

        # ---- 归一化输出 ----
        safe_l = tl.where(l_i > 0.0, l_i, 1.0)
        out = acc / safe_l
        tl.store(output_ptr + batch_idx * stride_qt_b
                 + head_idx * stride_qt_h + offs_d, out)

    # ================================================================
    # Flash-Decoding: Split-K Decode Kernel
    # Grid: (batch, num_heads, num_splits)
    # 每个 program 只处理 KV 序列的一个分段
    # ================================================================
    @triton.jit
    def _paged_attn_decode_splitk_kernel(
        partial_out_ptr,      # (B, num_heads, num_splits, head_dim) fp32
        partial_lse_ptr,      # (B, num_heads, num_splits) fp32
        query_ptr,
        k_cache_ptr, v_cache_ptr,
        block_tables_ptr, context_lens_ptr,
        scale,
        max_num_blocks,
        stride_qt_b, stride_qt_h,
        stride_kt_blk, stride_kt_h,
        stride_vt_blk, stride_vt_h,
        stride_bt_b,
        stride_po_b, stride_po_h, stride_po_s,   # partial_out strides
        stride_pl_b, stride_pl_h,                  # partial_lse strides: (B, H, S)
        NUM_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        split_idx = tl.program_id(2)
        kv_head_idx = head_idx // (NUM_HEADS // NUM_KV_HEADS)

        ctx_len = tl.load(context_lens_ptr + batch_idx)
        num_blocks_seq = tl.cdiv(ctx_len, BLOCK_SIZE)

        # 该 split 负责的 block 范围
        blocks_per_split = tl.cdiv(num_blocks_seq, NUM_SPLITS)
        blk_start = split_idx * blocks_per_split
        blk_end = tl.minimum(blk_start + blocks_per_split, num_blocks_seq)

        offs_d = tl.arange(0, HEAD_DIM)
        offs_bs = tl.arange(0, BLOCK_SIZE)

        q = tl.load(query_ptr + batch_idx * stride_qt_b
                     + head_idx * stride_qt_h + offs_d).to(tl.float32)

        m_i = float("-inf")
        l_i = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        for block_num in range(blocks_per_split):
            cur_blk = blk_start + block_num
            cur_start = cur_blk * BLOCK_SIZE
            if cur_blk < blk_end and cur_start < ctx_len:
                phys_block = tl.load(block_tables_ptr
                                     + batch_idx * stride_bt_b + cur_blk)

                k_base = phys_block * stride_kt_blk + kv_head_idx * stride_kt_h
                kt_block = tl.load(k_cache_ptr + k_base
                                   + offs_d[:, None] * BLOCK_SIZE
                                   + offs_bs[None, :]).to(tl.float32)

                scores = tl.sum(q[:, None] * kt_block, axis=0) * scale
                valid_mask = (cur_start + offs_bs) < ctx_len
                scores = tl.where(valid_mask, scores, float("-inf"))

                m_ij = tl.max(scores)
                m_new = tl.maximum(m_i, m_ij)
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(scores - m_new)
                l_i = l_i * alpha + tl.sum(p)
                acc = acc * alpha

                v_base = phys_block * stride_vt_blk + kv_head_idx * stride_vt_h
                v_block = tl.load(v_cache_ptr + v_base
                                  + offs_bs[:, None] * HEAD_DIM
                                  + offs_d[None, :]).to(tl.float32)

                acc += tl.sum(p[:, None] * v_block, axis=0)
                m_i = m_new

        # 存 partial 结果
        lse_val = tl.where(l_i > 0.0, m_i + tl.log(l_i), float("-inf"))
        safe_l = tl.where(l_i > 0.0, l_i, 1.0)
        partial_o = acc / safe_l

        po_off = (batch_idx * stride_po_b + head_idx * stride_po_h
                  + split_idx * stride_po_s + offs_d)
        tl.store(partial_out_ptr + po_off, partial_o)

        pl_off = (batch_idx * stride_pl_b + head_idx * stride_pl_h
                  + split_idx)
        tl.store(partial_lse_ptr + pl_off, lse_val)

    # ================================================================
    # Split-K Reduction Kernel
    # Grid: (batch, num_heads)
    # 合并 num_splits 个 partial 结果
    # ================================================================
    @triton.jit
    def _splitk_reduce_kernel(
        output_ptr,           # (B, num_heads, head_dim)
        partial_out_ptr,      # (B, num_heads, num_splits, head_dim)
        partial_lse_ptr,      # (B, num_heads, num_splits)
        stride_o_b, stride_o_h,
        stride_po_b, stride_po_h, stride_po_s,
        stride_pl_b, stride_pl_h,
        HEAD_DIM: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)

        offs_d = tl.arange(0, HEAD_DIM)

        # Online merge across splits
        m_i = float("-inf")
        l_i = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        for s in range(NUM_SPLITS):
            lse_s = tl.load(partial_lse_ptr
                            + batch_idx * stride_pl_b
                            + head_idx * stride_pl_h + s)

            # 跳过空 split
            is_valid = lse_s > float("-inf")
            if is_valid:
                po = tl.load(partial_out_ptr
                             + batch_idx * stride_po_b
                             + head_idx * stride_po_h
                             + s * stride_po_s + offs_d)

                m_new = tl.maximum(m_i, lse_s)
                # 旧累积缩放
                alpha = tl.exp(m_i - m_new)
                # 新 split 权重
                w_new = tl.exp(lse_s - m_new)

                l_i = l_i * alpha + w_new
                acc = acc * alpha + w_new * po
                m_i = m_new

        safe_l = tl.where(l_i > 0.0, l_i, 1.0)
        out = acc / safe_l

        tl.store(output_ptr + batch_idx * stride_o_b
                 + head_idx * stride_o_h + offs_d, out)

    # ================================================================
    # Paged Prefix Attention for Prefill — Tiled tl.dot
    # Grid: (cdiv(max_q_len, BLOCK_M), B, num_heads)
    #
    # 一个 program 处理 BLOCK_M 个连续 query 位置对全部 prefix KV 的注意力
    # 使用 tl.dot (Tensor Core) 执行:
    #   scores = Q_tile @ K^T_block  →  (BLOCK_M, BLOCK_N)
    #   acc   += P_tile @ V_block    →  (BLOCK_M, HEAD_DIM)
    # ================================================================
    @triton.jit
    def _paged_prefix_attn_tiled_kernel(
        output_ptr,           # (B, max_q_len, num_heads, head_dim) fp32
        lse_ptr,              # (B, max_q_len, num_heads) fp32
        query_ptr,            # (B, max_q_len, num_heads, head_dim) fp16
        k_cache_ptr,          # (max_blocks, num_kv_heads, head_dim, block_size)
        v_cache_ptr,          # (max_blocks, num_kv_heads, block_size, head_dim)
        block_tables_ptr,     # (B, max_num_blocks) int32
        prefix_lens_ptr,      # (B,) int64 — 已缓存长度
        valid_lens_ptr,       # (B,) int64 — chunk 内有效 query 数
        scale,
        max_num_blocks,
        stride_q_b, stride_q_t, stride_q_h,    # query strides
        stride_o_b, stride_o_t, stride_o_h,    # output strides
        stride_l_b, stride_l_t,                 # lse strides: (B, T, H)
        stride_kt_blk, stride_kt_h,
        stride_vt_blk, stride_vt_h,
        stride_bt_b,
        NUM_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,     # = page block size
        BLOCK_M: tl.constexpr,        # query tile size (e.g., 64)
        MAX_Q_LEN: tl.constexpr,
    ):
        q_tile_idx = tl.program_id(0)
        batch_idx = tl.program_id(1)
        head_idx = tl.program_id(2)
        kv_head_idx = head_idx // (NUM_HEADS // NUM_KV_HEADS)

        prefix_len = tl.load(prefix_lens_ptr + batch_idx)
        valid_len = tl.load(valid_lens_ptr + batch_idx)

        q_start = q_tile_idx * BLOCK_M
        offs_m = q_start + tl.arange(0, BLOCK_M)          # (BLOCK_M,)
        offs_d = tl.arange(0, HEAD_DIM)                    # (HEAD_DIM,)
        offs_bs = tl.arange(0, BLOCK_SIZE)                 # (BLOCK_SIZE,)

        # 输出偏移
        out_base = batch_idx * stride_o_b + head_idx * stride_o_h
        lse_base = batch_idx * stride_l_b + head_idx

        # 全 tile 在 query 范围外或无 prefix → 写零 + -inf
        any_valid = (q_start < valid_len) & (prefix_len > 0)

        if any_valid == 0:
            for i in range(BLOCK_M):
                qi = q_start + i
                if qi < MAX_Q_LEN:
                    tl.store(output_ptr + out_base + qi * stride_o_t + offs_d,
                             tl.zeros([HEAD_DIM], dtype=tl.float32))
                    tl.store(lse_ptr + lse_base + qi * stride_l_t,
                             float("-inf"))
            return

        q_valid_mask = offs_m < valid_len                  # (BLOCK_M,)

        # ---- 加载 Q tile: (BLOCK_M, HEAD_DIM) ----
        q_base = (batch_idx * stride_q_b + head_idx * stride_q_h)
        q_ptrs = q_base + offs_m[:, None] * stride_q_t + offs_d[None, :]
        # 越界 query 位置 mask 为 0
        q_tile = tl.load(query_ptr + q_ptrs,
                         mask=q_valid_mask[:, None],
                         other=0.0)                         # (BLOCK_M, HEAD_DIM) fp16

        # ---- Online softmax 状态: 每行独立 ----
        m_i = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # ---- 迭代所有 page block ----
        num_prefix_blocks = tl.cdiv(prefix_len, BLOCK_SIZE)
        for block_num in range(max_num_blocks):
            cur_start = block_num * BLOCK_SIZE
            if cur_start < prefix_len:
                phys_block = tl.load(block_tables_ptr
                                     + batch_idx * stride_bt_b + block_num)

                # ---- 加载 K^T: (HEAD_DIM, BLOCK_SIZE) 连续 ----
                k_base = phys_block * stride_kt_blk + kv_head_idx * stride_kt_h
                kt_ptrs = k_base + offs_d[:, None] * BLOCK_SIZE + offs_bs[None, :]
                kt_block = tl.load(k_cache_ptr + kt_ptrs)  # (HEAD_DIM, BLOCK_SIZE) fp16

                # ---- scores = Q @ K^T: (BLOCK_M, BLOCK_SIZE) ----
                # tl.dot: (BLOCK_M, HEAD_DIM) @ (HEAD_DIM, BLOCK_SIZE)
                scores = tl.dot(q_tile, kt_block).to(tl.float32) * scale

                # key mask: 尾块中超出 prefix_len 的位置
                k_valid = (cur_start + offs_bs) < prefix_len
                scores = tl.where(k_valid[None, :], scores, float("-inf"))
                # query mask: 超出 valid_len 的 query 行
                scores = tl.where(q_valid_mask[:, None], scores, float("-inf"))

                # ---- Online softmax 更新 (行级) ----
                m_ij = tl.max(scores, axis=1)              # (BLOCK_M,)
                m_new = tl.maximum(m_i, m_ij)
                alpha = tl.exp(m_i - m_new)                # (BLOCK_M,)
                p = tl.exp(scores - m_new[:, None])        # (BLOCK_M, BLOCK_SIZE)

                l_i = l_i * alpha + tl.sum(p, axis=1)
                acc = acc * alpha[:, None]

                # ---- 加载 V: (BLOCK_SIZE, HEAD_DIM) 连续 ----
                v_base = phys_block * stride_vt_blk + kv_head_idx * stride_vt_h
                v_ptrs = v_base + offs_bs[:, None] * HEAD_DIM + offs_d[None, :]
                v_block = tl.load(v_cache_ptr + v_ptrs)    # (BLOCK_SIZE, HEAD_DIM) fp16

                # ---- acc += P @ V: (BLOCK_M, HEAD_DIM) ----
                acc += tl.dot(p.to(v_block.dtype), v_block).to(tl.float32)
                m_i = m_new

        # ---- 归一化 + 存储 ----
        safe_l = tl.where(l_i > 0.0, l_i, 1.0)
        out = acc / safe_l[:, None]
        lse_val = tl.where(l_i > 0.0, m_i + tl.log(l_i), float("-inf"))
        # 无效 query → -inf LSE
        lse_val = tl.where(q_valid_mask, lse_val, float("-inf"))

        # 逐行写出（BLOCK_M 行）
        out_ptrs = out_base + offs_m[:, None] * stride_o_t + offs_d[None, :]
        out_mask = q_valid_mask[:, None] & (offs_m[:, None] < MAX_Q_LEN)
        tl.store(output_ptr + out_ptrs, out, mask=out_mask)

        lse_ptrs = lse_base + offs_m * stride_l_t
        lse_mask = q_valid_mask & (offs_m < MAX_Q_LEN)
        tl.store(lse_ptr + lse_ptrs, lse_val, mask=lse_mask)


# ============================================================================
# PyTorch fallback — 基础算子
# ============================================================================
def pt_rms_norm(x, w, eps=1e-6):
    xf = x.to(torch.float32)
    return ((xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps))
            * w.float()).to(x.dtype)


def pt_silu_mul(g, u):
    return F.silu(g) * u


def fused_rms_norm(x, w, eps=1e-6):
    if HAS_TRITON and x.is_cuda:
        return triton_rms_norm(x, w, eps)
    return pt_rms_norm(x, w, eps)


def fused_silu_mul(g, u):
    if HAS_TRITON and g.is_cuda:
        return triton_silu_mul(g, u)
    return pt_silu_mul(g, u)


# ============================================================================
# Monkey-patch RMSNorm / MLP
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
# Rotary Position Embedding — per-layer
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


def _apply_rope_for_layer(model, q, k, position_ids):
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        rotary = model.model.rotary_emb
        try:
            cos, sin = rotary(k, position_ids)
        except Exception:
            cos, sin = rotary(q, position_ids)
        return _apply_rotary_pos_emb(q, k, cos, sin)

    # 某些旧/改造版本可能挂在别的位置，做一个兼容兜底
    if hasattr(model, "rotary_emb"):
        rotary = model.rotary_emb
        try:
            cos, sin = rotary(k, position_ids)
        except Exception:
            cos, sin = rotary(q, position_ids)
        return _apply_rotary_pos_emb(q, k, cos, sin)

    raise RuntimeError(
        "未找到 rotary_emb。当前 transformers/Qwen2 实现中通常应位于 model.model.rotary_emb。"
    )


# ============================================================================
# Paged KV Cache — 工业级布局
#   K cache: (max_blocks, num_kv_heads, head_dim, block_size)  ← K 转置存储
#   V cache: (max_blocks, num_kv_heads, block_size, head_dim)  ← V 正常存储
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

        # K 转置: attention 时可直接做 Q @ K^T 的矩阵乘
        k_shape = (max_blocks, num_kv_heads, head_dim, block_size)
        # V 正常: attention 时可直接做 P @ V 的矩阵乘
        v_shape = (max_blocks, num_kv_heads, block_size, head_dim)

        self.k_pools = [torch.zeros(k_shape, dtype=dtype, device=device)
                        for _ in range(num_layers)]
        self.v_pools = [torch.zeros(v_shape, dtype=dtype, device=device)
                        for _ in range(num_layers)]

        self.free_blocks: deque = deque(range(max_blocks))
        self.page_tables: Dict[int, List[int]] = {}
        self.seq_lens: Dict[int, int] = {}

    def _alloc_block(self):
        if not self.free_blocks:
            raise RuntimeError("PagedKVCache: 物理 block 用尽")
        return self.free_blocks.popleft()

    def allocate_seq(self, seq_idx: int, init_tokens: int = 0):
        n_blks = max(1, math.ceil(max(init_tokens, 1) / self.block_size))
        self.page_tables[seq_idx] = [self._alloc_block()
                                     for _ in range(n_blks)]
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
        tbl = torch.full((len(seq_ids), max_blks), -1,
                         dtype=torch.int32, device=self.device)
        for i, s in enumerate(seq_ids):
            pt = self.page_tables[s]
            tbl[i, :len(pt)] = torch.tensor(pt, dtype=torch.int32,
                                            device=self.device)
        return tbl

    def get_context_lens_tensor(self, seq_ids: List[int]):
        return torch.tensor([self.seq_lens[s] for s in seq_ids],
                            dtype=torch.long, device=self.device)

    def free_seq(self, seq_idx: int):
        if seq_idx in self.page_tables:
            self.free_blocks.extend(self.page_tables[seq_idx])
            del self.page_tables[seq_idx]
            del self.seq_lens[seq_idx]


# ============================================================================
# Store KV Cache — 适配新布局
# ============================================================================
def store_kvcache(key, value, k_cache, v_cache, slot_mapping, block_size):
    """
    key, value:   (num_tokens, num_kv_heads, head_dim)
    k_cache:      (max_blocks, num_kv_heads, head_dim, block_size) ← K^T
    v_cache:      (max_blocks, num_kv_heads, block_size, head_dim)
    slot_mapping: (num_tokens,) int64
    """
    num_tokens, num_kv_heads, head_dim = key.shape
    if HAS_TRITON and key.is_cuda:
        _store_kvcache_kernel[(num_tokens, num_kv_heads)](
            key.contiguous(), value.contiguous(),
            k_cache, v_cache, slot_mapping.contiguous(),
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
        )
    else:
        _pt_store_kvcache(key, value, k_cache, v_cache,
                          slot_mapping, block_size)


def _pt_store_kvcache(key, value, k_cache, v_cache, slot_mapping, block_size):
    """PyTorch fallback: K 转置存储"""
    num_tokens = key.shape[0]
    for t in range(num_tokens):
        slot = int(slot_mapping[t].item())
        if slot == -1:
            continue
        blk = slot // block_size
        off = slot % block_size
        # K^T: k_cache[blk, :, :, off] = key[t]    shape: (nkv, D) → cache[:, :, off]
        k_cache[blk, :, :, off] = key[t]           # (nkv, D) broadcast correctly
        # V:   v_cache[blk, :, off, :] = value[t]
        v_cache[blk, :, off, :] = value[t]


# ============================================================================
# Decode PagedAttention — 带 Split-K Flash-Decoding
# ============================================================================
def paged_attention_decode(query, k_cache, v_cache, block_tables,
                           context_lens, scale, num_heads, num_kv_heads,
                           head_dim, block_size):
    """
    query:        (B, num_heads, head_dim)
    k_cache:      (max_blocks, num_kv_heads, head_dim, block_size)
    v_cache:      (max_blocks, num_kv_heads, block_size, head_dim)
    block_tables: (B, max_num_blocks) int32
    context_lens: (B,) int64
    """
    B = query.shape[0]
    max_num_blocks = block_tables.shape[1]

    if HAS_TRITON and query.is_cuda:
        max_ctx = int(context_lens.max().item())
        max_kv_blocks = math.ceil(max_ctx / block_size) if max_ctx > 0 else 1
        num_splits = min(DECODE_NUM_SPLITS,
                         max(1, max_kv_blocks))

        if num_splits > 1 and max_kv_blocks >= 4:
            return _triton_decode_splitk(
                query, k_cache, v_cache, block_tables, context_lens,
                scale, num_heads, num_kv_heads, head_dim, block_size,
                max_num_blocks, num_splits)
        else:
            return _triton_decode_simple(
                query, k_cache, v_cache, block_tables, context_lens,
                scale, num_heads, num_kv_heads, head_dim, block_size,
                max_num_blocks)

    return _pt_paged_attn_decode(query, k_cache, v_cache, block_tables,
                                 context_lens, scale, num_heads,
                                 num_kv_heads, head_dim, block_size)


def _triton_decode_simple(query, k_cache, v_cache, block_tables,
                          context_lens, scale, num_heads, num_kv_heads,
                          head_dim, block_size, max_num_blocks):
    B = query.shape[0]
    output = torch.empty_like(query, dtype=torch.float32)
    _paged_attn_decode_kernel_v2[(B, num_heads)](
        output, query.contiguous(), k_cache, v_cache,
        block_tables, context_lens, scale, max_num_blocks,
        stride_qt_b=num_heads * head_dim,
        stride_qt_h=head_dim,
        stride_kt_blk=k_cache.stride(0),
        stride_kt_h=k_cache.stride(1),
        stride_vt_blk=v_cache.stride(0),
        stride_vt_h=v_cache.stride(1),
        stride_bt_b=block_tables.stride(0),
        NUM_HEADS=num_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
    )
    return output.to(query.dtype)


def _triton_decode_splitk(query, k_cache, v_cache, block_tables,
                          context_lens, scale, num_heads, num_kv_heads,
                          head_dim, block_size, max_num_blocks, num_splits):
    B = query.shape[0]
    device = query.device

    # 分配 partial 缓冲
    partial_out = torch.empty(B, num_heads, num_splits, head_dim,
                              device=device, dtype=torch.float32)
    partial_lse = torch.full((B, num_heads, num_splits), float("-inf"),
                             device=device, dtype=torch.float32)

    # Phase 1: Split-K 并行计算
    _paged_attn_decode_splitk_kernel[(B, num_heads, num_splits)](
        partial_out, partial_lse,
        query.contiguous(), k_cache, v_cache,
        block_tables, context_lens, scale, max_num_blocks,
        stride_qt_b=num_heads * head_dim,
        stride_qt_h=head_dim,
        stride_kt_blk=k_cache.stride(0),
        stride_kt_h=k_cache.stride(1),
        stride_vt_blk=v_cache.stride(0),
        stride_vt_h=v_cache.stride(1),
        stride_bt_b=block_tables.stride(0),
        stride_po_b=partial_out.stride(0),
        stride_po_h=partial_out.stride(1),
        stride_po_s=partial_out.stride(2),
        stride_pl_b=partial_lse.stride(0),
        stride_pl_h=partial_lse.stride(1),
        NUM_HEADS=num_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        NUM_SPLITS=num_splits,
    )

    # Phase 2: Reduction
    output = torch.empty(B, num_heads, head_dim,
                         device=device, dtype=torch.float32)
    _splitk_reduce_kernel[(B, num_heads)](
        output, partial_out, partial_lse,
        stride_o_b=num_heads * head_dim,
        stride_o_h=head_dim,
        stride_po_b=partial_out.stride(0),
        stride_po_h=partial_out.stride(1),
        stride_po_s=partial_out.stride(2),
        stride_pl_b=partial_lse.stride(0),
        stride_pl_h=partial_lse.stride(1),
        HEAD_DIM=head_dim,
        NUM_SPLITS=num_splits,
    )
    return output.to(query.dtype)


def _pt_paged_attn_decode(query, k_cache, v_cache, block_tables,
                          context_lens, scale, num_heads, num_kv_heads,
                          head_dim, block_size):
    """
    PyTorch fallback — 适配新 cache 布局
    k_cache: (max_blocks, num_kv_heads, head_dim, block_size)
    v_cache: (max_blocks, num_kv_heads, block_size, head_dim)
    """
    B = query.shape[0]
    gqa = num_heads // num_kv_heads
    o = torch.zeros_like(query, dtype=torch.float32)
    for i in range(B):
        ctx = int(context_lens[i].item())
        if ctx == 0:
            continue
        n_blks = math.ceil(ctx / block_size)
        kl, vl = [], []
        for b in range(n_blks):
            pb = int(block_tables[i, b].item())
            if pb == -1:
                continue
            length = min(block_size, ctx - b * block_size)
            # K^T: (nkv, D, BS) → 取 :length 列转回 (nkv, length, D)
            kl.append(k_cache[pb, :, :, :length].permute(0, 2, 1))
            # V:   (nkv, BS, D) → 取 :length 行
            vl.append(v_cache[pb, :, :length, :])
        ks = torch.cat(kl, 1)     # (Hkv, ctx, D)
        vs = torch.cat(vl, 1)     # (Hkv, ctx, D)
        if gqa > 1:
            ks = ks.unsqueeze(1).expand(-1, gqa, -1, -1) \
                   .reshape(num_heads, ctx, head_dim)
            vs = vs.unsqueeze(1).expand(-1, gqa, -1, -1) \
                   .reshape(num_heads, ctx, head_dim)
        qi = query[i].float()
        sc = torch.einsum('hd,hsd->hs', qi, ks.float()) * scale
        o[i] = torch.einsum('hs,hsd->hd',
                            F.softmax(sc, dim=-1), vs.float())
    return o.to(query.dtype)


# ============================================================================
# Prefill ① — Prefix Paged Attention
# ============================================================================
def paged_prefix_attention(q, k_cache, v_cache, block_tables, prefix_lens,
                           valid_lens_t, num_q_heads, num_kv_heads,
                           head_dim, block_size, scale):
    """
    q: (B, T, Hq, D)
    返回: out (B, T, Hq, D) fp32,  lse (B, T, Hq) fp32
    """
    B, T, _, D = q.shape
    max_num_blocks = block_tables.shape[1]

    if HAS_TRITON and q.is_cuda:
        q_c = q.contiguous()
        output = torch.empty((B, T, num_q_heads, D),
                             device=q.device, dtype=torch.float32)
        lse = torch.full((B, T, num_q_heads), float("-inf"),
                         device=q.device, dtype=torch.float32)

        # 确定 BLOCK_M: 对齐到 16 的最小值, 不超过 T
        block_m = min(PREFILL_BLOCK_M, triton.next_power_of_2(T))
        block_m = max(block_m, 16)     # tl.dot 最小 16
        num_q_tiles = math.ceil(T / block_m)

        _paged_prefix_attn_tiled_kernel[(num_q_tiles, B, num_q_heads)](
            output, lse, q_c, k_cache, v_cache,
            block_tables, prefix_lens, valid_lens_t,
            scale, max_num_blocks,
            stride_q_b=q_c.stride(0), stride_q_t=q_c.stride(1),
            stride_q_h=q_c.stride(2),
            stride_o_b=output.stride(0), stride_o_t=output.stride(1),
            stride_o_h=output.stride(2),
            stride_l_b=lse.stride(0), stride_l_t=lse.stride(1),
            stride_kt_blk=k_cache.stride(0), stride_kt_h=k_cache.stride(1),
            stride_vt_blk=v_cache.stride(0), stride_vt_h=v_cache.stride(1),
            stride_bt_b=block_tables.stride(0),
            NUM_HEADS=num_q_heads,
            NUM_KV_HEADS=num_kv_heads,
            HEAD_DIM=head_dim,
            BLOCK_SIZE=block_size,
            BLOCK_M=block_m,
            MAX_Q_LEN=T,
        )
        return output, lse

    return _pt_paged_prefix_attention_batched(
        q, k_cache, v_cache, block_tables, prefix_lens, valid_lens_t,
        num_q_heads, num_kv_heads, head_dim, block_size, scale)


def _pt_paged_prefix_attention_batched(
        q, k_cache, v_cache, block_tables, prefix_lens, valid_lens_t,
        num_q_heads, num_kv_heads, head_dim, block_size, scale):
    """
    PyTorch 批量 prefix attention — 适配新 cache 布局
    k_cache: (max_blocks, num_kv_heads, head_dim, block_size) ← K^T
    v_cache: (max_blocks, num_kv_heads, block_size, head_dim)
    """
    B, T, Hq, D = q.shape
    Hkv = num_kv_heads
    gqa = Hq // Hkv
    device = q.device

    max_prefix = int(prefix_lens.max().item())
    if max_prefix == 0:
        return (torch.zeros(B, T, Hq, D, device=device, dtype=torch.float32),
                torch.full((B, T, Hq), float("-inf"),
                           device=device, dtype=torch.float32))

    # gather KV from pages
    k_g = torch.zeros(B, max_prefix, Hkv, D, device=device, dtype=q.dtype)
    v_g = torch.zeros(B, max_prefix, Hkv, D, device=device, dtype=q.dtype)
    for b in range(B):
        ctx = int(prefix_lens[b].item())
        if ctx == 0:
            continue
        n_blks = math.ceil(ctx / block_size)
        off = 0
        for bi in range(n_blks):
            pb = int(block_tables[b, bi].item())
            if pb == -1:
                continue
            length = min(block_size, ctx - bi * block_size)
            # K^T: (nkv, D, BS) → (nkv, length, D)
            k_g[b, off:off + length] = k_cache[pb, :, :, :length] \
                                        .permute(0, 2, 1) \
                                        .transpose(0, 1)       # (length, nkv, D)
            # V:   (nkv, BS, D) → (length, nkv, D)
            v_g[b, off:off + length] = v_cache[pb, :, :length, :] \
                                        .transpose(0, 1)       # (length, nkv, D)
            off += length

    # (B, Hkv, max_prefix, D)
    kf = k_g.permute(0, 2, 1, 3).float()
    vf = v_g.permute(0, 2, 1, 3).float()
    if gqa > 1:
        kf = kf.unsqueeze(2).expand(B, Hkv, gqa, max_prefix, D) \
               .reshape(B, Hq, max_prefix, D)
        vf = vf.unsqueeze(2).expand(B, Hkv, gqa, max_prefix, D) \
               .reshape(B, Hq, max_prefix, D)

    qf = q.permute(0, 2, 1, 3).float()   # (B, Hq, T, D)

    scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale

    key_mask = (torch.arange(max_prefix, device=device).unsqueeze(0)
                >= prefix_lens.unsqueeze(1))
    key_mask = key_mask[:, None, None, :]

    q_mask = (torch.arange(T, device=device).unsqueeze(0)
              >= valid_lens_t.unsqueeze(1))
    q_mask = q_mask[:, None, :, None]

    full_mask = key_mask | q_mask
    scores = scores.masked_fill(full_mask, float("-inf"))

    max_s = scores.max(dim=-1, keepdim=True).values
    max_s = torch.clamp(max_s, min=-1e9)
    exp_s = torch.exp(scores - max_s)
    exp_s = exp_s.masked_fill(full_mask, 0.0)
    sum_exp = exp_s.sum(dim=-1, keepdim=True).clamp(min=1e-10)

    out = torch.matmul(exp_s / sum_exp, vf)
    lse = (max_s + torch.log(sum_exp)).squeeze(-1)

    q_pad_2d = q_mask.squeeze(-1)
    no_prefix = (prefix_lens == 0).unsqueeze(1).unsqueeze(2)
    lse = lse.masked_fill(q_pad_2d | no_prefix, float("-inf"))

    out = out.permute(0, 2, 1, 3).contiguous()
    lse = lse.permute(0, 2, 1).contiguous()
    return out, lse


# ============================================================================
# Prefill ② — Local Causal Attention（chunk 内因果自注意力，全批次并行）
# ============================================================================
def local_causal_attention(q, k, v, num_q_heads, num_kv_heads,
                           valid_lens_t, scale):
    """
    q: (B, T, Hq, D) post-RoPE
    k: (B, T, Hkv, D) post-RoPE
    v: (B, T, Hkv, D) raw
    """
    B, T, Hq, D = q.shape
    Hkv = num_kv_heads
    gqa = Hq // Hkv
    device = q.device

    qf = q.permute(0, 2, 1, 3).float()
    kf = k.permute(0, 2, 1, 3).float()
    vf = v.permute(0, 2, 1, 3).float()

    if gqa > 1:
        kf = kf.unsqueeze(2).expand(B, Hkv, gqa, T, D).reshape(B, Hq, T, D)
        vf = vf.unsqueeze(2).expand(B, Hkv, gqa, T, D).reshape(B, Hq, T, D)

    scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale

    causal = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool),
                        diagonal=1)

    pos = torch.arange(T, device=device)
    key_pad = (pos.unsqueeze(0) >= valid_lens_t.unsqueeze(1))
    q_pad   = key_pad

    full_mask = (causal[None, None, :, :]
                 | key_pad[:, None, None, :]
                 | q_pad[:, None, :, None])

    scores = scores.masked_fill(full_mask, float("-inf"))

    max_s = scores.max(dim=-1, keepdim=True).values
    max_s = torch.clamp(max_s, min=-1e9)
    exp_s = torch.exp(scores - max_s)
    exp_s = exp_s.masked_fill(full_mask, 0.0)
    sum_exp = exp_s.sum(dim=-1, keepdim=True).clamp(min=1e-10)

    out = torch.matmul(exp_s / sum_exp, vf)
    lse = (max_s + torch.log(sum_exp)).squeeze(-1)

    q_pad_mask = q_pad[:, None, :].expand_as(lse)
    lse = lse.masked_fill(q_pad_mask, float("-inf"))

    out = out.permute(0, 2, 1, 3).contiguous()
    lse = lse.permute(0, 2, 1).contiguous()
    return out, lse


# ============================================================================
# Prefill ③ — Online-Softmax Merge
# ============================================================================
def merge_attn_outputs(prefix_out, prefix_lse, local_out, local_lse):
    m = torch.maximum(prefix_lse, local_lse)
    w_p = torch.exp(prefix_lse - m)
    w_l = torch.exp(local_lse  - m)
    denom = w_p + w_l
    denom_safe = torch.where(denom > 0, denom,
                             torch.ones_like(denom))
    return ((w_p.unsqueeze(-1) * prefix_out
             + w_l.unsqueeze(-1) * local_out)
            / denom_safe.unsqueeze(-1))


# ============================================================================
# Prefill 完整一步: paged_extend_step
# ============================================================================
@torch.inference_mode()
def paged_extend_step(
    model,
    input_ids,         # (B_active, T)
    position_ids,      # (B_active, T)
    paged_cache: PagedKVCache,
    seq_ids: List[int],
    valid_lens: List[int],
):
    device = input_ids.device
    cfg = model.config
    num_q_heads  = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_q_heads)
    head_dim     = cfg.hidden_size // num_q_heads
    block_size   = paged_cache.block_size
    scale        = 1.0 / math.sqrt(head_dim)

    B, T = input_ids.shape
    hidden = model.model.embed_tokens(input_ids)

    prefix_lens  = paged_cache.get_context_lens_tensor(seq_ids)
    block_tables = paged_cache.get_block_table_tensor(seq_ids)
    valid_lens_t = torch.tensor(valid_lens, dtype=torch.long, device=device)

    flat_slots = []
    real_token_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
    for i, sid in enumerate(seq_ids):
        L = valid_lens[i]
        real_token_mask[i, :L] = True
        flat_slots.extend(paged_cache.get_slot_mapping_for_append(sid, L))
    flat_slots_t = torch.tensor(flat_slots, dtype=torch.long, device=device)

    for layer_idx, layer in enumerate(model.model.layers):
        residual = hidden
        h = layer.input_layernorm(hidden)

        q = layer.self_attn.q_proj(h).view(B, T, num_q_heads,  head_dim)
        k = layer.self_attn.k_proj(h).view(B, T, num_kv_heads, head_dim)
        v = layer.self_attn.v_proj(h).view(B, T, num_kv_heads, head_dim)

        q_r = q.transpose(1, 2).contiguous()
        k_r = k.transpose(1, 2).contiguous()
        q_r, k_r = _apply_rope_for_layer(model, q_r, k_r, position_ids)

        q_bt = q_r.transpose(1, 2).contiguous()   # (B, T, Hq, D)
        k_bt = k_r.transpose(1, 2).contiguous()   # (B, T, Hkv, D)

        # ① prefix paged attention
        prefix_out, prefix_lse = paged_prefix_attention(
            q_bt,
            paged_cache.k_pools[layer_idx],
            paged_cache.v_pools[layer_idx],
            block_tables, prefix_lens, valid_lens_t,
            num_q_heads, num_kv_heads, head_dim, block_size, scale,
        )

        # ② local causal attention
        local_out, local_lse = local_causal_attention(
            q_bt, k_bt, v, num_q_heads, num_kv_heads, valid_lens_t, scale,
        )

        # ③ merge
        attn_out = merge_attn_outputs(
            prefix_out.float(), prefix_lse.float(),
            local_out.float(), local_lse.float(),
        ).to(hidden.dtype)

        attn_out = attn_out * real_token_mask[:, :, None, None]
        attn_out = attn_out.reshape(B, T, num_q_heads * head_dim)
        attn_out = layer.self_attn.o_proj(attn_out)
        hidden = residual + attn_out

        residual = hidden
        h = layer.post_attention_layernorm(hidden)
        mlp_out = layer.mlp(h) * real_token_mask.unsqueeze(-1)
        hidden = residual + mlp_out

        # ④ store KV (RoPE'd K + raw V)
        k_real, v_real = [], []
        for i in range(B):
            L = valid_lens[i]
            if L > 0:
                k_real.append(k_bt[i, :L])   # (L, Hkv, D) post-RoPE
                v_real.append(v[i, :L])       # (L, Hkv, D) raw
        if k_real:
            store_kvcache(
                torch.cat(k_real, 0).contiguous(),
                torch.cat(v_real, 0).contiguous(),
                paged_cache.k_pools[layer_idx],
                paged_cache.v_pools[layer_idx],
                flat_slots_t, block_size,
            )

    # ⑤ advance seq_lens
    for i, sid in enumerate(seq_ids):
        paged_cache.append_len(sid, valid_lens[i])

    hidden = model.model.norm(hidden)
    return model.lm_head(hidden)


# ============================================================================
# Decode Step
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
    num_q_heads  = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_q_heads)
    head_dim     = cfg.hidden_size // num_q_heads
    block_size   = paged_cache.block_size
    scale        = 1.0 / math.sqrt(head_dim)
    B = token_ids.shape[0]

    hidden = model.model.embed_tokens(token_ids)

    slot_list = []
    for sid in seq_ids:
        slot_list.extend(paged_cache.get_slot_mapping_for_append(sid, 1))
    slot_mapping = torch.tensor(slot_list, dtype=torch.long, device=device)

    for layer_idx, layer in enumerate(model.model.layers):
        residual = hidden
        h = layer.input_layernorm(hidden)

        q = layer.self_attn.q_proj(h).view(B, 1, num_q_heads,  head_dim)
        k = layer.self_attn.k_proj(h).view(B, 1, num_kv_heads, head_dim)
        v = layer.self_attn.v_proj(h).view(B, 1, num_kv_heads, head_dim)

        q_r = q.transpose(1, 2).contiguous()
        k_r = k.transpose(1, 2).contiguous()
        q_r, k_r = _apply_rope_for_layer(model, q_r, k_r, position_ids)

        # store: RoPE'd K + raw V
        k_store = k_r.squeeze(2).contiguous()     # (B, Hkv, D)
        v_store = v.squeeze(1).contiguous()        # (B, Hkv, D)
        store_kvcache(
            k_store, v_store,
            paged_cache.k_pools[layer_idx],
            paged_cache.v_pools[layer_idx],
            slot_mapping, block_size,
        )

        # paged attention decode（含刚写入的 token）
        ctx_lens   = paged_cache.get_context_lens_tensor(seq_ids) + 1
        blk_tables = paged_cache.get_block_table_tensor(seq_ids)
        q_attn     = q_r.squeeze(2)               # (B, Hq, D)

        attn_out = paged_attention_decode(
            q_attn, paged_cache.k_pools[layer_idx],
            paged_cache.v_pools[layer_idx],
            blk_tables, ctx_lens, scale,
            num_q_heads, num_kv_heads, head_dim, block_size,
        )

        attn_out = attn_out.reshape(B, 1, num_q_heads * head_dim) \
                           .to(hidden.dtype)
        attn_out = layer.self_attn.o_proj(attn_out)
        hidden = residual + attn_out

        residual = hidden
        h = layer.post_attention_layernorm(hidden)
        hidden = residual + layer.mlp(h)

    for sid in seq_ids:
        paged_cache.append_len(sid, 1)

    hidden = model.model.norm(hidden)
    return model.lm_head(hidden[:, -1, :])


# ============================================================================
# 批量生成: Prefill (chunked extend) + Decode (paged attn)
# ============================================================================
@torch.inference_mode()
def batch_generate_paged(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = MAX_NEW_TOKENS,
    prefill_chunk_size: int = PREFILL_CHUNK_SIZE,
):
    device  = torch.device(DEVICE)
    pad_id  = tokenizer.pad_token_id or 0
    eos_ids = _eos_ids(tokenizer)
    eos_t   = torch.tensor(sorted(eos_ids), dtype=torch.long, device=device)
    B       = len(prompts)

    cfg          = model.config
    num_layers   = cfg.num_hidden_layers
    num_q_heads  = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_q_heads)
    head_dim     = cfg.hidden_size // num_q_heads

    enc = tokenizer(prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=2048).to(device)
    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    input_lengths  = attention_mask.sum(dim=1).tolist()
    padded_len     = input_ids.shape[1]

    max_total_tokens = max(input_lengths) + max_new_tokens + 8
    max_blks_per_seq = math.ceil(max_total_tokens / PAGE_BLOCK_SIZE) + 2
    total_max_blocks = max_blks_per_seq * B + 64

    paged_cache = PagedKVCache(
        num_layers=num_layers, num_kv_heads=num_kv_heads,
        head_dim=head_dim, block_size=PAGE_BLOCK_SIZE,
        max_blocks=total_max_blocks, device=device, dtype=DTYPE)

    seq_ids = list(range(B))
    for i in range(B):
        paged_cache.allocate_seq(i, init_tokens=input_lengths[i])

    real_token_lists = []
    for i in range(B):
        L = input_lengths[i]
        real_token_lists.append(input_ids[i, padded_len - L:])

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    # ==================== Prefill: chunked extend ====================
    last_logits_per_seq = [None] * B
    max_input_len = max(input_lengths)

    for chunk_start in range(0, max_input_len, prefill_chunk_size):
        chunk_end = min(chunk_start + prefill_chunk_size, max_input_len)

        active_ids  = []
        chunk_list  = []
        pos_list    = []
        valid_lens  = []

        for sid in range(B):
            seq = real_token_lists[sid]
            if chunk_start < seq.shape[0]:
                sub = seq[chunk_start:chunk_end]
                active_ids.append(sid)
                chunk_list.append(sub)
                valid_lens.append(sub.shape[0])
                pos_list.append(torch.arange(
                    chunk_start, chunk_start + sub.shape[0],
                    device=device, dtype=torch.long))

        if not active_ids:
            continue

        T = max(valid_lens)
        chunk_ids = torch.full((len(active_ids), T), pad_id,
                               dtype=torch.long, device=device)
        chunk_pos = torch.zeros((len(active_ids), T),
                                dtype=torch.long, device=device)
        for i in range(len(active_ids)):
            L = valid_lens[i]
            chunk_ids[i, :L] = chunk_list[i]
            chunk_pos[i, :L] = pos_list[i]

        logits = paged_extend_step(
            model, chunk_ids, chunk_pos,
            paged_cache, active_ids, valid_lens)

        for i, sid in enumerate(active_ids):
            last_logits_per_seq[sid] = logits[i, valid_lens[i] - 1].float()

    torch.cuda.synchronize(device)
    ttft = (time.perf_counter() - t0) * 1000.0

    # ==================== Decode ====================
    first_logits = torch.stack(last_logits_per_seq, dim=0)
    first_tokens = first_logits.argmax(dim=-1)

    unfinished     = torch.ones(B, dtype=torch.bool, device=device)
    generated      = [first_tokens.unsqueeze(1)]
    sample_lengths = torch.zeros(B, dtype=torch.long, device=device)
    cur_tokens     = first_tokens

    is_eos = (cur_tokens.unsqueeze(1) == eos_t.unsqueeze(0)).any(dim=1)
    sample_lengths += (unfinished & ~is_eos).long()
    unfinished = unfinished & ~is_eos
    positions = torch.tensor(input_lengths, dtype=torch.long, device=device)

    for _ in range(1, max_new_tokens):
        if not unfinished.any():
            break

        logits = paged_decode_step(
            model, cur_tokens.unsqueeze(1), positions.unsqueeze(1),
            paged_cache, seq_ids)

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

    gen_ids = torch.cat(generated, dim=1)
    lengths = sample_lengths.tolist()
    texts = []
    for i in range(B):
        L = min(max(lengths[i], 0), gen_ids.shape[1])
        texts.append(
            tokenizer.decode(gen_ids[i, :L].tolist(), skip_special_tokens=True)
            if L > 0 else "")
    for i in range(B):
        paged_cache.free_seq(i)

    return texts, lengths, input_lengths, ttft, total_ms


# ============================================================================
# 标准 HF 版本（对照/回退）
# ============================================================================
@torch.inference_mode()
def batch_generate_standard(model, tokenizer, prompts,
                            max_new_tokens: int = MAX_NEW_TOKENS):
    device  = torch.device(DEVICE)
    pad_id  = tokenizer.pad_token_id or 0
    eos_ids = _eos_ids(tokenizer)
    eos_t   = torch.tensor(sorted(eos_ids), dtype=torch.long, device=device)
    B       = len(prompts)

    enc = tokenizer(prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=2048).to(device)
    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    input_lengths  = attention_mask.sum(dim=1).tolist()

    unfinished     = torch.ones(B, dtype=torch.bool, device=device)
    generated      = []
    sample_lengths = torch.zeros(B, dtype=torch.long, device=device)
    past, cur_ids, cur_mask = None, input_ids, attention_mask

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    ttft = None

    for step in range(max_new_tokens):
        out = model(input_ids=cur_ids, attention_mask=cur_mask,
                    past_key_values=past, use_cache=True, return_dict=True)
        logits = out.logits[:, -1, :]
        past   = out.past_key_values
        next_tok = logits.argmax(dim=-1)
        if step == 0:
            torch.cuda.synchronize(device)
            ttft = (time.perf_counter() - t0) * 1000.0
        next_tok = torch.where(unfinished, next_tok,
                               torch.full_like(next_tok, pad_id))
        is_eos = (next_tok.unsqueeze(1) == eos_t.unsqueeze(0)).any(dim=1)
        sample_lengths += (unfinished & ~is_eos).long()
        generated.append(next_tok.unsqueeze(1))
        unfinished = unfinished & ~is_eos
        if not unfinished.any():
            break
        cur_ids  = next_tok.unsqueeze(1)
        cur_mask = torch.cat([cur_mask,
                              torch.ones(B, 1, device=device,
                                         dtype=cur_mask.dtype)], dim=1)

    torch.cuda.synchronize(device)
    total_ms = (time.perf_counter() - t0) * 1000.0
    if ttft is None:
        ttft = total_ms

    gen_ids = (torch.cat(generated, dim=1) if generated
               else torch.zeros(B, 0, dtype=torch.long, device=device))
    lengths = sample_lengths.tolist()
    texts = []
    for i in range(B):
        L = min(max(lengths[i], 0), gen_ids.shape[1])
        texts.append(
            tokenizer.decode(gen_ids[i, :L].tolist(), skip_special_tokens=True)
            if L > 0 else "")
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
                model(input_ids=enc["input_ids"],
                      attention_mask=enc["attention_mask"],
                      use_cache=True, return_dict=True)
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
        model_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None
    for attn in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=DTYPE, device_map=DEVICE,
                trust_remote_code=True, attn_implementation=attn)
            print(f"[OPT] Attention backend: {attn}")
            break
        except Exception:
            continue
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=DTYPE, device_map=DEVICE,
            trust_remote_code=True)
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

    n_p  = sum(p.numel() for p in model.parameters()) / 1e9
    vram = torch.cuda.memory_allocated(DEVICE) / 1e9
    print(f"[INFO] 就绪 | {n_p:.1f}B params | VRAM {vram:.2f} GB")
    return tokenizer, model


# ============================================================================
# 高层接口
# ============================================================================
def infer_all(tokenizer, model, prompts: list,
              batch_size: int = BATCH_SIZE,
              max_new_tokens: int = MAX_NEW_TOKENS,
              show_progress: bool = True,
              use_paged: bool = True,
              prefill_chunk_size: int = PREFILL_CHUNK_SIZE):
    n = len(prompts)
    if n == 0:
        return []

    enc_lens   = [len(tokenizer.encode(p, add_special_tokens=False))
                  for p in prompts]
    sorted_idx = sorted(range(n), key=lambda i: enc_lens[i])

    all_results = [None] * n
    num_batches = math.ceil(n / batch_size)

    for b in range(num_batches):
        s = b * batch_size
        e = min(s + batch_size, n)
        idx_b = sorted_idx[s:e]
        p_b   = [prompts[i] for i in idx_b]

        if use_paged:
            texts, out_lens, in_lens, ttft, total = batch_generate_paged(
                model, tokenizer, p_b,
                max_new_tokens=max_new_tokens,
                prefill_chunk_size=prefill_chunk_size)
        else:
            texts, out_lens, in_lens, ttft, total = batch_generate_standard(
                model, tokenizer, p_b,
                max_new_tokens=max_new_tokens)

        for j in range(len(p_b)):
            oi  = idx_b[j]
            tps = ((out_lens[j] / total * 1000.0)
                   if (total > 0 and out_lens[j] > 0) else 0.0)
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
            mode = "paged" if use_paged else "standard"
            print(f"  [batch {b+1}/{num_batches}] mode={mode} "
                  f"bs={len(p_b)} ttft={ttft:.0f}ms total={total:.0f}ms "
                  f"out_tok={sum(out_lens)} ({e}/{n} done)")

    return all_results


def infer_single(tokenizer, model, prompt: str,
                 use_paged: bool = True) -> dict:
    return infer_all(tokenizer, model, [prompt],
                     batch_size=1, show_progress=False,
                     use_paged=use_paged)[0]


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    import argparse

    pa = argparse.ArgumentParser()
    pa.add_argument("--model_path", type=str, required=True)
    pa.add_argument("--prompt", type=str,
                    default="请用三句话解释 KV Cache 的作用。")
    pa.add_argument("--batch_size", type=int, default=-1)
    pa.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    pa.add_argument("--prefill_chunk_size", type=int,
                    default=PREFILL_CHUNK_SIZE)
    pa.add_argument("--no_paged", action="store_true")
    args = pa.parse_args()

    tok, mdl = load_model(args.model_path)
    if args.batch_size <= 0:
        args.batch_size = _auto_batch_size(mdl, tok)
        print(f"[INFO] auto batch_size = {args.batch_size}")

    r = infer_single(tok, mdl, args.prompt, use_paged=not args.no_paged)
    print(f"\n{'='*60}")
    print(f"输出: {r['output'][:300]}")
    print(f"in={r['input_tokens']}  out={r['output_tokens']}")
    print(f"延迟={r['total_latency_ms']:.1f}ms  TTFT={r['ttft_ms']:.1f}ms")
    print(f"吞吐={r['throughput_tps']:.1f} tok/s")
    print(f"峰值显存={torch.cuda.max_memory_allocated(DEVICE)/1e9:.2f} GB")
    print(f"{'='*60}")