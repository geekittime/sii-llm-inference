"""
PagedAttention Triton Kernel
=============================

Triton 加速的 PagedAttention Decode Kernel。

核心思想:
  - KV Cache 以固定大小的 Block 存储在预分配的 GPU 显存池 (KVPool) 中
  - Decode 时 Query 只有 1 个 token，逐 Block 迭代，使用 online softmax
  - 直接按 block_table 索引访问 KVPool，无需将分散的 Block 拼接成连续 Tensor
  - 内置 GQA 支持: 多个 Query Head 共享同一组 KV Head

Grid: (batch_size, num_heads) — 每个 program 处理一个 (batch, head) 对

KVPool 布局:
  k_cache / v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
  - head_dim 是最内层维度 (连续)，保证 coalesced access
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F

# ── Triton 导入 ──────────────────────────────────────────────────

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    pass


# ── Triton Kernel ────────────────────────────────────────────────

if _HAS_TRITON:
    @triton.jit
    def _paged_attn_decode_kernel(
        Out,            # [B, num_heads, head_dim]
        Q,              # [B, num_heads, head_dim]
        K_cache,        # [num_blocks, block_size, num_kv_heads, head_dim]
        V_cache,        # [num_blocks, block_size, num_kv_heads, head_dim]
        Block_tables,   # [B, max_num_blocks_per_seq]
        Seq_lens,       # [B]  int32
        scale,          # float scalar
        stride_ob, stride_oh, stride_od,
        stride_qb, stride_qh, stride_qd,
        stride_kb, stride_ks, stride_kh, stride_kd,
        stride_vb, stride_vs, stride_vh, stride_vd,
        stride_btb, stride_btm,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        NUM_KV_GROUPS: tl.constexpr,
        MAX_NUM_BLOCKS: tl.constexpr,
    ):
        """
        PagedAttention Decode Kernel (单 token query, online softmax)

        每个 program 处理一个 (batch, query_head) 对:
          1. 加载 query 向量 [HEAD_DIM]
          2. 遍历该序列的所有 KV block
          3. 逐 block 计算 score = q @ K^T，在线更新 softmax 状态
          4. 累积 V 的加权和
          5. 归一化输出
        """
        bid = tl.program_id(0)   # batch index
        hid = tl.program_id(1)   # query head index
        kv_hid = hid // NUM_KV_GROUPS  # 对应的 KV head (GQA)

        seq_len = tl.load(Seq_lens + bid)

        # ── 加载 query: [HEAD_DIM] ──
        d_range = tl.arange(0, HEAD_DIM)
        q = tl.load(
            Q + bid * stride_qb + hid * stride_qh + d_range * stride_qd
        ).to(tl.float32)
        q = q * scale  # 预乘 scale

        # ── Online softmax 状态 ──
        m_i = float('-inf')  # running max
        l_i = 0.0            # running sum of exp
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)  # weighted sum

        s_range = tl.arange(0, BLOCK_SIZE)

        for block_idx in range(MAX_NUM_BLOCKS):
            block_start = block_idx * BLOCK_SIZE
            valid_mask = (block_start + s_range) < seq_len

            # 从 block_table 读取物理 block ID
            phys_block = tl.load(
                Block_tables + bid * stride_btb + block_idx,
                mask=block_start < seq_len,
                other=0,
            )

            # ── 加载 K block: [BLOCK_SIZE, HEAD_DIM] ──
            k_base = phys_block * stride_kb + kv_hid * stride_kh
            k = tl.load(
                K_cache + k_base
                + s_range[:, None] * stride_ks
                + d_range[None, :] * stride_kd,
                mask=valid_mask[:, None],
                other=0.0,
            ).to(tl.float32)

            # attention score: q·k^T → [BLOCK_SIZE]
            s = tl.sum(q[None, :] * k, axis=1)
            s = tl.where(valid_mask, s, float('-inf'))

            # ── 加载 V block: [BLOCK_SIZE, HEAD_DIM] ──
            v_base = phys_block * stride_vb + kv_hid * stride_vh
            v = tl.load(
                V_cache + v_base
                + s_range[:, None] * stride_vs
                + d_range[None, :] * stride_vd,
                mask=valid_mask[:, None],
                other=0.0,
            ).to(tl.float32)

            # ── Online softmax 更新 ──
            m_block = tl.max(s)
            m_new = tl.maximum(m_i, m_block)

            alpha = tl.exp(m_i - m_new)   # 旧状态的衰减系数
            p = tl.exp(s - m_new)          # 当前 block 的 softmax 权重

            l_i = l_i * alpha + tl.sum(p)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            m_i = m_new

        # ── 归一化输出 ──
        out = acc / tl.maximum(l_i, 1e-10)

        tl.store(
            Out + bid * stride_ob + hid * stride_oh + d_range * stride_od,
            out,
        )


# ── Python 接口 ─────────────────────────────────────────────────

def paged_attention_decode(
    query: torch.Tensor,          # [B, num_heads, 1, head_dim]
    k_cache: torch.Tensor,        # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: torch.Tensor,        # [num_blocks, block_size, num_kv_heads, head_dim]
    block_tables: torch.Tensor,   # [B, max_blocks_per_seq]  int32
    seq_lens: torch.Tensor,       # [B]  int32
    scale: float,
    num_kv_groups: int,
    block_size: int,
) -> torch.Tensor:                # [B, num_heads, 1, head_dim]
    """
    PagedAttention Decode — 单 token query attend 到分页 KV Cache。

    Triton 可用时走 GPU kernel，否则回退到 PyTorch 实现。
    """
    B, num_heads, _, head_dim = query.shape
    max_num_blocks = block_tables.shape[1]

    if _HAS_TRITON and query.is_cuda:
        q = query.squeeze(2).contiguous()   # [B, num_heads, head_dim]
        out = torch.empty_like(q)

        grid = (B, num_heads)
        _paged_attn_decode_kernel[grid](
            out, q, k_cache, v_cache, block_tables, seq_lens,
            scale,
            out.stride(0), out.stride(1), out.stride(2),
            q.stride(0), q.stride(1), q.stride(2),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
            block_tables.stride(0), block_tables.stride(1),
            BLOCK_SIZE=block_size,
            HEAD_DIM=head_dim,
            NUM_KV_GROUPS=num_kv_groups,
            MAX_NUM_BLOCKS=max_num_blocks,
        )
        return out.unsqueeze(2)  # [B, num_heads, 1, head_dim]

    else:
        return _paged_attention_decode_pytorch(
            query, k_cache, v_cache, block_tables, seq_lens,
            scale, num_kv_groups, block_size,
        )


def _paged_attention_decode_pytorch(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    num_kv_groups: int,
    block_size: int,
) -> torch.Tensor:
    """PyTorch 回退实现 (逐序列 gather + SDPA)"""
    B, num_heads, _, head_dim = query.shape
    output = torch.zeros_like(query)

    for b in range(B):
        sl = int(seq_lens[b].item())
        if sl == 0:
            continue
        num_blocks = (sl + block_size - 1) // block_size

        # 从 block pool 中 gather 出连续 K/V
        k_parts, v_parts = [], []
        for bi in range(num_blocks):
            block_id = int(block_tables[b, bi].item())
            end_pos = min((bi + 1) * block_size, sl) - bi * block_size
            k_parts.append(k_cache[block_id, :end_pos])
            v_parts.append(v_cache[block_id, :end_pos])

        # [sl, kv_heads, head_dim] → [1, kv_heads, sl, head_dim]
        full_k = torch.cat(k_parts, dim=0).unsqueeze(0).permute(0, 2, 1, 3)
        full_v = torch.cat(v_parts, dim=0).unsqueeze(0).permute(0, 2, 1, 3)

        # GQA 扩展
        if num_kv_groups > 1:
            full_k = full_k.repeat_interleave(num_kv_groups, dim=1)
            full_v = full_v.repeat_interleave(num_kv_groups, dim=1)

        attn = F.scaled_dot_product_attention(
            query[b : b + 1], full_k, full_v, scale=scale,
        )
        output[b] = attn

    return output
