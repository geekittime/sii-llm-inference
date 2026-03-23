import os, sys, time, math, sysconfig
from typing import List, Dict, Set, Optional
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================================
# 全局配置
# ============================================================================
DEVICE          = "cuda:0"
DTYPE           = torch.float16
MAX_NEW_TOKENS  = 1024
BATCH_SIZE      = 32
SEED            = 42
PAGE_BLOCK_SIZE = 16
DYNAMIC_REFILL_RATIO = 0.25
DYNAMIC_MIN_ADMIT = 4
DYNAMIC_PROGRESS_INTERVAL = 128


@dataclass
class RequestState:
    request_idx: int
    slot_id: int
    input_len: int
    position: int
    current_token: int
    sample_len: int
    generated_ids: List[int]

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
        HAS_TRITON = False
        print("[INFO] Triton 未安装，使用 PyTorch fallback")
        return
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
        print("[INFO] 回退到 PyTorch fallback")

# ============================================================================
# Triton 融合算子
# ============================================================================
if _TRITON_IMPORTED:

    # ─── RMSNorm ───
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

    # ─── SiLU×Mul ───
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

    # ─── Residual Add ───
    @triton.jit
    def _residual_add_k(X, Y, O, sx, sy, so, N, BLOCK: tl.constexpr):
        r = tl.program_id(0)
        for b in range(0, N, BLOCK):
            c = b + tl.arange(0, BLOCK)
            m = c < N
            xv = tl.load(X + r * sx + c, mask=m, other=0.0).to(tl.float32)
            yv = tl.load(Y + r * sy + c, mask=m, other=0.0).to(tl.float32)
            tl.store(O + r * so + c, xv + yv, mask=m)

    def triton_residual_add(x, y):
        s = x.shape
        x2 = x.reshape(-1, s[-1]).contiguous()
        y2 = y.reshape(-1, s[-1]).contiguous()
        M, N = x2.shape
        o = torch.empty_like(x2)
        BLK = min(triton.next_power_of_2(N), 4096)
        _residual_add_k[(M,)](x2, y2, o, x2.stride(0), y2.stride(0), o.stride(0), N, BLOCK=BLK)
        return o.reshape(s)

    # ─── Rotary Position Embedding ───
    @triton.jit
    def _rotary_pos_emb_k(X, COS, SIN, O,
                          sx, sc, ss, so,
                          N, HALF, BLOCK: tl.constexpr):
        r = tl.program_id(0)
        for b in range(0, N, BLOCK):
            c = b + tl.arange(0, BLOCK)
            m = c < N

            xv = tl.load(X + r * sx + c, mask=m, other=0.0).to(tl.float32)
            cv = tl.load(COS + r * sc + c, mask=m, other=0.0).to(tl.float32)
            sv = tl.load(SIN + r * ss + c, mask=m, other=0.0).to(tl.float32)

            first_half = c < HALF
            pair_c = tl.where(first_half, c + HALF, c - HALF)
            pair_v = tl.load(X + r * sx + pair_c, mask=pair_c < N, other=0.0).to(tl.float32)
            rot_v = tl.where(first_half, -pair_v, pair_v)

            tl.store(O + r * so + c, xv * cv + rot_v * sv, mask=m)

    def triton_apply_rotary_pos_emb(x, cos, sin):
        s = x.shape
        dim = s[-1]
        if dim % 2 != 0:
            raise ValueError(f"RoPE head_dim 必须为偶数，当前为 {dim}")

        x2 = x.reshape(-1, dim).contiguous()
        cos2 = cos.expand_as(x).reshape(-1, dim).contiguous()
        sin2 = sin.expand_as(x).reshape(-1, dim).contiguous()
        M, N = x2.shape
        o = torch.empty_like(x2)
        BLK = min(triton.next_power_of_2(N), 4096)
        _rotary_pos_emb_k[(M,)](
            x2, cos2, sin2, o,
            x2.stride(0), cos2.stride(0), sin2.stride(0), o.stride(0),
            N, N // 2, BLOCK=BLK,
        )
        return o.reshape(s)

    # ─── Store KV Cache — K 转置存储 ───
    #   K cache: (max_blocks, num_kv_heads, head_dim, block_size)
    #   V cache: (max_blocks, num_kv_heads, block_size, head_dim)
    #   input key/value: (num_tokens, num_kv_heads, head_dim)
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
        head_idx  = tl.program_id(1)

        slot_idx = tl.load(slot_mapping_ptr + token_idx)
        if slot_idx == -1:
            return

        block_idx    = slot_idx // block_size
        block_offset = slot_idx % block_size

        offs_d = tl.arange(0, head_dim)

        # Input: (num_tokens, num_kv_heads, head_dim)
        in_off = (token_idx * num_kv_heads * head_dim
                  + head_idx * head_dim + offs_d)

        key_vec = tl.load(key_ptr + in_off)
        val_vec = tl.load(value_ptr + in_off)

        # K cache 转置: k_cache[block_idx, head_idx, d, block_offset]
        k_off = (block_idx * num_kv_heads * head_dim * block_size
                 + head_idx * head_dim * block_size
                 + offs_d * block_size + block_offset)
        tl.store(k_cache_ptr + k_off, key_vec)

        # V cache 正常: v_cache[block_idx, head_idx, block_offset, d]
        v_off = (block_idx * num_kv_heads * block_size * head_dim
                 + head_idx * block_size * head_dim
                 + block_offset * head_dim + offs_d)
        tl.store(v_cache_ptr + v_off, val_vec)

    # ─── Paged Attention Decode — 向量化 + Online Softmax ───
    # Grid: (batch, num_heads)
    @triton.jit
    def _paged_attn_decode_kernel_v2(
        output_ptr,
        query_ptr,
        k_cache_ptr,          # (max_blocks, num_kv_heads, head_dim, block_size)
        v_cache_ptr,          # (max_blocks, num_kv_heads, block_size, head_dim)
        block_tables_ptr,
        context_lens_ptr,
        scale,
        stride_qt_b, stride_qt_h,
        stride_bt_b,
        NUM_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        MAX_NUM_BLOCKS: tl.constexpr,   # 修复: 改为 constexpr
    ):
        batch_idx = tl.program_id(0)
        head_idx  = tl.program_id(1)
        kv_head_idx = head_idx // (NUM_HEADS // NUM_KV_HEADS)

        ctx_len = tl.load(context_lens_ptr + batch_idx)

        offs_d  = tl.arange(0, HEAD_DIM)
        offs_bs = tl.arange(0, BLOCK_SIZE)

        q = tl.load(query_ptr + batch_idx * stride_qt_b
                     + head_idx * stride_qt_h + offs_d).to(tl.float32)

        # Online softmax 状态
        m_i = float("-inf")
        l_i = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        # K/V cache strides (手动计算, 避免传入非 constexpr)
        stride_kt_blk = NUM_KV_HEADS * HEAD_DIM * BLOCK_SIZE
        stride_kt_h   = HEAD_DIM * BLOCK_SIZE
        stride_vt_blk = NUM_KV_HEADS * BLOCK_SIZE * HEAD_DIM
        stride_vt_h   = BLOCK_SIZE * HEAD_DIM

        for block_num in range(MAX_NUM_BLOCKS):
            cur_start = block_num * BLOCK_SIZE
            if cur_start < ctx_len:
                phys_block = tl.load(block_tables_ptr
                                     + batch_idx * stride_bt_b
                                     + block_num)

                # K^T 块: (HEAD_DIM, BLOCK_SIZE) — 连续读取
                k_base = phys_block * stride_kt_blk + kv_head_idx * stride_kt_h
                kt_block = tl.load(k_cache_ptr + k_base
                                   + offs_d[:, None] * BLOCK_SIZE
                                   + offs_bs[None, :]).to(tl.float32)

                # scores = q @ K^T → (BLOCK_SIZE,)
                scores = tl.sum(q[:, None] * kt_block, axis=0) * scale

                valid_mask = (cur_start + offs_bs) < ctx_len
                scores = tl.where(valid_mask, scores, float("-inf"))

                # Online softmax 更新
                m_ij  = tl.max(scores)
                m_new = tl.maximum(m_i, m_ij)
                alpha = tl.exp(m_i - m_new)
                p     = tl.exp(scores - m_new)

                l_i = l_i * alpha + tl.sum(p)
                acc = acc * alpha

                # V 块: (BLOCK_SIZE, HEAD_DIM) — 连续读取
                v_base = phys_block * stride_vt_blk + kv_head_idx * stride_vt_h
                v_block = tl.load(v_cache_ptr + v_base
                                  + offs_bs[:, None] * HEAD_DIM
                                  + offs_d[None, :]).to(tl.float32)

                acc += tl.sum(p[:, None] * v_block, axis=0)
                m_i = m_new

        safe_l = tl.where(l_i > 0.0, l_i, 1.0)
        out = acc / safe_l

        tl.store(output_ptr + batch_idx * stride_qt_b
                 + head_idx * stride_qt_h + offs_d, out)


# ============================================================================
# PyTorch 原生 fallback
# ============================================================================
def pt_rms_norm(x, w, eps=1e-6):
    xf = x.to(torch.float32)
    return ((xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)) * w.float()).to(x.dtype)

def pt_silu_mul(g, u):
    return F.silu(g) * u

def pt_residual_add(x, y):
    return x + y

def fused_rms_norm(x, w, eps=1e-6):
    return triton_rms_norm(x, w, eps) if (HAS_TRITON and x.is_cuda) else pt_rms_norm(x, w, eps)

def fused_silu_mul(g, u):
    return triton_silu_mul(g, u) if (HAS_TRITON and g.is_cuda) else pt_silu_mul(g, u)

def fused_residual_add(x, y):
    return triton_residual_add(x, y) if (HAS_TRITON and x.is_cuda and y.is_cuda) else pt_residual_add(x, y)


# ============================================================================
# Store KV Cache — 适配 K^T 布局
# ============================================================================
def store_kvcache(key, value, k_cache, v_cache, slot_mapping, block_size):
    """
    key, value:    (num_tokens, num_kv_heads, head_dim)
    k_cache:       (max_blocks, num_kv_heads, head_dim, block_size) ← K^T
    v_cache:       (max_blocks, num_kv_heads, block_size, head_dim)
    slot_mapping:  (num_tokens,) int64
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
        _pt_store_kvcache(key, value, k_cache, v_cache, slot_mapping, block_size)


def _pt_store_kvcache(key, value, k_cache, v_cache, slot_mapping, block_size):
    """PyTorch fallback — K 转置存储"""
    num_tokens = key.shape[0]
    for t in range(num_tokens):
        slot = int(slot_mapping[t].item())
        if slot == -1:
            continue
        blk = slot // block_size
        off = slot % block_size
        # K^T: k_cache[blk, :, :, off] = key[t, :, :]  — 转置存储
        k_cache[blk, :, :, off] = key[t]
        # V:   v_cache[blk, :, off, :] = value[t, :, :]
        v_cache[blk, :, off, :] = value[t]


# ============================================================================
# Paged Attention Decode — Triton / PyTorch 双路径
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
        output = torch.empty_like(query, dtype=torch.float32)
        _paged_attn_decode_kernel_v2[(B, num_heads)](
            output, query.contiguous(), k_cache, v_cache,
            block_tables, context_lens, scale,
            stride_qt_b=num_heads * head_dim,
            stride_qt_h=head_dim,
            stride_bt_b=block_tables.stride(0),
            NUM_HEADS=num_heads,
            NUM_KV_HEADS=num_kv_heads,
            HEAD_DIM=head_dim,
            BLOCK_SIZE=block_size,
            MAX_NUM_BLOCKS=max_num_blocks,   # constexpr
        )
        return output.to(query.dtype)

    return _pt_paged_attn_decode(query, k_cache, v_cache, block_tables,
                                 context_lens, scale, num_heads,
                                 num_kv_heads, head_dim, block_size)


def _pt_paged_attn_decode(query, k_cache, v_cache, block_tables,
                           context_lens, scale, num_heads, num_kv_heads,
                           head_dim, block_size):
    """
    PyTorch fallback — K^T 布局
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
            # K^T cache: (nkv, D, BS) → 取前 length 列 → 转为 (nkv, length, D)
            kl.append(k_cache[pb, :, :, :length].permute(0, 2, 1))
            # V cache:   (nkv, BS, D)  → 取前 length 行
            vl.append(v_cache[pb, :, :length, :])
        ks = torch.cat(kl, 1)   # (Hkv, ctx, D)
        vs = torch.cat(vl, 1)   # (Hkv, ctx, D)
        if gqa > 1:
            ks = ks.unsqueeze(1).expand(-1, gqa, -1, -1).reshape(num_heads, ctx, head_dim)
            vs = vs.unsqueeze(1).expand(-1, gqa, -1, -1).reshape(num_heads, ctx, head_dim)
        qi = query[i].float()
        sc = torch.einsum('hd,hsd->hs', qi, ks.float()) * scale
        o[i] = torch.einsum('hs,hsd->hd', F.softmax(sc, dim=-1), vs.float())
    return o.to(query.dtype)


# ============================================================================
# PagedKVCache — K^T 布局
# ============================================================================
class PagedKVCache:
    """
    K cache: (max_blocks, num_kv_heads, head_dim, block_size)  ← K 转置
    V cache: (max_blocks, num_kv_heads, block_size, head_dim)
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

        k_shape = (max_blocks, num_kv_heads, head_dim, block_size)
        v_shape = (max_blocks, num_kv_heads, block_size, head_dim)

        self.k_pools = [torch.zeros(k_shape, dtype=dtype, device=device)
                        for _ in range(num_layers)]
        self.v_pools = [torch.zeros(v_shape, dtype=dtype, device=device)
                        for _ in range(num_layers)]

        self.free_blocks: deque = deque(range(max_blocks))
        self.page_tables: Dict[int, List[int]] = {}
        self.seq_lens: Dict[int, int] = {}

    @property
    def num_free(self):
        return len(self.free_blocks)

    def _alloc_block(self) -> int:
        if not self.free_blocks:
            raise RuntimeError("PagedKVCache: 物理 block 用尽")
        return self.free_blocks.popleft()

    def allocate_seq(self, seq_idx: int, num_tokens: int):
        n_blks = max(1, math.ceil(num_tokens / self.block_size))
        self.page_tables[seq_idx] = [self._alloc_block() for _ in range(n_blks)]
        self.seq_lens[seq_idx] = 0

    def ensure_slot(self, seq_idx: int) -> int:
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
        tbl = torch.full((len(seq_ids), max_blks), -1,
                         dtype=torch.int32, device=self.device)
        for i, s in enumerate(seq_ids):
            pt = self.page_tables[s]
            tbl[i, :len(pt)] = torch.tensor(pt, dtype=torch.int32, device=self.device)
        return tbl

    def get_context_lens_tensor(self, seq_ids: List[int]) -> torch.Tensor:
        return torch.tensor([self.seq_lens[s] for s in seq_ids],
                            dtype=torch.long, device=self.device)


# ============================================================================
# Rotary Position Embedding
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


def fused_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    if HAS_TRITON and q.is_cuda and k.is_cuda:
        return (
            triton_apply_rotary_pos_emb(q, cos, sin),
            triton_apply_rotary_pos_emb(k, cos, sin),
        )
    return (q * cos + _rotate_half(q) * sin,
            k * cos + _rotate_half(k) * sin)


def _module_rms_norm(mod, x):
    eps = getattr(mod, "variance_epsilon", getattr(mod, "eps", 1e-6))
    return fused_rms_norm(x, mod.weight, eps)


def _module_mlp(mod, x):
    return mod.down_proj(fused_silu_mul(mod.gate_proj(x), mod.up_proj(x)))


# ============================================================================
# Prefill KV → Paged Cache 拷贝 (适配 K^T 布局)
# ============================================================================
def _copy_prefill_kv_to_paged(hf_past, paged_cache: PagedKVCache,
                               input_lengths: List[int], padded_len: int,
                               seq_ids: Optional[List[int]] = None):
    """
    将 HF past_key_values 拷贝到 PagedKVCache (K^T 布局)。
    HF KV: (B, num_kv_heads, seq_len, head_dim)
    → K cache: (blocks, num_kv_heads, head_dim, block_size)  [转置]
    → V cache: (blocks, num_kv_heads, block_size, head_dim)  [正常]
    """
    B = len(input_lengths)
    device = paged_cache.device
    block_size = paged_cache.block_size
    if seq_ids is None:
        seq_ids = list(range(B))

    for layer_idx in range(paged_cache.num_layers):
        K_layer = hf_past[layer_idx][0]   # (B, nkvh, padded_len, hd)
        V_layer = hf_past[layer_idx][1]   # (B, nkvh, padded_len, hd)

        all_k, all_v, all_slots = [], [], []
        for i in range(B):
            actual_len = input_lengths[i]
            # left-padding: 实际内容在后面
            k_actual = K_layer[i, :, padded_len - actual_len:, :]  # (nkvh, actual_len, hd)
            v_actual = V_layer[i, :, padded_len - actual_len:, :]

            # → (actual_len, nkvh, hd) 匹配 store_kvcache 输入格式
            all_k.append(k_actual.permute(1, 0, 2).contiguous())
            all_v.append(v_actual.permute(1, 0, 2).contiguous())

            pt = paged_cache.page_tables[seq_ids[i]]
            for j in range(actual_len):
                blk_idx = j // block_size
                offset  = j % block_size
                slot = pt[blk_idx] * block_size + offset
                all_slots.append(slot)

        cat_k = torch.cat(all_k, dim=0)
        cat_v = torch.cat(all_v, dim=0)
        slots = torch.tensor(all_slots, dtype=torch.long, device=device)

        store_kvcache(cat_k, cat_v,
                      paged_cache.k_pools[layer_idx],
                      paged_cache.v_pools[layer_idx],
                      slots, block_size)

    for i in range(B):
        paged_cache.seq_lens[seq_ids[i]] = input_lengths[i]


@torch.inference_mode()
def _prefill_prompts_to_slots(
    model,
    tokenizer,
    prompts: List[str],
    slot_ids: List[int],
    paged_cache: PagedKVCache,
):
    device = torch.device(DEVICE)
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

    for slot_id, input_len in zip(slot_ids, input_lengths):
        paged_cache.allocate_seq(slot_id, input_len)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )

    _copy_prefill_kv_to_paged(
        outputs.past_key_values,
        paged_cache,
        input_lengths,
        padded_len,
        seq_ids=slot_ids,
    )

    first_logits = outputs.logits[:, -1, :].clone()
    del outputs
    return first_logits, input_lengths


# ============================================================================
# Decode Step — PagedAttention (适配 K^T 布局)
# ============================================================================
@torch.inference_mode()
def paged_decode_step(model, token_ids, position_ids,
                      paged_cache: PagedKVCache, seq_ids: List[int],
                      num_q_heads: int, num_kv_heads: int, head_dim: int):
    """
    token_ids:    (B, 1)
    position_ids: (B, 1)
    返回:         logits (B, vocab_size)
    """
    B = token_ids.shape[0]
    device = token_ids.device
    scale = 1.0 / math.sqrt(head_dim)
    block_size = paged_cache.block_size

    # Step 1: slot mapping
    slot_list = []
    for sid in seq_ids:
        slot_list.append(paged_cache.ensure_slot(sid))
    slot_mapping = torch.tensor(slot_list, dtype=torch.long, device=device)

    # 当前 decode step 的 block table / context length 在各层间相同，无需重复构造。
    ctx_lens = paged_cache.get_context_lens_tensor(seq_ids) + 1
    blk_tables = paged_cache.get_block_table_tensor(seq_ids)

    # Step 2: Embedding
    hidden = model.model.embed_tokens(token_ids)

    # Step 3: 逐层 Transformer
    for layer_idx, layer in enumerate(model.model.layers):
        residual = hidden
        h = _module_rms_norm(layer.input_layernorm, hidden)

        q = layer.self_attn.q_proj(h)
        k = layer.self_attn.k_proj(h)
        v = layer.self_attn.v_proj(h)

        q = q.view(B, 1, num_q_heads,  head_dim).transpose(1, 2)
        k = k.view(B, 1, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(B, 1, num_kv_heads, head_dim).transpose(1, 2)

        # RoPE
        cos, sin = model.model.rotary_emb(q, position_ids)
        q, k = fused_apply_rotary_pos_emb(q, k, cos, sin)

        # Store KV (RoPE'd K + raw V)
        # k: (B, nkvh, 1, hd) → squeeze → (B, nkvh, hd)
        k_store = k.squeeze(2).contiguous()
        v_store = v.squeeze(2).contiguous()
        store_kvcache(k_store, v_store,
                      paged_cache.k_pools[layer_idx],
                      paged_cache.v_pools[layer_idx],
                      slot_mapping, block_size)

        # Paged Attention Decode
        q_attn     = q.squeeze(2)   # (B, Hq, D)

        attn_out = paged_attention_decode(
            q_attn,
            paged_cache.k_pools[layer_idx],
            paged_cache.v_pools[layer_idx],
            blk_tables, ctx_lens,
            scale, num_q_heads, num_kv_heads, head_dim, block_size,
        )

        attn_out = attn_out.unsqueeze(1).reshape(B, 1, num_q_heads * head_dim)
        attn_out = layer.self_attn.o_proj(attn_out)
        hidden = fused_residual_add(residual, attn_out)

        residual = hidden
        h = _module_rms_norm(layer.post_attention_layernorm, hidden)
        hidden = fused_residual_add(residual, _module_mlp(layer.mlp, h))

    # Step 4: 推进 seq_lens
    for sid in seq_ids:
        paged_cache.increment_seq_len(sid)

    # Step 5: Final norm + LM head
    hidden = _module_rms_norm(model.model.norm, hidden)
    return model.lm_head(hidden[:, -1, :])


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
# 批量生成 — HF Prefill + Paged Decode
# ============================================================================
@torch.inference_mode()
def batch_generate_paged(model, tokenizer, prompts: List[str],
                         max_new_tokens: int = MAX_NEW_TOKENS):
    """
    Phase 1: HF 原生 prefill (SDPA) → 保证 prefill 正确性
    Phase 2: 拷贝 KV 到 PagedKVCache (K^T 布局)
    Phase 3: 自定义 decode 循环 (向量化 Triton PagedAttention)
    """
    device  = torch.device(DEVICE)
    pad_id  = tokenizer.pad_token_id or 0
    eos_ids = _eos_ids(tokenizer)
    eos_t   = torch.tensor(sorted(eos_ids), dtype=torch.long, device=device)
    B       = len(prompts)

    cfg = model.config
    num_layers   = cfg.num_hidden_layers
    num_q_heads  = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, 'num_key_value_heads', num_q_heads)
    head_dim     = cfg.hidden_size // num_q_heads

    # Tokenize (left-padding)
    enc = tokenizer(prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=2048).to(device)
    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    input_lengths  = attention_mask.sum(dim=1).tolist()
    padded_len     = input_ids.shape[1]

    # ─── Phase 1: HF Prefill ───
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    out = model(input_ids=input_ids, attention_mask=attention_mask,
                use_cache=True, return_dict=True)

    torch.cuda.synchronize(device)
    ttft = (time.perf_counter() - t0) * 1000.0

    # ─── Phase 2: 初始化 PagedKVCache & 拷贝 ───
    max_total_tokens = max(input_lengths) + max_new_tokens + 8
    max_blocks_per_seq = math.ceil(max_total_tokens / PAGE_BLOCK_SIZE) + 2
    total_max_blocks = max_blocks_per_seq * B + 64

    paged_cache = PagedKVCache(
        num_layers=num_layers, num_kv_heads=num_kv_heads,
        head_dim=head_dim, block_size=PAGE_BLOCK_SIZE,
        max_blocks=total_max_blocks, device=device, dtype=DTYPE)

    seq_ids = list(range(B))
    for i in range(B):
        paged_cache.allocate_seq(i, input_lengths[i])

    _copy_prefill_kv_to_paged(out.past_key_values, paged_cache,
                               input_lengths, padded_len)

    first_logits = out.logits[:, -1, :].clone()
    del out

    # ─── Phase 3: Decode 循环 ───
    first_tokens = first_logits.argmax(dim=-1)
    del first_logits

    unfinished     = torch.ones(B, dtype=torch.bool, device=device)
    generated      = [first_tokens.unsqueeze(1)]
    sample_lengths = torch.zeros(B, dtype=torch.long, device=device)
    cur_tokens     = first_tokens

    is_eos = (cur_tokens.unsqueeze(1) == eos_t.unsqueeze(0)).any(dim=1)
    sample_lengths += (unfinished & ~is_eos).long()
    unfinished = unfinished & ~is_eos

    positions = torch.tensor(input_lengths, dtype=torch.long, device=device)

    for step in range(1, max_new_tokens):
        if not unfinished.any():
            break

        active_idx = unfinished.nonzero(as_tuple=False).squeeze(1)
        active_seq_ids = active_idx.tolist()

        active_logits = paged_decode_step(
            model,
            cur_tokens.index_select(0, active_idx).unsqueeze(1),
            positions.index_select(0, active_idx).unsqueeze(1),
            paged_cache,
            active_seq_ids,
            num_q_heads,
            num_kv_heads,
            head_dim,
        )

        next_tokens = torch.full_like(cur_tokens, pad_id)
        next_tokens.index_copy_(0, active_idx, active_logits.argmax(dim=-1))

        is_eos = (next_tokens.unsqueeze(1) == eos_t.unsqueeze(0)).any(dim=1)
        sample_lengths += (unfinished & ~is_eos).long()
        generated.append(next_tokens.unsqueeze(1))

        unfinished = unfinished & ~is_eos
        positions.index_add_(0, active_idx, torch.ones_like(active_idx, dtype=positions.dtype))
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


@torch.inference_mode()
def batch_generate_paged_dynamic(
    model,
    tokenizer,
    prompts: List[str],
    max_batch_size: int = BATCH_SIZE,
    max_new_tokens: int = MAX_NEW_TOKENS,
    show_progress: bool = False,
):
    """
    动态 batch 调度:
      1. 维护固定数量的活跃 slot
      2. slot 空出后立即将等待队列中的请求 prefill 入场
      3. decode 每步仅处理活跃请求
      4. 序列完成后立即回收 paged cache block
    """
    device = torch.device(DEVICE)
    pad_id = tokenizer.pad_token_id or 0
    eos_ids = _eos_ids(tokenizer)
    eos_t = torch.tensor(sorted(eos_ids), dtype=torch.long, device=device)
    total_requests = len(prompts)

    if total_requests == 0:
        return [], [], [], 0.0, 0.0

    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_q_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_q_heads)
    head_dim = cfg.hidden_size // num_q_heads

    prompt_lens = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
    max_prompt_len = max(prompt_lens) if prompt_lens else 1
    max_concurrent = max(1, min(max_batch_size, total_requests))
    refill_threshold = min(
        max_concurrent,
        max(DYNAMIC_MIN_ADMIT, int(math.ceil(max_concurrent * DYNAMIC_REFILL_RATIO))),
    )

    max_total_tokens = max_prompt_len + max_new_tokens + 8
    max_blocks_per_seq = math.ceil(max_total_tokens / PAGE_BLOCK_SIZE) + 2
    total_max_blocks = max_blocks_per_seq * max_concurrent + 64

    paged_cache = PagedKVCache(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=PAGE_BLOCK_SIZE,
        max_blocks=total_max_blocks,
        device=device,
        dtype=DTYPE,
    )

    free_slots: deque = deque(range(max_concurrent))
    active: Dict[int, RequestState] = {}
    completed = 0
    next_request = 0
    last_reported_completed = 0

    texts = [""] * total_requests
    lengths = [0] * total_requests
    input_lengths = [0] * total_requests

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    ttft: Optional[float] = None

    while completed < total_requests:
        should_admit = (
            free_slots
            and next_request < total_requests
            and (
                not active
                or len(free_slots) >= refill_threshold
                or next_request + len(free_slots) >= total_requests
            )
        )
        if should_admit:
            admit_count = min(len(free_slots), total_requests - next_request)
            slot_ids = [free_slots.popleft() for _ in range(admit_count)]
            prompt_batch = prompts[next_request : next_request + admit_count]

            first_logits, batch_input_lens = _prefill_prompts_to_slots(
                model,
                tokenizer,
                prompt_batch,
                slot_ids,
                paged_cache,
            )

            if ttft is None:
                torch.cuda.synchronize(device)
                ttft = (time.perf_counter() - t0) * 1000.0

            first_tokens = first_logits.argmax(dim=-1)

            for local_idx, slot_id in enumerate(slot_ids):
                request_idx = next_request + local_idx
                first_token = int(first_tokens[local_idx].item())
                input_len = int(batch_input_lens[local_idx])
                input_lengths[request_idx] = input_len

                is_eos = first_token in eos_ids
                sample_len = 0 if is_eos else 1
                generated_ids = [first_token]

                if is_eos or sample_len >= max_new_tokens:
                    texts[request_idx] = tokenizer.decode(
                        generated_ids[:sample_len],
                        skip_special_tokens=True,
                    ) if sample_len > 0 else ""
                    lengths[request_idx] = sample_len
                    paged_cache.free_seq(slot_id)
                    free_slots.append(slot_id)
                    completed += 1
                else:
                    active[slot_id] = RequestState(
                        request_idx=request_idx,
                        slot_id=slot_id,
                        input_len=input_len,
                        position=input_len,
                        current_token=first_token,
                        sample_len=sample_len,
                        generated_ids=generated_ids,
                    )

            next_request += admit_count

            if show_progress and (completed - last_reported_completed >= DYNAMIC_PROGRESS_INTERVAL or completed == total_requests):
                print(
                    f"  [dynamic admit] admitted={admit_count} active={len(active)} "
                    f"completed={completed}/{total_requests} pending={total_requests-next_request}"
                )
                last_reported_completed = completed

        if not active:
            continue

        active_slot_ids = sorted(active.keys())
        cur_tokens = torch.tensor(
            [active[slot_id].current_token for slot_id in active_slot_ids],
            dtype=torch.long,
            device=device,
        )
        positions = torch.tensor(
            [active[slot_id].position for slot_id in active_slot_ids],
            dtype=torch.long,
            device=device,
        )

        logits = paged_decode_step(
            model,
            cur_tokens.unsqueeze(1),
            positions.unsqueeze(1),
            paged_cache,
            active_slot_ids,
            num_q_heads,
            num_kv_heads,
            head_dim,
        )

        next_tokens = logits.argmax(dim=-1)
        finished_slots: List[int] = []

        for local_idx, slot_id in enumerate(active_slot_ids):
            state = active[slot_id]
            next_token = int(next_tokens[local_idx].item())
            state.generated_ids.append(next_token)

            is_eos = next_token in eos_ids
            if not is_eos:
                state.sample_len += 1

            state.current_token = next_token
            state.position += 1

            if is_eos or state.sample_len >= max_new_tokens:
                texts[state.request_idx] = tokenizer.decode(
                    state.generated_ids[:state.sample_len],
                    skip_special_tokens=True,
                ) if state.sample_len > 0 else ""
                lengths[state.request_idx] = state.sample_len
                paged_cache.free_seq(slot_id)
                free_slots.append(slot_id)
                finished_slots.append(slot_id)
                completed += 1

        for slot_id in finished_slots:
            del active[slot_id]

        if show_progress and (
            completed - last_reported_completed >= DYNAMIC_PROGRESS_INTERVAL
            or completed == total_requests
        ):
            print(
                f"  [dynamic step] finished={len(finished_slots)} active={len(active)} "
                f"completed={completed}/{total_requests} pending={total_requests-next_request}"
            )
            last_reported_completed = completed

    torch.cuda.synchronize(device)
    total_ms = (time.perf_counter() - t0) * 1000.0
    if ttft is None:
        ttft = total_ms

    return texts, lengths, input_lengths, ttft, total_ms


# ============================================================================
# 标准批量生成 (backup)
# ============================================================================
@torch.inference_mode()
def batch_generate_standard(model, tokenizer, prompts,
                            max_new_tokens=MAX_NEW_TOKENS):
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
        next_tok = torch.where(unfinished, next_tok,
                               torch.full_like(next_tok, pad_id))
        is_eos = (next_tok.unsqueeze(1) == eos_t.unsqueeze(0)).any(dim=1)
        sample_lengths += (unfinished & ~is_eos).long()
        generated.append(next_tok.unsqueeze(1))
        unfinished = unfinished & ~is_eos
        if not unfinished.any():
            break
        cur_ids = next_tok.unsqueeze(1)
        cur_mask = torch.cat([cur_mask,
                              torch.ones(B, 1, device=device, dtype=cur_mask.dtype)], dim=1)

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
# Monkey-patch (RMSNorm + SwiGLU)
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
    print(
        f"[OPT] Decode kernels: {'RMSNorm, SwiGLU, ResidualAdd, RoPE, KVCache, PagedAttention' if HAS_TRITON else 'PyTorch fallback'}"
    )


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
    w = tokenizer("hello world", return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        for _ in range(3):
            model(**w, use_cache=True)
    torch.cuda.synchronize(DEVICE)

    n_p  = sum(p.numel() for p in model.parameters()) / 1e9
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
              use_paged: bool = True,
              use_dynamic_batch: bool = True):
    n = len(prompts)
    if n == 0:
        return []

    enc_lens   = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    sorted_idx = sorted(range(n), key=lambda i: enc_lens[i])

    if use_paged and use_dynamic_batch:
        sorted_prompts = [prompts[i] for i in sorted_idx]
        texts, out_lens, in_lens, ttft, total = batch_generate_paged_dynamic(
            model,
            tokenizer,
            sorted_prompts,
            max_batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            show_progress=show_progress,
        )

        all_results = [None] * n
        for j in range(n):
            original_idx = sorted_idx[j]
            tps = (out_lens[j] / total * 1000.0) if (total > 0 and out_lens[j] > 0) else 0.0
            all_results[original_idx] = {
                "prompt": prompts[original_idx],
                "output": texts[j],
                "input_tokens": in_lens[j],
                "output_tokens": out_lens[j],
                "total_latency_ms": round(total, 2),
                "ttft_ms": round(ttft, 2),
                "throughput_tps": round(tps, 2),
            }
        return all_results

    all_results = [None] * n
    num_batches = math.ceil(n / batch_size)
    gen_fn = batch_generate_paged if use_paged else batch_generate_standard

    for b in range(num_batches):
        s = b * batch_size
        e = min(s + batch_size, n)
        idx_b = sorted_idx[s:e]
        p_b   = [prompts[i] for i in idx_b]

        texts, out_lens, in_lens, ttft, total = gen_fn(
            model, tokenizer, p_b, max_new_tokens)

        for j in range(len(p_b)):
            oi  = idx_b[j]
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
            mode = "paged" if use_paged else "standard"
            print(f"  [batch {b+1}/{num_batches}] mode={mode} bs={len(p_b)} "
                  f"ttft={ttft:.0f}ms total={total:.0f}ms "
                  f"out_tok={sum(out_lens)} ({e}/{n} done)")

    return all_results


def infer_single(tokenizer, model, prompt: str,
                 use_paged: bool = True,
                 use_dynamic_batch: bool = True) -> dict:
    return infer_all(tokenizer, model, [prompt],
                     batch_size=1, show_progress=False,
                     use_paged=use_paged,
                     use_dynamic_batch=use_dynamic_batch)[0]


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
    pa.add_argument("--no_paged", action="store_true")
    pa.add_argument("--no_dynamic_batch", action="store_true")
    args = pa.parse_args()

    tok, mdl = load_model(args.model_path)
    if args.batch_size <= 0:
        args.batch_size = _auto_batch_size(mdl, tok)
        print(f"[INFO] auto batch_size = {args.batch_size}")

    r = infer_single(
        tok,
        mdl,
        args.prompt,
        use_paged=not args.no_paged,
        use_dynamic_batch=not args.no_dynamic_batch,
    )
    print(f"\n{'='*60}")
    print(f"输出: {r['output'][:300]}")
    print(f"in={r['input_tokens']}  out={r['output_tokens']}")
    print(f"延迟={r['total_latency_ms']:.1f}ms  TTFT={r['ttft_ms']:.1f}ms")
    print(f"吞吐={r['throughput_tps']:.1f} tok/s")
    print(f"峰值显存={torch.cuda.max_memory_allocated(DEVICE)/1e9:.2f} GB")
    print(f"{'='*60}")