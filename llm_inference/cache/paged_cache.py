"""
分页 KV-Cache 实现
==================

核心组件:
  - KVPool:           预分配的 GPU tensor 池，所有 KV 数据存此
  - PagedKVCache:     分页 KV 缓存管理 (pool + allocator + block_table)
  - PagedCacheAdapter: HF Cache 接口适配器，协调 decode 步骤的状态

设计原则:
  1. 所有 KV 数据存储在 KVPool 的连续 GPU tensor 中 (按 block_id 索引)
  2. BlockAllocator 只做索引管理，不持有数据
  3. BlockTable 维护 seq → block 列表 + seq_len 映射
  4. PagedCacheAdapter 作为 HF past_key_values 传入 model.forward()，
     同时作为 monkey-patched attention 的数据源
"""

from typing import Dict, List, Optional, Tuple

import torch

try:
    from transformers.cache_utils import Cache as HFCache
    _HAS_HF_CACHE = True
except ImportError:
    class HFCache:
        pass
    _HAS_HF_CACHE = False

from llm_inference.block_manager.block_allocator import BlockAllocator
from llm_inference.block_manager.block_table import BlockTable


# =====================================================================
# KVPool — 预分配 GPU 显存池
# =====================================================================

class KVPool:
    """
    预分配的 KV Cache GPU tensor 池。

    布局 (每层独立):
        k[layer]: [num_blocks, block_size, num_kv_heads, head_dim]
        v[layer]: [num_blocks, block_size, num_kv_heads, head_dim]

    head_dim 是最内层维度 (连续)，保证 Triton kernel 的 coalesced access。
    所有 block 在初始化时一次性分配，运行时只做索引写入，零额外显存分配。
    """

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        shape = (num_blocks, block_size, num_kv_heads, head_dim)
        # 每层独立 tensor (避免 5D tensor 的 stride 复杂性)
        self.k: List[torch.Tensor] = [
            torch.zeros(shape, device=device, dtype=dtype) for _ in range(num_layers)
        ]
        self.v: List[torch.Tensor] = [
            torch.zeros(shape, device=device, dtype=dtype) for _ in range(num_layers)
        ]

    def write_slot(
        self,
        layer_idx: int,
        block_id: int,
        offset: int,
        k_token: torch.Tensor,   # [num_kv_heads, head_dim]
        v_token: torch.Tensor,
    ) -> None:
        """写入单个 token 到指定 (block, offset)"""
        self.k[layer_idx][block_id, offset] = k_token
        self.v[layer_idx][block_id, offset] = v_token

    def write_block_range(
        self,
        layer_idx: int,
        block_id: int,
        start: int,
        k_chunk: torch.Tensor,   # [n_tokens, num_kv_heads, head_dim]
        v_chunk: torch.Tensor,
    ) -> None:
        """写入多个 token 到一个 block 的连续位置"""
        n = k_chunk.shape[0]
        self.k[layer_idx][block_id, start : start + n] = k_chunk
        self.v[layer_idx][block_id, start : start + n] = v_chunk

    def get_memory_usage(self) -> int:
        """已分配的总 GPU 显存 (字节)"""
        if not self.k:
            return 0
        per_tensor = self.k[0].element_size() * self.k[0].nelement()
        return per_tensor * self.num_layers * 2  # K + V


# =====================================================================
# PagedKVCache — 分页 KV 缓存
# =====================================================================

class PagedKVCache:
    """
    分页 KV-Cache 管理器。

    整合 KVPool + BlockAllocator + BlockTable，提供:
      - allocate_seq / free_seq:  序列生命周期管理
      - append:                   写入 KV 数据到 pool (自动分配 block)
      - get_kv_cache:             获取某层的 (k_cache, v_cache) tensor 引用
      - get_memory_usage:         显存统计
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        num_blocks: int = 1000,
        device: torch.device = torch.device("cuda:0"),
        dtype: torch.dtype = torch.float16,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self._device = device
        self.dtype = dtype

        self.pool = KVPool(
            num_layers=num_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
        )
        self.allocator = BlockAllocator(num_blocks)
        self.block_table = BlockTable()

    # ── 序列管理 ──────────────────────────────────────────────

    def allocate_seq(self, seq_id: int) -> None:
        """注册新序列 (不预分配 block，由 append 动态申请)"""
        self.block_table.add_seq(seq_id)

    def free_seq(self, seq_id: int) -> None:
        """释放序列，回收其 block"""
        blocks = self.block_table.remove_seq(seq_id)
        self.allocator.free_many(blocks)

    # ── KV 写入 ──────────────────────────────────────────────

    def append(
        self,
        seq_id: int,
        layer_idx: int,
        k: torch.Tensor,   # [n_tokens, num_kv_heads, head_dim]
        v: torch.Tensor,
        start_pos: int = None,
    ) -> None:
        """
        追加 KV token 到指定序列的 paged cache。

        自动分配新 block (按需)，按 block_size 分片写入 KVPool。
        仅在 layer_idx == 0 时更新 seq_len (避免每层重复计数)。

        Args:
            start_pos: 显式指定写入起始位置。为 None 时使用 seq_len（适用于 decode 逐 token 追加）。
                       prefill 阶段应传入 0，避免 layer>0 时 seq_len 已被 layer 0 更新导致写入位置错误。
        """
        n_tokens = k.shape[0]
        if n_tokens == 0:
            return

        current_len = start_pos if start_pos is not None else self.block_table.get_seq_len(seq_id)
        blocks = self.block_table.get_blocks(seq_id)

        written = 0
        pos = current_len
        while written < n_tokens:
            block_idx = pos // self.block_size
            offset = pos % self.block_size

            # 分配新 block (如果需要)
            while block_idx >= len(blocks):
                new_block = self.allocator.allocate()
                self.block_table.append_block(seq_id, new_block)
                blocks = self.block_table.get_blocks(seq_id)

            block_id = blocks[block_idx]
            # 本 block 还能写入的 token 数
            can_write = min(self.block_size - offset, n_tokens - written)

            self.pool.write_block_range(
                layer_idx, block_id, offset,
                k[written : written + can_write],
                v[written : written + can_write],
            )
            written += can_write
            pos += can_write

        # 仅在第一层更新 seq_len (每个 decode step 调用 append 的层序是 0..L-1)
        if layer_idx == 0:
            self.block_table.set_seq_len(seq_id, current_len + n_tokens)

    def ensure_blocks_for(self, seq_id: int, total_tokens: int) -> None:
        """确保序列有足够的 block 容纳 total_tokens 个 token"""
        needed_blocks = (total_tokens + self.block_size - 1) // self.block_size
        blocks = self.block_table.get_blocks(seq_id)
        while len(blocks) < needed_blocks:
            new_block = self.allocator.allocate()
            self.block_table.append_block(seq_id, new_block)
            blocks = self.block_table.get_blocks(seq_id)

    # ── 查询 ─────────────────────────────────────────────────

    def get_kv_cache(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取某层的 (k_cache, v_cache) tensor 引用 (供 Triton kernel 使用)"""
        return self.pool.k[layer_idx], self.pool.v[layer_idx]

    def get_memory_usage(self) -> int:
        return self.pool.get_memory_usage()

    @property
    def device(self) -> torch.device:
        return self._device

    # ── 重置 ─────────────────────────────────────────────────

    def reset(self) -> None:
        # 先回收所有已分配的 block（避免内存泄漏）
        for seq_id in list(self.block_table._blocks.keys()):
            blocks = self.block_table.remove_seq(seq_id)
            self.allocator.free_many(blocks)
        # 再 reset block_table（不需要 reset allocator，因为已经回收了所有 block）
        self.block_table.reset()
        # 不清零 pool tensor (下次写入时会覆盖)


# =====================================================================
# PagedCacheAdapter — HF Cache 接口适配器
# =====================================================================

class PagedCacheAdapter:
    """
    将 PagedKVCache 桥接到 HF transformers 的 Cache 接口。

    使用协议 (在 engine 的 decode 循环中):
        adapter = PagedCacheAdapter(cache, seq_ids)
        for step in range(max_new_tokens):
            adapter.begin_step(active_mask)   # 分配 block，准备 tensor
            model.forward(..., past_key_values=adapter)
            adapter.end_step()                # 推进 seq_len

    monkey-patched attention forward 在 model.forward() 内部调用:
        adapter.write_kv(layer_idx, key_states, value_states)
        k_cache, v_cache = adapter.get_kv_cache(layer_idx)
        # → paged_attention_decode(query, k_cache, v_cache, adapter.block_tables_tensor, ...)
    """

    def __init__(self, cache: PagedKVCache, seq_ids: List[int]):
        self.cache = cache
        self.seq_ids = seq_ids
        self._B = len(seq_ids)

        # 由 begin_step() 填充，供 Triton kernel 和 monkey-patch 使用
        self.block_tables_tensor: Optional[torch.Tensor] = None  # [B, max_blocks]
        self.seq_lens_tensor: Optional[torch.Tensor] = None      # [B] int32
        self._write_positions: Dict[int, int] = {}
        self._active_mask: Optional[torch.Tensor] = None

    # ── Decode 步骤协调 ──────────────────────────────────────

    def begin_step(self, active_mask: torch.Tensor) -> None:
        """
        decode 步骤开始前调用。

        1. 为活跃序列在写入位置分配 block (如果需要)
        2. 计算 seq_lens_tensor (包含即将写入的新 token，即 +1)
        3. 计算 block_tables_tensor
        """
        self._active_mask = active_mask
        device = self.cache.device

        # 1. 分配 block，记录写入位置
        self._write_positions.clear()
        for i, sid in enumerate(self.seq_ids):
            if not active_mask[i]:
                continue
            pos = self.cache.block_table.get_seq_len(sid)
            self._write_positions[sid] = pos
            # 确保 pos 位置所在的 block 已分配
            self.cache.ensure_blocks_for(sid, pos + 1)

        # 2. seq_lens: 活跃序列 +1 (新 token 写入后内核需要 attend 到它)
        lens = []
        for i, sid in enumerate(self.seq_ids):
            sl = self.cache.block_table.get_seq_len(sid)
            if active_mask[i]:
                sl += 1
            lens.append(sl)
        self.seq_lens_tensor = torch.tensor(lens, dtype=torch.int32, device=device)

        # 3. block_tables
        self.block_tables_tensor, _ = self.cache.block_table.to_tensor(
            self.seq_ids, device
        )

    def end_step(self) -> None:
        """decode 步骤结束后调用: 推进活跃序列的 seq_len"""
        if self._active_mask is None:
            return
        for i, sid in enumerate(self.seq_ids):
            if self._active_mask[i]:
                self.cache.block_table.add_tokens(sid, 1)

    # ── 供 monkey-patched attention 调用 ─────────────────────

    def write_kv(
        self,
        layer_idx: int,
        key_states: torch.Tensor,    # [B, num_kv_heads, 1, head_dim]
        value_states: torch.Tensor,
    ) -> None:
        """将新 token 的 K, V 写入 paged cache pool"""
        for i, sid in enumerate(self.seq_ids):
            if sid not in self._write_positions:
                continue  # 已结束的序列，跳过
            pos = self._write_positions[sid]
            block_idx = pos // self.cache.block_size
            offset = pos % self.cache.block_size
            blocks = self.cache.block_table.get_blocks(sid)
            block_id = blocks[block_idx]

            # key_states[i, :, 0, :] → [num_kv_heads, head_dim]
            self.cache.pool.write_slot(
                layer_idx, block_id, offset,
                key_states[i, :, 0, :],
                value_states[i, :, 0, :],
            )

    def get_kv_cache(
        self, layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取该层的 (k_cache, v_cache) tensor 引用"""
        return self.cache.get_kv_cache(layer_idx)

    # ── HF Cache 接口 (供 model forward 内部使用) ────────────

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """返回最长序列的已缓存 token 数 (HF 用于 causal mask 计算)"""
        return max(
            (self.cache.block_table.get_seq_len(sid) for sid in self.seq_ids),
            default=0,
        )

    def get_max_length(self) -> Optional[int]:
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    @property
    def seen_tokens(self) -> int:
        return self.get_seq_length()

    def get_mask_sizes(self, cache_position, layer_idx: int):
        """HF Cache 接口：返回 mask 计算所需的尺寸"""
        return self.get_seq_length(layer_idx), 0

    @property
    def is_compileable(self) -> bool:
        """HF Cache 接口：是否可编译"""
        return False

    def update(self, key_states, value_states, layer_idx, cache_kwargs):
        """
        HF Cache 接口：更新缓存

        注意：PagedCacheAdapter 的写入由 write_kv 方法处理，
        这里返回原始的 key_states, value_states 以兼容 HF 接口
        """
        return key_states, value_states
