"""
Block Table (Page Table)
========================

维护 seq_id → 物理 block 列表 + 已缓存 token 数 的映射。
类似 OS 页表。
"""

from typing import Dict, List, Tuple

import torch


class BlockTable:
    """序列到物理 Block 的映射表"""

    def __init__(self):
        self._blocks: Dict[int, List[int]] = {}
        self._seq_lens: Dict[int, int] = {}

    # ── 序列生命周期 ──────────────────────────────────────────

    def add_seq(self, seq_id: int) -> None:
        self._blocks[seq_id] = []
        self._seq_lens[seq_id] = 0

    def remove_seq(self, seq_id: int) -> List[int]:
        """移除序列，返回其持有的 block 列表 (供 allocator 回收)"""
        blocks = self._blocks.pop(seq_id, [])
        self._seq_lens.pop(seq_id, None)
        return blocks

    # ── Block 管理 ────────────────────────────────────────────

    def append_block(self, seq_id: int, block_id: int) -> None:
        self._blocks[seq_id].append(block_id)

    def get_blocks(self, seq_id: int) -> List[int]:
        return self._blocks.get(seq_id, [])

    # ── Token 计数 ────────────────────────────────────────────

    def get_seq_len(self, seq_id: int) -> int:
        return self._seq_lens.get(seq_id, 0)

    def set_seq_len(self, seq_id: int, length: int) -> None:
        self._seq_lens[seq_id] = length

    def add_tokens(self, seq_id: int, n: int) -> None:
        self._seq_lens[seq_id] = self._seq_lens.get(seq_id, 0) + n

    # ── 导出为 Tensor (供 Triton kernel) ──────────────────────

    def to_tensor(
        self,
        seq_ids: List[int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        导出为 Triton kernel 所需的 tensor 格式。

        Returns:
            block_tables: [B, max_blocks] int32
            seq_lens:     [B] int32
        """
        B = len(seq_ids)
        if B == 0:
            return (
                torch.zeros(0, 0, dtype=torch.int32, device=device),
                torch.zeros(0, dtype=torch.int32, device=device),
            )

        max_blocks = max(
            (len(self._blocks[sid]) for sid in seq_ids), default=0
        )
        bt = torch.zeros(B, max(max_blocks, 1), dtype=torch.int32, device=device)
        sl = torch.zeros(B, dtype=torch.int32, device=device)

        for i, sid in enumerate(seq_ids):
            blocks = self._blocks[sid]
            if blocks:
                bt[i, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
            sl[i] = self._seq_lens[sid]

        return bt, sl

    # ── 重置 ──────────────────────────────────────────────────

    def reset(self) -> None:
        self._blocks.clear()
        self._seq_lens.clear()
