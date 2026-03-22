"""
Block 索引分配器
================

纯索引管理，不持有 KV 数据。
实际 KV 存储在 KVPool 的预分配 GPU tensor 中。
"""

from collections import deque


class BlockAllocator:
    """
    物理 Block 索引分配器

    维护一个空闲 block_id 的队列，allocate() 弹出，free() 归还。
    类似 OS 物理页帧分配器。
    """

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self._free: deque = deque(range(num_blocks))

    def allocate(self) -> int:
        """分配一个 block，返回 block_id"""
        if not self._free:
            raise RuntimeError(
                f"Block pool exhausted (total={self.num_blocks})"
            )
        return self._free.popleft()

    def free(self, block_id: int) -> None:
        """释放一个 block"""
        self._free.append(block_id)

    def free_many(self, block_ids: list) -> None:
        """批量释放"""
        self._free.extend(block_ids)

    @property
    def num_free(self) -> int:
        return len(self._free)

    def reset(self) -> None:
        """重置，所有 block 回归空闲"""
        self._free = deque(range(self.num_blocks))
