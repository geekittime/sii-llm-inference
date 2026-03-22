"""
Block 管理模块
==============

PagedAttention 所需的 Block 管理:
- BlockAllocator: 物理 Block 索引分配器
- BlockTable: 序列到物理 Block 的映射表
"""

from llm_inference.block_manager.block_allocator import BlockAllocator
from llm_inference.block_manager.block_table import BlockTable

__all__ = [
    "BlockAllocator",
    "BlockTable",
]
