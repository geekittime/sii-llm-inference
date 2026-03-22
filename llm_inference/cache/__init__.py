"""
KV-Cache 模块
=============

支持可插拔的 KV-Cache 实现:
- ContinuousKVCache: 连续内存 (HF 原生 past_key_values)
- PagedKVCache:      分页 KV 缓存 (KVPool + BlockAllocator + BlockTable)
- PagedCacheAdapter: HF Cache 接口适配器 (供 monkey-patched attention 使用)
"""

from llm_inference.cache.continuous_cache import ContinuousKVCache
from llm_inference.cache.paged_cache import KVPool, PagedKVCache, PagedCacheAdapter

__all__ = [
    "ContinuousKVCache",
    "KVPool",
    "PagedKVCache",
    "PagedCacheAdapter",
]
