"""
LLM Inference Framework
========================

模块化 LLM 推理框架，支持可插拔的 KV-Cache 实现。

核心组件:
  - cache/:         KV-Cache (连续 / 分页 KVPool)
  - attention/:     Triton PagedAttention Kernel
  - block_manager/: Block 索引分配与页表管理
  - inference/:     统一推理引擎

使用示例:
    from llm_inference import InferenceEngine

    engine = InferenceEngine(
        model_path="/path/to/model",
        cache_type="paged",  # or "continuous"
    )
    result = engine.infer_single("你好")
    print(result.output)
"""

__version__ = "0.3.0"

# Cache
from llm_inference.cache.continuous_cache import ContinuousKVCache
from llm_inference.cache.paged_cache import KVPool, PagedKVCache, PagedCacheAdapter

# Attention
from llm_inference.attention.paged_attention import paged_attention_decode

# Block Manager
from llm_inference.block_manager import BlockAllocator, BlockTable

# Inference
from llm_inference.inference.engine import InferenceEngine, InferenceResult

__all__ = [
    # Cache
    "ContinuousKVCache",
    "KVPool",
    "PagedKVCache",
    "PagedCacheAdapter",
    # Attention
    "paged_attention_decode",
    # Block Manager
    "BlockAllocator",
    "BlockTable",
    # Inference
    "InferenceEngine",
    "InferenceResult",
]
