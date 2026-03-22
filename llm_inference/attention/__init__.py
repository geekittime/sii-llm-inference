"""
Attention 模块
==============

PagedAttention Triton Kernel 及其 Python 接口。
"""

from llm_inference.attention.paged_attention import paged_attention_decode

__all__ = [
    "paged_attention_decode",
]
