"""
评测模块
==========

Cache 对比评测工具。
"""

from llm_inference.evaluation.compare_cache import (
    run_cache_comparison,
    print_comparison,
)

__all__ = [
    "run_cache_comparison",
    "print_comparison",
]
