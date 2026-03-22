"""
连续 KV-Cache 实现
==================

基于 HuggingFace past_key_values 的连续内存 KV-Cache 实现。
"""

from typing import Dict, Tuple
import torch


class ContinuousKVCache:
    """
    连续 KV-Cache

    使用 HuggingFace Transformers 的 past_key_values 格式，
    内存连续分配，适合固定长度的序列。
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Args:
            num_layers: 模型层数
            num_heads: attention head 数
            head_dim: head 维度
            device: 设备
            dtype: 数据类型
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # {seq_id: {layer_idx: (K, V)}}
        self._cache: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {}

    def append(
        self,
        seq_id: int,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """
        追加 K/V 到缓存

        Args:
            seq_id: 序列 ID
            layer_idx: 层索引
            k: Key tensor, [num_tokens, num_kv_heads, head_dim]
            v: Value tensor, [num_tokens, num_kv_heads, head_dim]
        """
        if seq_id not in self._cache:
            self._cache[seq_id] = {}

        if layer_idx in self._cache[seq_id]:
            # 追加到现有缓存
            existing_k, existing_v = self._cache[seq_id][layer_idx]
            new_k = torch.cat([existing_k, k], dim=0)
            new_v = torch.cat([existing_v, v], dim=0)
            self._cache[seq_id][layer_idx] = (new_k, new_v)
        else:
            # 新建缓存
            self._cache[seq_id][layer_idx] = (k.clone(), v.clone())

    def get_num_cached_tokens(self, seq_id: int) -> int:
        """获取已缓存的 token 数量"""
        if seq_id not in self._cache:
            return 0
        # 检查第一层的 K cache
        kv = self._cache[seq_id].get(0)
        if kv is None:
            return 0
        return kv[0].shape[0]

    def reset(self) -> None:
        """重置所有缓存"""
        self._cache.clear()

    def get_memory_usage(self) -> int:
        """
        计算当前显存占用（字节）

        注意：这是估算值
        """
        total_tokens = 0
        for seq_cache in self._cache.values():
            for layer_idx in range(self.num_layers):
                kv = seq_cache.get(layer_idx)
                if kv is not None:
                    total_tokens += kv[0].shape[0]

        element_size = torch.tensor([], dtype=self.dtype).element_size()
        # 2 (K+V) * num_layers * total_tokens * num_heads * head_dim * element_size
        bytes_per_token = 2 * self.num_heads * self.head_dim * element_size
        return total_tokens * bytes_per_token
