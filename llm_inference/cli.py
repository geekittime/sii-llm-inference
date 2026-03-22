#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Inference CLI
=================

命令行工具，支持连续和分页 KV-Cache 推理。
"""

import argparse
from llm_inference.inference import InferenceEngine


def main():
    parser = argparse.ArgumentParser(description="LLM 推理框架")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--prompt", type=str, default="请解释什么是大语言模型。", help="测试 prompt")
    parser.add_argument("--cache_type", type=str, default="continuous", choices=["continuous", "paged"],
                        help="Cache 类型")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="最大生成 token 数")
    args = parser.parse_args()

    # 创建推理引擎
    engine = InferenceEngine(
        model_path=args.model_path,
        cache_type=args.cache_type,
    )

    # 运行推理
    result = engine.infer_single(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )

    # 打印结果
    print(f"\n{'='*60}")
    print(f" Cache 类型: {args.cache_type.upper()}")
    print(f"{'='*60}")
    print(f"  输入: {result.prompt}")
    print(f"  输出: {result.output}")
    print(f"{'='*60}")
    print(f"  输入 tokens:  {result.input_tokens}")
    print(f"  输出 tokens:  {result.output_tokens}")
    print(f"  总延迟:       {result.total_latency_ms:.2f} ms")
    print(f"  TTFT:         {result.ttft_ms:.2f} ms")
    print(f"  吞吐量:       {result.throughput_tps:.2f} tokens/sec")
    print(f"  Cache 显存:   {engine.get_cache_memory_usage() / 1e9:.2f} GB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
