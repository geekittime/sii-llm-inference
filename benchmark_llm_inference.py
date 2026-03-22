#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_llm_inference.py
==========================
使用 llm_inference 模块的吞吐量 & 延迟基准测试

支持两种模式：
  1. continuous: 连续 KV-Cache，使用批量 decode
  2. paged: 分页 KV-Cache (PagedAttention)

使用方式：
  # continuous 模式
  python benchmark_llm_inference.py \
    --model_path /path/to/model \
    --cache_type continuous \
    --batch_size 32 \
    --output results_continuous.json

  # paged 模式（自动占满 90% 显存）
  python benchmark_llm_inference.py \
    --model_path /path/to/model \
    --cache_type paged \
    --batch_size 32 \
    --output results_paged.json

  # paged 模式（调整显存利用率）
  python benchmark_llm_inference.py \
    --model_path /path/to/model \
    --cache_type paged \
    --gpu_memory_utilization 0.85 \
    --output results_paged.json

  # paged 模式（手动指定 block 数，跳过自动计算）
  python benchmark_llm_inference.py \
    --model_path /path/to/model \
    --cache_type paged \
    --num_blocks 2048 \
    --output results_paged.json
"""

import argparse
import json
import time
import torch
import numpy as np
from pathlib import Path

from llm_inference.inference import InferenceEngine

DEFAULT_PROMPT_FILE = Path(__file__).parent / "prompts.jsonl"


def load_prompts(prompt_file: str) -> list:
    """读取 prompts.jsonl，每行格式：{"id": 1, "prompt": "..."}"""
    prompts = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                prompt = obj["prompt"] if isinstance(obj, dict) else obj
                prompts.append(prompt)
    print(f"[INFO] 已加载 {len(prompts)} 条 prompt（来自 {prompt_file}）")
    return prompts


def run_benchmark(
    engine: InferenceEngine,
    prompts: list,
    batch_size: int = 32,
    max_new_tokens: int = 256,
) -> dict:
    """
    运行 benchmark

    Args:
        engine: InferenceEngine 实例
        prompts: prompt 列表
        batch_size: 批量大小
        max_new_tokens: 最大生成 token 数

    Returns:
        统计结果字典
    """
    print(f"\n[Benchmark] 共 {len(prompts)} 条 prompt，cache_type={engine.cache_type}")
    print(f"[Benchmark] batch_size={batch_size}, max_new_tokens={max_new_tokens}")
    print("-" * 68)

    device = engine._device

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    t_wall_start = time.perf_counter()

    # 两种路径均走 infer_batch；paged 在内部分叉到 _batch_decode_paged
    results = engine.infer_batch(
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        show_progress=True,
    )

    torch.cuda.synchronize(device)
    wall_time = time.perf_counter() - t_wall_start

    # 收集统计信息
    latencies = [r.total_latency_ms for r in results]
    ttfts = [r.ttft_ms for r in results]
    out_tokens = [r.output_tokens for r in results]
    in_tokens = [r.input_tokens for r in results]
    total_out = sum(out_tokens)
    total_in = sum(in_tokens)

    stats = {
        "cache_type": engine.cache_type,
        "batch_size": batch_size,
        "total_prompts": len(prompts),
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "wall_time_sec": round(wall_time, 2),
        "max_new_tokens_cfg": max_new_tokens,

        "overall_throughput_tps": round(total_out / wall_time, 2) if wall_time > 0 else 0,
        "input_throughput_tps": round(total_in / wall_time, 2) if wall_time > 0 else 0,

        "avg_latency_ms": round(float(np.mean(latencies)), 2),
        "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_latency_ms": round(float(np.percentile(latencies, 99)), 2),

        "avg_ttft_ms": round(float(np.mean(ttfts)), 2),
        "p95_ttft_ms": round(float(np.percentile(ttfts, 95)), 2),

        "peak_gpu_mem_gb": round(
            torch.cuda.max_memory_allocated(device) / 1e9, 3
        ),
        "cache_memory_gb": round(
            engine.get_cache_memory_usage() / 1e9, 3
        ),
    }
    return stats


def print_stats(stats: dict):
    """格式化打印 benchmark 结果"""
    labels = {
        "cache_type": "Cache 类型",
        "batch_size": "批量大小",
        "total_prompts": "测试 prompt 数",
        "total_input_tokens": "总输入 tokens",
        "total_output_tokens": "总输出 tokens",
        "wall_time_sec": "总耗时 (sec)",
        "max_new_tokens_cfg": "max_new_tokens 配置",
        "overall_throughput_tps": "整体吞吐率 (tokens/sec)",
        "input_throughput_tps": "输入吞吐率 (tokens/sec)",
        "avg_latency_ms": "平均延迟 (ms)",
        "p50_latency_ms": "P50 延迟 (ms)",
        "p95_latency_ms": "P95 延迟 (ms)",
        "p99_latency_ms": "P99 延迟 (ms)",
        "avg_ttft_ms": "平均 TTFT (ms)",
        "p95_ttft_ms": "P95 TTFT (ms)",
        "peak_gpu_mem_gb": "峰值显存 (GB)",
        "cache_memory_gb": "Cache 显存 (GB)",
    }
    print("\n" + "=" * 68)
    print(f" Benchmark 结果汇总（{stats.get('cache_type', 'unknown')}）")
    print("=" * 68)
    for key, label in labels.items():
        val = stats.get(key, "N/A")
        print(f"  {label:<40s}: {val}")
    print("=" * 68)


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM 推理吞吐 & 延迟基准测试（llm_inference 模块）"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="模型本地路径，例如 /aisi-nas/xiaoquanjia/base_model/Qwen2.5-14B-Instruct"
    )
    parser.add_argument(
        "--cache_type", type=str, default="continuous",
        choices=["continuous", "paged"],
        help="Cache 类型（默认：continuous）"
    )
    parser.add_argument(
        "--prompt_file", type=str, default=str(DEFAULT_PROMPT_FILE),
        help="prompt 文件路径（默认：prompts.jsonl）"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="批量大小（默认：32）"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256,
        help="最大生成 token 数（默认：256）"
    )
    parser.add_argument(
        "--block_size", type=int, default=16,
        help="PagedCache Block 大小（默认：16，仅 paged 有效）"
    )
    parser.add_argument(
        "--num_blocks", type=int, default=0,
        help=(
            "PagedCache Block 数量（仅 paged 有效）。"
            "0（默认）= 自动计算，根据 gpu_memory_utilization 占满剩余显存；"
            ">0 = 使用固定值"
        )
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.90,
        help=(
            "GPU 总显存利用率上限，含模型权重（默认：0.90，仅 paged + num_blocks=0 时生效）。"
            "可用于 KVPool 的空间 = total × gpu_memory_utilization − 已分配量。"
            "剩余比例作为 forward 激活值的安全余量。"
        )
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="结果保存路径（JSON），例如 results_continuous.json"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 创建推理引擎
    print(f"[INFO] 初始化 InferenceEngine（cache_type={args.cache_type}）...")
    engine = InferenceEngine(
        model_path=args.model_path,
        cache_type=args.cache_type,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        gpu_memory_utilization=args.gpu_memory_utilization,
        batch_size=args.batch_size,
    )

    # 加载 prompts
    prompts = load_prompts(args.prompt_file)

    # 运行测试
    stats = run_benchmark(
        engine=engine,
        prompts=prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    # 打印结果
    print_stats(stats)

    # 保存结果
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] 结果已保存至: {args.output}")


if __name__ == "__main__":
    main()
