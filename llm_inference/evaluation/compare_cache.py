"""
Cache 对比评测
==============

对比连续 KV-Cache 和 PagedAttention 的性能。
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np

from llm_inference.inference import InferenceEngine


def load_prompts(path: str) -> List[str]:
    """加载测试 prompts"""
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj["prompt"] if isinstance(obj, dict) else obj
            prompts.append(prompt)
    return prompts


def run_cache_comparison(
    model_path: str,
    prompts: List[str],
    cache_types: List[str] = ["continuous", "paged"],
    batch_size: int = 1,
    max_new_tokens: int = 256,
) -> Dict[str, Dict]:
    """
    运行 Cache 对比测试

    Args:
        model_path: 模型路径
        prompts: 测试 prompts
        cache_types: Cache 类型列表
        batch_size: 批量大小
        max_new_tokens: 最大生成 token 数

    Returns:
        {cache_type: {metrics}}
    """
    results = {}

    for cache_type in cache_types:
        print(f"\n{'='*60}")
        print(f" 测试 {cache_type.upper()} Cache")
        print(f"{'='*60}")

        # 创建推理引擎
        engine = InferenceEngine(
            model_path=model_path,
            cache_type=cache_type,
            block_size=16,
            num_blocks=1000,
        )

        # 运行推理
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        wall_start = time.perf_counter()

        inference_results = engine.infer_batch(
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
        )

        torch.cuda.synchronize()
        wall_time = time.perf_counter() - wall_start

        # 收集指标
        latencies = [r.total_latency_ms for r in inference_results]
        ttfts = [r.ttft_ms for r in inference_results]
        out_tokens = [r.output_tokens for r in inference_results]
        total_out = sum(out_tokens)

        cache_memory_gb = engine.get_cache_memory_usage() / 1e9
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9

        results[cache_type] = {
            "total_prompts": len(prompts),
            "batch_size": batch_size,
            "total_output_tokens": total_out,
            "wall_time_sec": round(wall_time, 2),
            "overall_throughput_tps": round(total_out / wall_time, 2) if wall_time > 0 else 0,
            "avg_latency_ms": round(float(np.mean(latencies)), 2),
            "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
            "p99_latency_ms": round(float(np.percentile(latencies, 99)), 2),
            "avg_ttft_ms": round(float(np.mean(ttfts)), 2),
            "p95_ttft_ms": round(float(np.percentile(ttfts, 95)), 2),
            "cache_memory_gb": round(cache_memory_gb, 3),
            "peak_memory_gb": round(peak_memory_gb, 3),
        }

        # 打印结果
        print(f"\n{cache_type.upper()} 结果:")
        print(f"  吞吐量:     {results[cache_type]['overall_throughput_tps']} tokens/sec")
        print(f"  平均延迟:   {results[cache_type]['avg_latency_ms']} ms")
        print(f"  P95 延迟:   {results[cache_type]['p95_latency_ms']} ms")
        print(f"  平均 TTFT:  {results[cache_type]['avg_ttft_ms']} ms")
        print(f"  Cache 显存: {results[cache_type]['cache_memory_gb']} GB")
        print(f"  峰值显存:   {results[cache_type]['peak_memory_gb']} GB")

    return results


def print_comparison(results: Dict[str, Dict]) -> None:
    """打印对比结果"""
    if len(results) < 2:
        return

    cache_types = list(results.keys())
    if len(cache_types) != 2:
        return

    a, b = cache_types
    a_results = results[a]
    b_results = results[b]

    print(f"\n{'='*60}")
    print(f" 对比: {a.upper()} vs {b.upper()}")
    print(f"{'='*60}")

    metrics = [
        ("overall_throughput_tps", "吞吐量 (tokens/sec)", "higher"),
        ("avg_latency_ms", "平均延迟 (ms)", "lower"),
        ("p95_latency_ms", "P95 延迟 (ms)", "lower"),
        ("avg_ttft_ms", "平均 TTFT (ms)", "lower"),
        ("cache_memory_gb", "Cache 显存 (GB)", "lower"),
        ("peak_memory_gb", "峰值显存 (GB)", "lower"),
    ]

    for key, label, direction in metrics:
        a_val = a_results[key]
        b_val = b_results[key]

        if direction == "higher":
            improvement = ((b_val - a_val) / a_val * 100) if a_val > 0 else 0
            better = b if improvement > 0 else a
        else:
            improvement = ((a_val - b_val) / a_val * 100) if a_val > 0 else 0
            better = b if improvement > 0 else a

        diff_str = f"{improvement:+.1f}%" if improvement != 0 else "持平"
        print(f"  {label:<25s}: {a_val:.2f} vs {b_val:.2f} ({diff_str})")


def main():
    parser = argparse.ArgumentParser(description="Cache 对比评测")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--prompt_file", type=str, default="prompts.jsonl", help="测试 prompt 文件")
    parser.add_argument("--batch_size", type=int, default=1, help="批量大小")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="最大生成 token 数")
    parser.add_argument("--output", type=str, default=None, help="输出 JSON 文件")
    args = parser.parse_args()

    # 加载 prompts
    prompts = load_prompts(args.prompt_file)
    print(f"[INFO] 加载 {len(prompts)} 条测试 prompt")

    # 运行对比
    results = run_cache_comparison(
        model_path=args.model_path,
        prompts=prompts,
        cache_types=["continuous", "paged"],
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    # 打印对比
    print_comparison(results)

    # 保存结果
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] 结果已保存: {args.output}")


if __name__ == "__main__":
    main()
