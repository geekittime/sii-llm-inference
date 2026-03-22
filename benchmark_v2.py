#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_v2.py - 改进版
性能基准测试脚本（兼容完整PagedAttention版本）
"""

import argparse
import json
import time
import torch
import numpy as np
from pathlib import Path

from optimized_inference_v2 import (
    load_model, infer_all, DEVICE, MAX_NEW_TOKENS, BATCH_SIZE,
)

DEFAULT_PROMPT_FILE = Path(__file__).parent / "prompts.jsonl"


def load_prompts(path: str) -> list:
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    import json as j
                    obj = j.loads(line)
                    text = obj.get("prompt") if isinstance(obj, dict) else obj
                    if text:
                        prompts.append(text)
                except:
                    prompts.append(line)
    print(f"[INFO] 加载 {len(prompts)} 条 prompt")
    return prompts


def run_benchmark(tokenizer, model, kv_cache, prompts: list, batch_size: int):
    """运行基准测试"""
    prompt_texts = prompts

    torch.cuda.reset_peak_memory_stats(DEVICE)
    torch.cuda.synchronize(DEVICE)
    wall_start = time.perf_counter()

    results = infer_all(
        tokenizer, model,
        prompt_texts,
        batch_size=batch_size,
        show_progress=True,
    )

    torch.cuda.synchronize(DEVICE)
    wall_time = time.perf_counter() - wall_start

    # 统计
    latencies = [r["total_latency_ms"] for r in results]
    ttfts = [r["ttft_ms"] for r in results]
    out_tokens = [r["output_tokens"] for r in results]
    total_out = sum(out_tokens)

    mem = kv_cache.get_memory_usage()
    peak_mem = torch.cuda.max_memory_allocated(DEVICE) / 1e9

    stats = {
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
        "peak_gpu_mem_gb": round(peak_mem, 3),
        "kv_cache_blocks_used": mem["used_blocks"],
        "kv_cache_blocks_total": mem["total_blocks"],
    }
    return stats


def print_stats(s: dict):
    print(f"\n{'='*64}")
    print(f" 基准测试结果 (batch_size={s['batch_size']})")
    print(f"{'='*64}")
    print(f"  总prompt数: {s['total_prompts']}")
    print(f"  总输出tokens: {s['total_output_tokens']}")
    print(f"  总耗时: {s['wall_time_sec']} sec")
    print(f"  整体吞吐: {s['overall_throughput_tps']} tokens/sec ⭐")
    print(f"  平均延迟: {s['avg_latency_ms']} ms")
    print(f"  P95延迟: {s['p95_latency_ms']} ms")
    print(f"  平均TTFT: {s['avg_ttft_ms']} ms")
    print(f"  峰值显存: {s['peak_gpu_mem_gb']} GB")
    print(f"  KV缓存块: {s['kv_cache_blocks_used']}/{s['kv_cache_blocks_total']}")
    print(f"{'='*64}")


def main():
    parser = argparse.ArgumentParser(description="性能基准测试 v2")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, default=str(DEFAULT_PROMPT_FILE))
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    tokenizer, model, kv_cache = load_model(args.model_path, quantize=args.quantize)
    prompts = load_prompts(args.prompt_file)
    stats = run_benchmark(tokenizer, model, kv_cache, prompts, args.batch_size)
    print_stats(stats)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 结果保存到: {args.output}")


if __name__ == "__main__":
    main()
