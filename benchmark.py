#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark.py — 批量推理基准测试
"""

import argparse
import json
import time
import torch
import numpy as np
from pathlib import Path

from optimized_inference import (
    load_model, infer_all, DEVICE, MAX_NEW_TOKENS, BATCH_SIZE,
)

DEFAULT_PROMPT_FILE = Path(__file__).parent / "prompts.jsonl"


def load_prompts(path: str) -> list:
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompts.append(obj if isinstance(obj, dict) else {"prompt": obj})
    print(f"[INFO] 加载 {len(prompts)} 条 prompt")
    return prompts


def run_benchmark(tokenizer, model, prompts: list, batch_size: int) -> dict:
    prompt_texts = [
        p["prompt"] if isinstance(p, dict) else str(p)
        for p in prompts
    ]

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

    latencies  = [r["total_latency_ms"] for r in results]
    ttfts      = [r["ttft_ms"]          for r in results]
    out_tokens = [r["output_tokens"]    for r in results]
    total_out  = sum(out_tokens)

    stats = {
        "total_prompts":          len(prompts),
        "batch_size":             batch_size,
        "total_output_tokens":    total_out,
        "wall_time_sec":          round(wall_time, 2),
        "max_new_tokens_cfg":     MAX_NEW_TOKENS,
        "overall_throughput_tps": round(total_out / wall_time, 2) if wall_time > 0 else 0,
        "avg_latency_ms":         round(float(np.mean(latencies)), 2),
        "p50_latency_ms":         round(float(np.percentile(latencies, 50)), 2),
        "p95_latency_ms":         round(float(np.percentile(latencies, 95)), 2),
        "p99_latency_ms":         round(float(np.percentile(latencies, 99)), 2),
        "avg_ttft_ms":            round(float(np.mean(ttfts)), 2),
        "p95_ttft_ms":            round(float(np.percentile(ttfts, 95)), 2),
        "peak_gpu_mem_gb":        round(torch.cuda.max_memory_allocated(DEVICE) / 1e9, 3),
    }
    return stats


def print_stats(s: dict):
    print(f"\n{'='*64}")
    print(f" Benchmark 结果 (batch_size={s['batch_size']})")
    print(f"{'='*64}")
    for label, key in [
        ("测试 prompt 数",           "total_prompts"),
        ("Batch 大小",               "batch_size"),
        ("总输出 tokens",            "total_output_tokens"),
        ("总耗时 (sec)",             "wall_time_sec"),
        ("max_new_tokens",           "max_new_tokens_cfg"),
        ("▶ 整体吞吐 (tokens/sec)", "overall_throughput_tps"),
        ("平均延迟 (ms)",            "avg_latency_ms"),
        ("P50 延迟 (ms)",            "p50_latency_ms"),
        ("P95 延迟 (ms)",            "p95_latency_ms"),
        ("P99 延迟 (ms)",            "p99_latency_ms"),
        ("平均 TTFT (ms)",           "avg_ttft_ms"),
        ("P95  TTFT (ms)",           "p95_ttft_ms"),
        ("峰值显存 (GB)",            "peak_gpu_mem_gb"),
    ]:
        print(f"  {label:<30s}: {s[key]}")
    print(f"{'='*64}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  type=str, required=True)
    parser.add_argument("--prompt_file", type=str, default=str(DEFAULT_PROMPT_FILE))
    parser.add_argument("--batch_size",  type=int, default=BATCH_SIZE)
    parser.add_argument("--output",      type=str, default=None)
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_path)
    prompts = load_prompts(args.prompt_file)
    stats   = run_benchmark(tokenizer, model, prompts, args.batch_size)
    print_stats(stats)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] 结果已保存: {args.output}")


if __name__ == "__main__":
    main()