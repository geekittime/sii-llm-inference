#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark.py
============
吞吐量 & 延迟基准测试脚本

使用方式：
  # 运行基准测试
  python benchmark.py --model_path /path/to/model --output results_baseline.json
"""

import argparse
import json
import time
import torch
import numpy as np
from pathlib import Path
from baseline_inference import load_model, infer_single, DEVICE, MAX_NEW_TOKENS

DEFAULT_PROMPT_FILE = Path(__file__).parent / "prompts.jsonl"


def load_prompts(prompt_file: str) -> list:
    """读取 prompts.jsonl，每行格式：{"id": 1, "prompt": "..."}"""
    prompts = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    print(f"[INFO] 已加载 {len(prompts)} 条 prompt（来自 {prompt_file}）")
    return prompts


def run_benchmark(tokenizer, model, prompts: list) -> dict:
    print(f"\n[Benchmark] 共 {len(prompts)} 条 prompt，开始推理...")
    print("-" * 68)

    latencies   = []
    ttfts       = []
    total_out   = 0

    torch.cuda.reset_peak_memory_stats(DEVICE)
    t_wall_start = time.perf_counter()

    for i, item in enumerate(prompts):
        prompt = item["prompt"] if isinstance(item, dict) else item
        res    = infer_single(tokenizer, model, prompt)

        latencies.append(res["total_latency_ms"])
        ttfts.append(res["ttft_ms"])
        total_out += res["output_tokens"]

        print(
            f"  [{i+1:3d}/{len(prompts)}]  "
            f"latency={res['total_latency_ms']:8.1f} ms  "
            f"ttft={res['ttft_ms']:8.1f} ms  "
            f"throughput={res['throughput_tps']:6.1f} token/s "
            f"output={res['output_tokens']:4d} tokens"
        )

    t_wall_end = time.perf_counter()
    wall_time  = t_wall_end - t_wall_start

    stats = {
        "total_prompts":          len(prompts),
        "total_output_tokens":    total_out,
        "wall_time_sec":          round(wall_time, 2),
        "max_new_tokens_cfg":     MAX_NEW_TOKENS,

        "overall_throughput_tps": round(total_out / wall_time, 2),

        "avg_latency_ms":         round(float(np.mean(latencies)), 2),
        "p50_latency_ms":         round(float(np.percentile(latencies, 50)), 2),
        "p95_latency_ms":         round(float(np.percentile(latencies, 95)), 2),
        "p99_latency_ms":         round(float(np.percentile(latencies, 99)), 2),

        "avg_ttft_ms":            round(float(np.mean(ttfts)), 2),
        "p95_ttft_ms":            round(float(np.percentile(ttfts, 95)), 2),

        "peak_gpu_mem_gb":        round(
            torch.cuda.max_memory_allocated(DEVICE) / 1e9, 3
        ),
    }
    return stats


def print_stats(stats: dict):
    """格式化打印 benchmark 结果"""
    labels = {
        "total_prompts":          "测试 prompt 数",
        "total_output_tokens":    "总输出 tokens",
        "wall_time_sec":          "总耗时 (sec)",
        "max_new_tokens_cfg":     "max_new_tokens 配置",
        "overall_throughput_tps": "整体吞吐率 (tokens/sec)  ",
        "avg_latency_ms":         "平均延迟 (ms)            ",
        "p50_latency_ms":         "P50 延迟 (ms)            ",
        "p95_latency_ms":         "P95 延迟 (ms)            ",
        "p99_latency_ms":         "P99 延迟 (ms)            ",
        "avg_ttft_ms":            "平均 TTFT (ms)            ",
        "p95_ttft_ms":            "P95 TTFT (ms)             ",
        "peak_gpu_mem_gb":        "峰值显存 (GB)             ",
    }
    print("\n" + "=" * 68)
    print(" Benchmark 结果汇总（baseline）")
    print("=" * 68)
    for key, label in labels.items():
        val = stats.get(key, "N/A")
        print(f"  {label:<40s}: {val}")
    print("=" * 68)
    print("  注：精度指标请运行 evaluate_accuracy.py")
    print("=" * 68)


def parse_args():
    parser = argparse.ArgumentParser(description="LLM 推理吞吐 & 延迟基准测试")
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="模型本地路径，例如 /data/models/Qwen2.5-7B-Instruct"
    )
    parser.add_argument(
        "--prompt_file", type=str, default=str(DEFAULT_PROMPT_FILE),
        help="prompt 文件路径（默认：prompts.jsonl）"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="结果保存路径（JSON），例如 results_baseline.json"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 加载模型
    tokenizer, model = load_model(args.model_path)

    # 加载 prompt
    prompts = load_prompts(args.prompt_file)

    # 运行测试
    stats = run_benchmark(tokenizer, model, prompts)

    # 打印结果
    print_stats(stats)

    # 保存结果
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] 结果已保存至: {args.output}")
