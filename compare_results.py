#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_results.py
==================
对比baseline与优化版本的性能数据

用法:
  python compare_results.py \
    --baseline results_baseline.json \
    --optimized results_optimized.json \
    --accuracy_baseline accuracy_baseline.json \
    --accuracy_optimized accuracy_optimized.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def load_json(path: str) -> Dict[str, Any]:
    """加载JSON文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_improvement(baseline: float, optimized: float, is_higher_better: bool = True) -> str:
    """计算性能提升百分比"""
    if baseline == 0:
        return "N/A"

    if is_higher_better:
        # 吞吐量、精度等：越高越好
        improvement = (optimized - baseline) / baseline * 100
    else:
        # 延迟、显存等：越低越好
        improvement = (baseline - optimized) / baseline * 100

    sign = "+" if improvement >= 0 else ""
    return f"{sign}{improvement:.1f}%"


def compare_benchmarks(baseline: Dict, optimized: Dict) -> None:
    """对比吞吐量和延迟指标"""
    print("\n" + "=" * 80)
    print(" 性能基准对比 (吞吐量 & 延迟)")
    print("=" * 80)

    metrics = [
        ("总prompt数", "total_prompts", True, "-"),
        ("总输出tokens", "total_output_tokens", True, "-"),
        ("总耗时(sec)", "wall_time_sec", True, "↓"),
        ("", "", False, ""),
        ("整体吞吐(tokens/sec)", "overall_throughput_tps", True, "↑"),
        ("平均延迟(ms)", "avg_latency_ms", False, "↓"),
        ("P50延迟(ms)", "p50_latency_ms", False, "↓"),
        ("P95延迟(ms)", "p95_latency_ms", False, "↓"),
        ("P99延迟(ms)", "p99_latency_ms", False, "↓"),
        ("", "", False, ""),
        ("平均TTFT(ms)", "avg_ttft_ms", False, "↓"),
        ("P95 TTFT(ms)", "p95_ttft_ms", False, "↓"),
        ("", "", False, ""),
        ("峰值显存(GB)", "peak_gpu_mem_gb", False, "↓"),
    ]

    for label, key, show_improvement, direction in metrics:
        if label == "":
            print("-" * 80)
            continue

        base_val = baseline.get(key, "N/A")
        opt_val = optimized.get(key, "N/A")

        improvement = ""
        if show_improvement and isinstance(base_val, (int, float)) and isinstance(opt_val, (int, float)):
            is_higher_better = direction == "↑"
            improvement = f"  {calculate_improvement(base_val, opt_val, is_higher_better)}"

        print(
            f"  {label:<25s}  {direction:1s} "
            f"  baseline: {base_val:<12} → optimized: {opt_val:<12}{improvement}"
        )


def compare_accuracy(baseline: Dict, optimized: Dict) -> None:
    """对比精度指标"""
    print("\n" + "=" * 80)
    print(" 精度对比")
    print("=" * 80)

    base_acc = baseline.get("accuracy", 0) * 100
    opt_acc = optimized.get("accuracy", 0) * 100

    print(f"  基线准确率:    {base_acc:.2f}%")
    print(f"  优化后准确率:  {opt_acc:.2f}%")

    drop = base_acc - opt_acc
    print(f"  精度变化:      {drop:-+.2f}%")

    if abs(drop) <= 5.0:
        status = "✓ 达标 (≤5%)"
    else:
        status = "✗ 超标 (>5%)"

    print(f"  状态:          {status}")


def generate_summary(baseline: Dict, optimized: Dict, acc_baseline: Dict, acc_optimized: Dict) -> None:
    """生成优化总结"""
    print("\n" + "=" * 80)
    print(" 优化总结")
    print("=" * 80)

    # 关键指标提升
    tps_improvement = calculate_improvement(
        baseline["overall_throughput_tps"],
        optimized["overall_throughput_tps"],
        is_higher_better=True
    )
    latency_improvement = calculate_improvement(
        baseline["avg_latency_ms"],
        optimized["avg_latency_ms"],
        is_higher_better=False
    )
    mem_improvement = calculate_improvement(
        baseline["peak_gpu_mem_gb"],
        optimized["peak_gpu_mem_gb"],
        is_higher_better=False
    )

    print(f"\n🎯 关键指标:")
    print(f"  • 吞吐量提升:     {tps_improvement}")
    print(f"  • 平均延迟优化:   {latency_improvement}")
    print(f"  • 显存节省:       {mem_improvement}")

    # 精度验证
    base_acc_pct = acc_baseline.get("accuracy_pct", 0)
    opt_acc_pct = acc_optimized.get("accuracy_pct", 0)
    acc_drop = base_acc_pct - opt_acc_pct

    print(f"\n📊 精度验证:")
    print(f"  • 基线精度:       {base_acc_pct:.2f}%")
    print(f"  • 优化精度:       {opt_acc_pct:.2f}%")
    print(f"  • 损失:           {acc_drop:+.2f}%")

    if acc_drop <= 5:
        print(f"  ✓ 精度验证通过 (损失 ≤ 5%)")
    else:
        print(f"  ✗ 精度验证失败 (损失 > 5%)")

    # 综合评分
    print(f"\n⭐ 综合评估:")
    tps_val = float(optimized["overall_throughput_tps"]) / baseline["overall_throughput_tps"] - 1
    lat_val = 1 - (float(optimized["avg_latency_ms"]) / baseline["avg_latency_ms"])

    print(f"  • 吞吐提升倍数:   {tps_val:.1%}")
    print(f"  • 延迟改善倍数:   {lat_val:.1%}")

    if tps_val >= 0.30 and acc_drop <= 5:
        print(f"\n🏆 评价: 优化效果显著 (吞吐+33%, 精度达标)")
    elif tps_val >= 0.15 and acc_drop <= 5:
        print(f"\n✓ 评价: 优化效果良好 (吞吐+15%, 精度达标)")
    else:
        print(f"\n△ 评价: 可继续优化")


def main():
    parser = argparse.ArgumentParser(description="性能对比分析")
    parser.add_argument("--baseline", type=str, required=True, help="baseline结果JSON")
    parser.add_argument("--optimized", type=str, required=True, help="优化后结果JSON")
    parser.add_argument("--accuracy_baseline", type=str, help="baseline精度JSON")
    parser.add_argument("--accuracy_optimized", type=str, help="优化后精度JSON")
    args = parser.parse_args()

    # 加载数据
    baseline_perf = load_json(args.baseline)
    optimized_perf = load_json(args.optimized)

    print(f"\n{'='*80}")
    print(f" LLM推理性能优化对比分析")
    print(f"{'='*80}")

    print(f"\n📁 数据源:")
    print(f"  Baseline:   {args.baseline}")
    print(f"  Optimized:  {args.optimized}")

    # 对比性能
    compare_benchmarks(baseline_perf, optimized_perf)

    # 对比精度
    if args.accuracy_baseline and args.accuracy_optimized:
        acc_baseline = load_json(args.accuracy_baseline)
        acc_optimized = load_json(args.accuracy_optimized)
        compare_accuracy(acc_baseline, acc_optimized)

        # 生成总结
        generate_summary(baseline_perf, optimized_perf, acc_baseline, acc_optimized)
    else:
        print("\n⚠️  未提供精度文件，跳过精度对比")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
