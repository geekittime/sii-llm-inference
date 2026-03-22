#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_accuracy_llm_inference.py
==================================
使用 llm_inference 模块的精度评测脚本

支持两种模式：
  1. continuous: 连续 KV-Cache，使用批量 decode
  2. paged: 分页 KV-Cache (PagedAttention)

使用方式：
  # continuous 模式
  python evaluate_accuracy_llm_inference.py \
    --model_path /path/to/model \
    --cache_type continuous \
    --eval_file ceval_subset.jsonl \
    --batch_size 8

  # paged 模式（自动占满 90% 显存）
  python evaluate_accuracy_llm_inference.py \
    --model_path /path/to/model \
    --cache_type paged \
    --eval_file ceval_subset.jsonl \
    --batch_size 64

  # paged 模式（调整显存利用率）
  python evaluate_accuracy_llm_inference.py \
    --model_path /path/to/model \
    --cache_type paged \
    --gpu_memory_utilization 0.85 \
    --eval_file ceval_subset.jsonl

  # paged 模式（手动指定 block 数，跳过自动计算）
  python evaluate_accuracy_llm_inference.py \
    --model_path /path/to/model \
    --cache_type paged \
    --num_blocks 2048 \
    --eval_file ceval_subset.jsonl
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict

from llm_inference.inference import InferenceEngine

DEFAULT_EVAL_FILE = Path(__file__).parent / "ceval_subset.jsonl"
ACCURACY_DROP_LIMIT = 0.05   # 精度损失上限（绝对值）


def load_eval_data(eval_file: str) -> list:
    """加载 C-Eval 格式数据集"""
    data = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"[INFO] 已加载 {len(data)} 道评测题（来自 {eval_file}）")
    return data


def build_prompt(item: dict) -> str:
    """构建 C-Eval 格式的 prompt"""
    return (
        f"以下是一道单选题，请直接回答选项字母（A/B/C/D），不要有任何解释。\n\n"
        f"题目：{item['question']}\n"
        f"A. {item['A']}\n"
        f"B. {item['B']}\n"
        f"C. {item['C']}\n"
        f"D. {item['D']}\n"
        f"答案："
    )


def extract_answer(output: str) -> str:
    """从输出中提取答案选项"""
    for ch in output.strip():
        if ch.upper() in ("A", "B", "C", "D"):
            return ch.upper()
    return "X"   # 未能解析时标记为错误


def run_accuracy_eval_batch(
    engine: InferenceEngine,
    eval_data: list,
    batch_size: int = 8,
) -> dict:
    """
    使用批量推理进行精度评测

    Args:
        engine: InferenceEngine 实例
        eval_data: C-Eval 格式的评测数据
        batch_size: 批量大小

    Returns:
        评测结果字典
    """
    print(f"\n[Accuracy] 开始精度评测，共 {len(eval_data)} 道题...")
    print(f"[Accuracy] cache_type={engine.cache_type}, batch_size={batch_size}")
    print("-" * 60)

    # 构建所有 prompts
    prompts = [build_prompt(item) for item in eval_data]

    # 批量推理
    t_start = time.perf_counter()
    inference_results = engine.infer_batch(
        prompts=prompts,
        max_new_tokens=8,  # C-Eval 只需要回答选项字母，不需要长输出
        show_progress=True,
    )
    t_end = time.perf_counter()

    # 统计结果
    correct = 0
    wrong_cases = []

    for i, (item, res) in enumerate(zip(eval_data, inference_results)):
        pred = extract_answer(res.output)
        gold = item["answer"].upper()
        is_correct = (pred == gold)

        if is_correct:
            correct += 1
        else:
            wrong_cases.append({
                "id": item.get("id", i),
                "question": item["question"][:60] + "...",
                "pred": pred,
                "gold": gold,
            })

        if (i + 1) % 20 == 0 or (i + 1) == len(eval_data):
            acc_so_far = correct / (i + 1)
            print(
                f"  [{i+1:4d}/{len(eval_data)}]  "
                f"当前准确率: {acc_so_far*100:.1f}%  "
                f"正确: {correct}  错误: {i+1-correct}"
            )

    accuracy = correct / len(eval_data)

    result = {
        "cache_type": engine.cache_type,
        "batch_size": batch_size,
        "total": len(eval_data),
        "correct": correct,
        "wrong": len(eval_data) - correct,
        "accuracy": round(accuracy, 4),
        "accuracy_pct": round(accuracy * 100, 2),
        "eval_time_sec": round(t_end - t_start, 2),
        "wrong_cases": wrong_cases[:10],   # 最多展示前10个错误案例
    }
    return result


def run_accuracy_eval_single(
    engine: InferenceEngine,
    eval_data: list,
) -> dict:
    """
    使用单条推理进行精度评测（适用于 PagedCache 模式）

    Args:
        engine: InferenceEngine 实例
        eval_data: C-Eval 格式的评测数据

    Returns:
        评测结果字典
    """
    print(f"\n[Accuracy] 开始精度评测，共 {len(eval_data)} 道题...")
    print(f"[Accuracy] cache_type={engine.cache_type}, 单条推理")
    print("-" * 60)

    correct = 0
    wrong_cases = []
    t_start = time.perf_counter()

    for i, item in enumerate(eval_data):
        prompt = build_prompt(item)
        res = engine.infer_single(prompt=prompt, max_new_tokens=8)

        pred = extract_answer(res.output)
        gold = item["answer"].upper()
        is_correct = (pred == gold)

        if is_correct:
            correct += 1
        else:
            wrong_cases.append({
                "id": item.get("id", i),
                "question": item["question"][:60] + "...",
                "pred": pred,
                "gold": gold,
            })

        if (i + 1) % 20 == 0 or (i + 1) == len(eval_data):
            acc_so_far = correct / (i + 1)
            print(
                f"  [{i+1:4d}/{len(eval_data)}]  "
                f"当前准确率: {acc_so_far*100:.1f}%  "
                f"正确: {correct}  错误: {i+1-correct}"
            )

    t_end = time.perf_counter()
    accuracy = correct / len(eval_data)

    result = {
        "cache_type": engine.cache_type,
        "batch_size": 1,
        "total": len(eval_data),
        "correct": correct,
        "wrong": len(eval_data) - correct,
        "accuracy": round(accuracy, 4),
        "accuracy_pct": round(accuracy * 100, 2),
        "eval_time_sec": round(t_end - t_start, 2),
        "wrong_cases": wrong_cases[:10],
    }
    return result


def print_accuracy_result(result: dict, baseline_acc: float = None):
    """格式化打印精度评测结果"""
    print("\n" + "=" * 60)
    print(f" 精度评测结果（{result.get('cache_type', 'unknown')}）")
    print("=" * 60)
    print(f"  Cache 类型   : {result.get('cache_type', 'unknown')}")
    print(f"  Batch 大小   : {result.get('batch_size', 1)}")
    print(f"  总题数       : {result['total']}")
    print(f"  答对题数     : {result['correct']}")
    print(f"  答错题数     : {result['wrong']}")
    print(f"  准确率       : {result['accuracy_pct']:.2f}%")
    print(f"  评测耗时     : {result['eval_time_sec']} sec")

    if baseline_acc is not None:
        drop = baseline_acc - result["accuracy"]
        status = "达标" if drop <= ACCURACY_DROP_LIMIT else "超标（扣分）"
        print("-" * 60)
        print(f"  基线准确率   : {baseline_acc*100:.2f}%")
        print(f"  精度下降     : {drop*100:.2f}% （上限 {ACCURACY_DROP_LIMIT*100:.0f}%）")
        print(f"  精度约束状态 : {status}")
        if drop > ACCURACY_DROP_LIMIT:
            print("  [警告] 精度损失超过阈值，「优化效果」评分将扣 50%！")

    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(
        description="C-Eval 精度评测（llm_inference 模块）"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="模型本地路径"
    )
    parser.add_argument(
        "--cache_type", type=str, default="continuous",
        choices=["continuous", "paged"],
        help="Cache 类型（默认：continuous）"
    )
    parser.add_argument(
        "--eval_file", type=str, default=str(DEFAULT_EVAL_FILE),
        help="评测数据集路径（默认：ceval_subset.jsonl）"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="批量大小（默认：8，仅 continuous 模式有效）"
    )
    parser.add_argument(
        "--baseline_acc", type=float, default=None,
        help="基线准确率（0~1），用于判断精度是否达标，例如 --baseline_acc 0.72"
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
        help="结果保存路径（JSON），例如 accuracy_continuous.json"
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

    # 加载评测数据
    eval_data = load_eval_data(args.eval_file)

    # 运行评测（两种模式都使用批量推理）
    result = run_accuracy_eval_batch(engine, eval_data, args.batch_size)

    # 打印结果
    print_accuracy_result(result, args.baseline_acc)

    # 保存结果
    if args.output:
        out = {k: v for k, v in result.items() if k != "wrong_cases"}
        out["wrong_cases_count"] = result["wrong"]
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] 结果已保存至: {args.output}")


if __name__ == "__main__":
    main()
