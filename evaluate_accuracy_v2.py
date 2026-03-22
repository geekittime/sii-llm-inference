#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_accuracy_v2.py - 改进版
精度评测脚本（兼容完整PagedAttention版本）
"""

import argparse
import json
import time
from pathlib import Path

from optimized_inference_v2 import load_model, infer_all, BATCH_SIZE

DEFAULT_EVAL_FILE = Path(__file__).parent / "ceval_subset.jsonl"


def load_eval_data(path: str) -> list:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    import json as j
                    data.append(j.loads(line))
                except:
                    pass
    print(f"[INFO] 加载 {len(data)} 道题")
    return data


def build_prompt(item: dict) -> str:
    return (
        f"以下是一道单选题，请直接回答选项字母（A/B/C/D），不要有任何解释。\n\n"
        f"题目：{item['question']}\n"
        f"A. {item['A']}\n"
        f"B. {item['B']}\n"
        f"C. {item['C']}\n"
        f"D. {item['D']}\n"
        f"答案："
    )


def extract_answer(text: str) -> str:
    for ch in text.strip():
        if ch.upper() in "ABCD":
            return ch.upper()
    return "X"


def run_accuracy_eval(tokenizer, model, eval_data: list, batch_size: int) -> dict:
    print(f"\n[Accuracy] 评测 {len(eval_data)} 题，batch_size={batch_size}")
    print("-" * 60)

    prompts = [build_prompt(item) for item in eval_data]
    golds = [item["answer"].upper() for item in eval_data]

    t0 = time.perf_counter()

    results = infer_all(
        tokenizer, model,
        prompts,
        batch_size=batch_size,
        max_new_tokens=32,
        show_progress=True,
    )

    t1 = time.perf_counter()

    correct = 0
    wrong_cases = []
    for i, (res, gold) in enumerate(zip(results, golds)):
        pred = extract_answer(res["output"])
        if pred == gold:
            correct += 1
        else:
            wrong_cases.append({
                "id": eval_data[i].get("id", i),
                "question": eval_data[i]["question"][:60] + "...",
                "pred": pred,
                "gold": gold,
            })

    accuracy = correct / len(eval_data)
    return {
        "total": len(eval_data),
        "correct": correct,
        "wrong": len(eval_data) - correct,
        "accuracy": round(accuracy, 4),
        "accuracy_pct": round(accuracy * 100, 2),
        "eval_time_sec": round(t1 - t0, 2),
        "wrong_cases": wrong_cases[:10],
    }


def print_result(r: dict, baseline: float = None):
    print(f"\n{'='*60}")
    print(f" 精度评测结果")
    print(f"{'='*60}")
    print(f"  总题数: {r['total']}")
    print(f"  正确: {r['correct']}")
    print(f"  错误: {r['wrong']}")
    print(f"  准确率: {r['accuracy_pct']:.2f}%")
    print(f"  耗时: {r['eval_time_sec']:.1f} sec")
    if baseline is not None:
        drop = baseline - r["accuracy"]
        ok = "✓ 达标" if drop <= 0.05 else "✗ 超标"
        print(f"  基线: {baseline*100:.2f}%")
        print(f"  精度下降: {drop*100:.2f}% {ok}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="精度评测 v2")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--eval_file", type=str, default=str(DEFAULT_EVAL_FILE))
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--baseline_acc", type=float, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    tokenizer, model, kv_cache = load_model(args.model_path, quantize=args.quantize)
    eval_data = load_eval_data(args.eval_file)
    result = run_accuracy_eval(tokenizer, model, eval_data, args.batch_size)
    print_result(result, args.baseline_acc)

    if args.output:
        out = {k: v for k, v in result.items() if k != "wrong_cases"}
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 结果保存到: {args.output}")


if __name__ == "__main__":
    main()
