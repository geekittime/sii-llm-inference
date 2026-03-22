#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# /inspire/hdd/project/mianxiangdayuyanmoxing/public/Qwen2.5-14B-Instruct
"""
baseline_inference.py 
=====================
基准脚本

快速运行：
  python baseline_inference.py --model_path /path/to/model
"""

import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


DEVICE         = "cuda:0"      
DTYPE          = torch.float16  
MAX_NEW_TOKENS = 256            
SEED           = 42           

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def load_model(model_path: str):
    print(f"[INFO] 加载模型: {model_path}")
    print(f"[INFO] 设备: {DEVICE} | 数据类型: {DTYPE}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        # dtype=DTYPE,
        device_map=DEVICE,
        trust_remote_code=True,
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    mem_gb   = torch.cuda.memory_allocated() / 1e9
    print(f"[INFO] 加载完成 | 参数量: {n_params:.2f}B | 显存占用: {mem_gb:.2f} GB")
    return tokenizer, model

def infer_single(tokenizer, model, prompt: str) -> dict:
    inputs    = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]

    torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,                      
            use_cache=True,                      
            pad_token_id=tokenizer.pad_token_id,
        )

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    output_len   = output_ids.shape[1] - input_len
    total_ms     = (t_end - t_start) * 1000
    ttft_approx  = total_ms               # 因为此处非流式输出，所有token生成完毕一起返回，所以ttft等于总延迟。流式输出时该计时逻辑需要更改
    throughput   = output_len / total_ms *1000  # tokens/sec
    output_text  = tokenizer.decode(
        output_ids[0][input_len:], skip_special_tokens=True
    )

    return {
        "prompt":           prompt,
        "output":           output_text,
        "input_tokens":     input_len,
        "output_tokens":    output_len,
        "total_latency_ms": round(total_ms, 2),
        "ttft_ms":   round(ttft_approx, 2),
        "throughput_tps":   round(throughput, 2),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM 推理基准 —— 单条 prompt 快速验证"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="模型本地路径，例如 /data/models/Qwen2.5-7B-Instruct"
    )
    parser.add_argument(
        "--prompt", type=str,
        default="请用三句话解释大语言模型推理中KV Cache的作用。",
        help="测试用 prompt"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tokenizer, model = load_model(args.model_path)

    print(f"\n[推理] prompt: {args.prompt}")
    result = infer_single(tokenizer, model, args.prompt)

    print("\n" + "=" * 64)
    print(" 推理结果")
    print("=" * 64)
    print(f"  输入   : {result['prompt']}")
    print(f"  输出   : {result['output']}")
    print("-" * 64)
    print(f"  输入 tokens   : {result['input_tokens']}")
    print(f"  输出 tokens   : {result['output_tokens']}")
    print(f"  总延迟         : {result['total_latency_ms']} ms")
    print(f"  TTFT (近似)   : {result['ttft_ms']} ms/token")
    print(f"  吞吐率         : {result['throughput_tps']} tokens/sec")
    print(f"  峰值显存       : {torch.cuda.max_memory_allocated() / 1e9:.3f} GB")
    print("=" * 64)
