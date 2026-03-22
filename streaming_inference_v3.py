#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
streaming_inference_v3.py — v3 流式输出演示
"""

import time
import torch
from optimized_inference_v3 import load_model, get_eos_ids

DEVICE = "cuda:0"


@torch.inference_mode()
def streaming_generate(model, tokenizer, prompt: str, max_new_tokens: int = 256):
    """逐 token 流式生成 — 支持 KV-Cache"""
    device = torch.device(DEVICE)
    eos_ids = get_eos_ids(tokenizer)

    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attn_mask = enc.get("attention_mask", torch.ones_like(input_ids))

    past_kv = None
    cur_ids = input_ids
    cur_mask = attn_mask
    gen_count = 0

    torch.cuda.synchronize(device)
    t_start = time.perf_counter()
    ttft = None

    for step in range(max_new_tokens):
        out = model(
            input_ids=cur_ids,
            attention_mask=cur_mask,
            past_key_values=past_kv,
            use_cache=True,
            return_dict=True,
        )

        logits = out.logits[:, -1, :]
        past_kv = out.past_key_values
        next_tok = logits.argmax(dim=-1).item()

        if step == 0:
            torch.cuda.synchronize(device)
            ttft = (time.perf_counter() - t_start) * 1000.0

        if next_tok in eos_ids:
            break

        token_text = tokenizer.decode([next_tok], skip_special_tokens=True)
        gen_count += 1
        elapsed = (time.perf_counter() - t_start) * 1000.0
        tps = gen_count / (elapsed / 1000.0) if elapsed > 0 else 0

        yield token_text, {
            "generated_tokens": gen_count,
            "elapsed_ms": round(elapsed, 2),
            "ttft_ms": round(ttft, 2),
            "throughput_tps": round(tps, 2),
        }

        cur_ids = torch.tensor([[next_tok]], device=device)
        cur_mask = torch.cat(
            [cur_mask, torch.ones(1, 1, device=device, dtype=cur_mask.dtype)],
            dim=1,
        )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="v3 流式推理演示")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="请用三句话解释KV Cache的作用。")
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    tokenizer, model, kv_cache_pool = load_model(args.model_path, quantize=args.quantize)

    print(f"\n[推理] 开始流式生成...")
    print(f"Prompt: {args.prompt}\n")
    print("=" * 60)
    print("输出: ", end="", flush=True)

    final_metrics = None
    for token_text, metrics in streaming_generate(model, tokenizer, args.prompt):
        print(token_text, end="", flush=True)
        final_metrics = metrics
        if metrics["generated_tokens"] % 50 == 0:
            print(
                f"\n  [进度] Token数: {metrics['generated_tokens']}, "
                f"吞吐: {metrics['throughput_tps']} tok/s",
                flush=True,
            )

    print("\n" + "=" * 60)
    if final_metrics:
        print(f"[完成] 生成 {final_metrics['generated_tokens']} tokens")
        print(f"  TTFT: {final_metrics['ttft_ms']:.1f} ms")
        print(f"  吞吐: {final_metrics['throughput_tps']:.1f} tok/s")
        print(f"  总耗时: {final_metrics['elapsed_ms']:.1f} ms")
    print(f"  峰值显存: {torch.cuda.max_memory_allocated(DEVICE)/1e9:.2f} GB")


if __name__ == "__main__":
    main()
