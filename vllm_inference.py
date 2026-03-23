#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vllm_inference.py
=================
基于 vLLM 的基线推理实现。

特性:
  1. 使用 vLLM 原生的 PagedAttention / 动态 batch / KV cache 管理
  2. 暴露与 optimized_inference.py 兼容的接口:
       - load_model
       - infer_all
       - infer_single
       - DEVICE / MAX_NEW_TOKENS / BATCH_SIZE
  3. 可直接作为 benchmark.py / evaluate_accuracy.py 的替代后端
"""

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "vLLM 未安装。请先安装 requirements.txt 中的 vllm>=0.4.0。"
    ) from exc


DEVICE = "cuda:0"
DTYPE = "float16"
MAX_NEW_TOKENS = 256
BATCH_SIZE = 32
GPU_MEMORY_UTILIZATION = 0.90
MAX_MODEL_LEN = 2048
TEMPERATURE = 0.0
TOP_P = 1.0


@dataclass
class VLLMBackend:
    llm: Any
    tokenizer: Any
    model_path: str
    tensor_parallel_size: int = 1


def _get_eos_ids(tokenizer) -> List[int]:
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    for token in ["<|endoftext|>", "<|im_end|>", "<|end|>", "</s>"]:
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", -1):
                ids.add(int(token_id))
        except Exception:
            pass
    return sorted(ids) if ids else [0]


def _extract_text_and_metrics(
    output,
    fallback_total_ms: float,
) -> Tuple[str, int, float, float]:
    text = ""
    out_tokens = 0

    first_output = output.outputs[0] if getattr(output, "outputs", None) else None
    if first_output is not None:
        text = getattr(first_output, "text", "") or ""
        token_ids = getattr(first_output, "token_ids", None)
        if token_ids is not None:
            out_tokens = len(token_ids)

    metrics = getattr(output, "metrics", None)
    ttft_ms = fallback_total_ms
    total_ms = fallback_total_ms

    if metrics is not None:
        # vLLM 版本间字段可能不同，做兼容处理。
        first_token_time = getattr(metrics, "first_token_time", None)
        finished_time = getattr(metrics, "finished_time", None)
        arrival_time = getattr(metrics, "arrival_time", None)

        if first_token_time is not None and arrival_time is not None:
            ttft_ms = max((first_token_time - arrival_time) * 1000.0, 0.0)
        if finished_time is not None and arrival_time is not None:
            total_ms = max((finished_time - arrival_time) * 1000.0, 0.0)

    return text, out_tokens, ttft_ms, total_ms


def load_model(model_path: str):
    print(f"[INFO] 加载 vLLM 模型: {model_path}")
    print(f"[INFO] 设备={DEVICE}  精度={DTYPE}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        dtype=DTYPE,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        tensor_parallel_size=1,
        enforce_eager=False,
        enable_prefix_caching=True,
    )

    backend = VLLMBackend(
        llm=llm,
        tokenizer=tokenizer,
        model_path=model_path,
    )

    print("[INFO] 预热推理...")
    warm_params = SamplingParams(
        n=1,
        max_tokens=8,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        skip_special_tokens=True,
    )
    _ = llm.generate(["hello world"], warm_params, use_tqdm=False)
    torch.cuda.synchronize(DEVICE)

    peak_vram = torch.cuda.memory_allocated(DEVICE) / 1e9
    print(f"[INFO] 就绪 | VRAM {peak_vram:.2f} GB")
    return tokenizer, backend


def infer_all(
    tokenizer,
    model: VLLMBackend,
    prompts: list,
    batch_size: int = BATCH_SIZE,
    max_new_tokens: int = MAX_NEW_TOKENS,
    show_progress: bool = True,
    use_paged: bool = True,
    use_dynamic_batch: bool = True,
):
    del tokenizer, use_paged, use_dynamic_batch  # vLLM 内部始终使用 paged attention + 动态调度。

    n = len(prompts)
    if n == 0:
        return []

    enc_lens = [len(model.tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
    sorted_idx = sorted(range(n), key=lambda i: enc_lens[i])

    all_results = [None] * n
    num_batches = math.ceil(n / batch_size)
    eos_ids = _get_eos_ids(model.tokenizer)
    sampling_params = SamplingParams(
        n=1,
        max_tokens=max_new_tokens,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        stop_token_ids=eos_ids,
        skip_special_tokens=True,
    )

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n)
        batch_indices = sorted_idx[start:end]
        batch_prompts = [prompts[i] for i in batch_indices]

        torch.cuda.synchronize(DEVICE)
        t0 = time.perf_counter()
        outputs = model.llm.generate(batch_prompts, sampling_params, use_tqdm=False)
        torch.cuda.synchronize(DEVICE)
        fallback_total_ms = (time.perf_counter() - t0) * 1000.0

        batch_out_tokens: List[int] = []
        batch_ttfts: List[float] = []
        batch_totals: List[float] = []

        for local_idx, output in enumerate(outputs):
            text, out_tokens, ttft_ms, total_ms = _extract_text_and_metrics(output, fallback_total_ms)
            batch_out_tokens.append(out_tokens)
            batch_ttfts.append(ttft_ms)
            batch_totals.append(total_ms)

            original_idx = batch_indices[local_idx]
            tps = (out_tokens / total_ms * 1000.0) if total_ms > 0 and out_tokens > 0 else 0.0
            all_results[original_idx] = {
                "prompt": prompts[original_idx],
                "output": text,
                "input_tokens": enc_lens[original_idx],
                "output_tokens": out_tokens,
                "total_latency_ms": round(total_ms, 2),
                "ttft_ms": round(ttft_ms, 2),
                "throughput_tps": round(tps, 2),
            }

        if show_progress:
            print(
                f"  [batch {batch_idx+1}/{num_batches}] mode=vllm "
                f"bs={len(batch_prompts)} ttft={sum(batch_ttfts)/max(len(batch_ttfts),1):.0f}ms "
                f"total={sum(batch_totals)/max(len(batch_totals),1):.0f}ms "
                f"out_tok={sum(batch_out_tokens)} ({end}/{n} done)"
            )

    return all_results


def infer_single(
    tokenizer,
    model: VLLMBackend,
    prompt: str,
    use_paged: bool = True,
    use_dynamic_batch: bool = True,
) -> dict:
    del use_paged, use_dynamic_batch
    return infer_all(
        tokenizer,
        model,
        [prompt],
        batch_size=1,
        show_progress=False,
    )[0]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="请用三句话解释 KV Cache 的作用。")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_path)
    result = infer_all(
        tokenizer,
        model,
        [args.prompt],
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        show_progress=False,
    )[0]

    print(f"\n{'='*60}")
    print(f"输出: {result['output'][:300]}")
    print(f"in={result['input_tokens']}  out={result['output_tokens']}")
    print(f"延迟={result['total_latency_ms']:.1f}ms  TTFT={result['ttft_ms']:.1f}ms")
    print(f"吞吐={result['throughput_tps']:.1f} tok/s")
    print(f"峰值显存={torch.cuda.max_memory_allocated(DEVICE)/1e9:.2f} GB")
    print(f"{'='*60}")