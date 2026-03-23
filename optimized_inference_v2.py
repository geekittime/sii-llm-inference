import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import optimized_inference as _base
from cuda_kernels import (
    cuda_paged_attention_decode,
    cuda_rms_norm,
    cuda_silu_mul,
    cuda_store_kvcache,
    probe_cuda_ops,
)

DEVICE = _base.DEVICE
DTYPE = _base.DTYPE
MAX_NEW_TOKENS = _base.MAX_NEW_TOKENS
BATCH_SIZE = _base.BATCH_SIZE
SEED = _base.SEED
PAGE_BLOCK_SIZE = _base.PAGE_BLOCK_SIZE
DYNAMIC_REFILL_RATIO = _base.DYNAMIC_REFILL_RATIO
DYNAMIC_MIN_ADMIT = _base.DYNAMIC_MIN_ADMIT
DYNAMIC_PROGRESS_INTERVAL = _base.DYNAMIC_PROGRESS_INTERVAL
RequestState = _base.RequestState
PagedKVCache = _base.PagedKVCache

HAS_CUDA_KERNELS = False
_CUDA_PROBED = False
_CUDA_FAILURE_REPORTED = False


def _install_cuda_hooks() -> None:
    _base.fused_rms_norm = fused_rms_norm
    _base.fused_silu_mul = fused_silu_mul
    _base.store_kvcache = store_kvcache
    _base.paged_attention_decode = paged_attention_decode


def _ensure_cuda_kernels() -> bool:
    global HAS_CUDA_KERNELS, _CUDA_PROBED
    if not _CUDA_PROBED:
        HAS_CUDA_KERNELS = probe_cuda_ops(device=DEVICE)
        _CUDA_PROBED = True
    return HAS_CUDA_KERNELS


def _report_cuda_fallback(exc: Exception) -> None:
    global _CUDA_FAILURE_REPORTED, HAS_CUDA_KERNELS
    if not _CUDA_FAILURE_REPORTED:
        print(f"[WARN] CUDA kernel 执行失败，回退到 PyTorch: {exc}")
        _CUDA_FAILURE_REPORTED = True
    HAS_CUDA_KERNELS = False


def fused_rms_norm(x, w, eps=1e-6):
    if x.is_cuda and _ensure_cuda_kernels():
        try:
            return cuda_rms_norm(x, w, eps)
        except Exception as exc:
            _report_cuda_fallback(exc)
    return _base.pt_rms_norm(x, w, eps)


def fused_silu_mul(g, u):
    if g.is_cuda and _ensure_cuda_kernels():
        try:
            return cuda_silu_mul(g, u)
        except Exception as exc:
            _report_cuda_fallback(exc)
    return _base.pt_silu_mul(g, u)


def store_kvcache(key, value, k_cache, v_cache, slot_mapping, block_size):
    if key.is_cuda and _ensure_cuda_kernels():
        try:
            cuda_store_kvcache(key, value, k_cache, v_cache, slot_mapping, block_size)
            return
        except Exception as exc:
            _report_cuda_fallback(exc)
    _base._pt_store_kvcache(key, value, k_cache, v_cache, slot_mapping, block_size)


def paged_attention_decode(
    query,
    k_cache,
    v_cache,
    block_tables,
    context_lens,
    scale,
    num_heads,
    num_kv_heads,
    head_dim,
    block_size,
):
    if query.is_cuda and _ensure_cuda_kernels():
        try:
            return cuda_paged_attention_decode(
                query,
                k_cache,
                v_cache,
                block_tables,
                context_lens,
                scale,
                num_heads,
                num_kv_heads,
                head_dim,
                block_size,
            ).to(query.dtype)
        except Exception as exc:
            _report_cuda_fallback(exc)
    return _base._pt_paged_attn_decode(
        query,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        scale,
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
    )


def _make_rmsnorm_fwd(mod):
    w = mod.weight
    eps = getattr(mod, "variance_epsilon", getattr(mod, "eps", 1e-6))

    def fwd(x):
        return fused_rms_norm(x, w, eps)

    return fwd


def _make_mlp_fwd(mod):
    gp, up, dp = mod.gate_proj, mod.up_proj, mod.down_proj

    def fwd(x):
        return dp(fused_silu_mul(gp(x), up(x)))

    return fwd


def apply_optimizations(model):
    nr = nm = 0
    for _, module in model.named_modules():
        class_name = type(module).__name__
        if "RMSNorm" in class_name:
            module.forward = _make_rmsnorm_fwd(module)
            nr += 1
        if "MLP" in class_name and hasattr(module, "gate_proj"):
            module.forward = _make_mlp_fwd(module)
            nm += 1
    tag = "CUDA" if HAS_CUDA_KERNELS else "PyTorch"
    print(f"[OPT] {tag} 融合 RMSNorm×{nr}, SwiGLU×{nm}")


def load_model(model_path: str):
    _install_cuda_hooks()

    print(f"[INFO] 加载模型: {model_path}")
    print(f"[INFO] 设备={DEVICE}  精度={DTYPE}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None
    for attn in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=DTYPE,
                device_map=DEVICE,
                trust_remote_code=True,
                attn_implementation=attn,
            )
            print(f"[OPT] Attention backend: {attn}")
            break
        except Exception:
            continue
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=DTYPE,
            device_map=DEVICE,
            trust_remote_code=True,
        )
        print("[OPT] Attention backend: default")

    model.eval()
    _ensure_cuda_kernels()
    apply_optimizations(model)

    print("[INFO] 预热推理...")
    warmup = tokenizer("hello world", return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        for _ in range(3):
            model(**warmup, use_cache=True)
    torch.cuda.synchronize(DEVICE)

    n_params = sum(param.numel() for param in model.parameters()) / 1e9
    vram = torch.cuda.memory_allocated(DEVICE) / 1e9
    print(f"[INFO] 就绪 | {n_params:.1f}B params | VRAM {vram:.2f} GB")
    return tokenizer, model


def infer_all(
    tokenizer,
    model,
    prompts: list,
    batch_size: int = BATCH_SIZE,
    max_new_tokens: int = MAX_NEW_TOKENS,
    show_progress: bool = True,
    use_paged: bool = True,
    use_dynamic_batch: bool = True,
):
    _install_cuda_hooks()
    return _base.infer_all(
        tokenizer,
        model,
        prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        show_progress=show_progress,
        use_paged=use_paged,
        use_dynamic_batch=use_dynamic_batch,
    )


def infer_single(
    tokenizer,
    model,
    prompt: str,
    use_paged: bool = True,
    use_dynamic_batch: bool = True,
) -> dict:
    _install_cuda_hooks()
    return _base.infer_single(
        tokenizer,
        model,
        prompt,
        use_paged=use_paged,
        use_dynamic_batch=use_dynamic_batch,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="请用三句话解释 KV Cache 的作用。")
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--no_paged", action="store_true")
    parser.add_argument("--no_dynamic_batch", action="store_true")
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_path)
    if args.batch_size <= 0:
        args.batch_size = _base._auto_batch_size(model, tokenizer)
        print(f"[INFO] auto batch_size = {args.batch_size}")

    result = infer_single(
        tokenizer,
        model,
        args.prompt,
        use_paged=not args.no_paged,
        use_dynamic_batch=not args.no_dynamic_batch,
    )
    print(f"\n{'=' * 60}")
    print(f"输出: {result['output'][:300]}")
    print(f"in={result['input_tokens']}  out={result['output_tokens']}")
    print(f"延迟={result['total_latency_ms']:.1f}ms  TTFT={result['ttft_ms']:.1f}ms")
    print(f"吞吐={result['throughput_tps']:.1f} tok/s")
    print(f"峰值显存={torch.cuda.max_memory_allocated(DEVICE) / 1e9:.2f} GB")
    print(f"{'=' * 60}")