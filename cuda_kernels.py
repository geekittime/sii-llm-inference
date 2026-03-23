import os
import sys
import traceback
from pathlib import Path
from typing import Optional

import torch

_CUDA_OPS = None
_CUDA_OPS_ERROR: Optional[str] = None

_ROOT = Path(__file__).resolve().parent
_SOURCE_DIR = _ROOT / "cuda_ops"
_CPP_SOURCE = _SOURCE_DIR / "optimized_ops.cpp"
_CU_SOURCE = _SOURCE_DIR / "optimized_ops_kernel.cu"


def _extension_name() -> str:
    cuda_tag = (torch.version.cuda or "cpu").replace(".", "_")
    py_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    return f"optimized_cuda_ops_{cuda_tag}_{py_tag}"


def load_cuda_ops(verbose: bool = False):
    global _CUDA_OPS, _CUDA_OPS_ERROR

    if _CUDA_OPS is not None:
        return _CUDA_OPS
    if _CUDA_OPS_ERROR is not None:
        return None
    if not torch.cuda.is_available():
        _CUDA_OPS_ERROR = "CUDA is not available"
        return None

    try:
        from torch.utils.cpp_extension import load

        build_dir = _ROOT / ".torch_extensions" / _extension_name()
        build_dir.mkdir(parents=True, exist_ok=True)

        extra_cflags = ["/O2"] if os.name == "nt" else ["-O3"]
        extra_cuda_cflags = ["-O3", "--use_fast_math"]
        if os.name == "nt":
            extra_cuda_cflags.extend(["-Xcompiler", "/O2"])

        _CUDA_OPS = load(
            name=_extension_name(),
            sources=[str(_CPP_SOURCE), str(_CU_SOURCE)],
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            build_directory=str(build_dir),
            verbose=verbose,
        )
        return _CUDA_OPS
    except Exception:
        _CUDA_OPS_ERROR = traceback.format_exc()
        return None


def get_last_error() -> Optional[str]:
    return _CUDA_OPS_ERROR


def probe_cuda_ops(device: str = "cuda:0", verbose: bool = False) -> bool:
    ops = load_cuda_ops(verbose=verbose)
    if ops is None:
        if _CUDA_OPS_ERROR:
            print("[WARN] CUDA 扩展加载失败，回退到 PyTorch 实现")
            first_line = _CUDA_OPS_ERROR.strip().splitlines()[-1]
            print(f"[WARN] {first_line}")
        return False

    try:
        x = torch.randn(4, 128, device=device, dtype=torch.float16)
        w = torch.ones(128, device=device, dtype=torch.float16)
        y = ops.rms_norm_forward(x, w, 1e-6)
        torch.cuda.synchronize(device)
        ref = ((x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)) * w.float()).to(x.dtype)
        if not torch.allclose(y, ref, atol=5e-3, rtol=5e-3):
            raise RuntimeError("CUDA RMSNorm probe mismatch")
        print("[INFO] 自定义 CUDA 算子可用，启用 CUDA kernel 路径")
        return True
    except Exception as exc:
        print(f"[WARN] CUDA 扩展运行失败: {exc}")
        print("[INFO] 回退到 PyTorch 实现")
        return False


def cuda_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    ops = load_cuda_ops()
    if ops is None:
        raise RuntimeError("CUDA ops extension is not available")
    return ops.rms_norm_forward(x.contiguous(), weight.contiguous(), float(eps))


def cuda_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    ops = load_cuda_ops()
    if ops is None:
        raise RuntimeError("CUDA ops extension is not available")
    return ops.silu_mul_forward(gate.contiguous(), up.contiguous())


def cuda_store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    ops = load_cuda_ops()
    if ops is None:
        raise RuntimeError("CUDA ops extension is not available")
    ops.store_kvcache_forward(
        key.contiguous(),
        value.contiguous(),
        k_cache,
        v_cache,
        slot_mapping.contiguous(),
        int(block_size),
    )


def cuda_paged_attention_decode(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> torch.Tensor:
    ops = load_cuda_ops()
    if ops is None:
        raise RuntimeError("CUDA ops extension is not available")
    return ops.paged_attention_decode_forward(
        query.contiguous(),
        k_cache,
        v_cache,
        block_tables.contiguous().to(dtype=torch.int32),
        context_lens.contiguous().to(dtype=torch.long),
        float(scale),
        int(num_heads),
        int(num_kv_heads),
        int(head_dim),
        int(block_size),
    )