"""
统一推理引擎
==============

整合所有优化的推理引擎:
  1. 支持连续/分页 KV-Cache 切换
  2. Triton/PyTorch 融合算子 (RMSNorm, SwiGLU)
  3. Flash Attention (SDPA) — Prefill 阶段
  4. Triton PagedAttention Kernel — Decode 阶段 (paged 路径)
  5. 按长度排序分批 (减少 padding)
  6. 真正的批量贪心解码 (多序列并行生成)

Paged 路径流水线:
  Prefill  → 标准 HF forward (flash attention), 提取 KV 到 KVPool
  Decode   → monkey-patched attention forward, Triton PagedAttention kernel
"""

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from llm_inference.cache.continuous_cache import ContinuousKVCache
from llm_inference.cache.paged_cache import PagedKVCache, PagedCacheAdapter
from llm_inference.attention.paged_attention import paged_attention_decode


# ============================================================================
# 融合算子 (Triton/PyTorch)
# ============================================================================

_TRITON_IMPORTED = False
_HAS_TRITON = False

try:
    import triton
    import triton.language as tl
    _TRITON_IMPORTED = True
except ImportError:
    pass


def _probe_triton(device: str = "cuda") -> bool:
    """运行时探测 Triton 是否真正可用"""
    global _HAS_TRITON
    if _HAS_TRITON:
        return True
    if not _TRITON_IMPORTED:
        print("[INFO] Triton 未安装，使用 PyTorch 原生优化")
        return False
    try:
        @triton.jit
        def _test_kernel(X, BLOCK: tl.constexpr):
            idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            tl.store(X + idx, tl.load(X + idx) + 1.0)

        x = torch.zeros(128, device=device, dtype=torch.float32)
        _test_kernel[(1,)](x, BLOCK=128)
        torch.cuda.synchronize(device)
        if x.sum().item() == 128.0:
            _HAS_TRITON = True
            print(f"[INFO] Triton {triton.__version__} 探测成功，启用融合算子")
        else:
            raise RuntimeError("Triton 探测结果不正确")
        return _HAS_TRITON
    except Exception as e:
        _HAS_TRITON = False
        print(f"[WARN] Triton 运行时不可用: {e}")
        print("[INFO] 回退到 PyTorch 原生融合实现")
        return False


# ── Triton 内核 (模块级定义) ──────────────────────────────────

if _TRITON_IMPORTED:
    @triton.jit
    def _rms_norm_kernel(
        X, W, Y, stride_x, stride_y, N, eps, BLOCK: tl.constexpr
    ):
        row = tl.program_id(0)
        x_off = X + row * stride_x
        y_off = Y + row * stride_y
        _acc = tl.zeros([BLOCK], dtype=tl.float32)
        for b in range(0, N, BLOCK):
            cols = b + tl.arange(0, BLOCK)
            mask = cols < N
            xv = tl.load(x_off + cols, mask=mask, other=0.0).to(tl.float32)
            _acc += xv * xv
        var = tl.sum(_acc, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        for b in range(0, N, BLOCK):
            cols = b + tl.arange(0, BLOCK)
            mask = cols < N
            xv = tl.load(x_off + cols, mask=mask, other=0.0).to(tl.float32)
            wv = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
            tl.store(y_off + cols, xv * rstd * wv, mask=mask)

    @triton.jit
    def _silu_mul_kernel(
        G, U, O, stride_g, stride_u, stride_o, N, BLOCK: tl.constexpr
    ):
        row = tl.program_id(0)
        g_off = G + row * stride_g
        u_off = U + row * stride_u
        o_off = O + row * stride_o
        for b in range(0, N, BLOCK):
            cols = b + tl.arange(0, BLOCK)
            mask = cols < N
            gv = tl.load(g_off + cols, mask=mask, other=0.0).to(tl.float32)
            uv = tl.load(u_off + cols, mask=mask, other=0.0).to(tl.float32)
            sv = gv * tl.sigmoid(gv) * uv
            tl.store(o_off + cols, sv, mask=mask)


def fused_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """融合 RMSNorm (Triton 或 PyTorch)"""
    if _HAS_TRITON and x.is_cuda:
        try:
            shape = x.shape
            x2 = x.reshape(-1, shape[-1]).contiguous()
            M, N = x2.shape
            y = torch.empty_like(x2)
            BLK = min(triton.next_power_of_2(N), 4096)
            _rms_norm_kernel[(M,)](x2, weight, y, x2.stride(0), y.stride(0), N, eps, BLOCK=BLK)
            return y.reshape(shape)
        except Exception:
            pass
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps) * weight.to(torch.float32)).to(x.dtype)


def fused_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """融合 SwiGLU (SiLU×gate)"""
    if _HAS_TRITON and gate.is_cuda:
        try:
            shape = gate.shape
            g2 = gate.reshape(-1, shape[-1]).contiguous()
            u2 = up.reshape(-1, shape[-1]).contiguous()
            M, N = g2.shape
            o = torch.empty_like(g2)
            BLK = min(triton.next_power_of_2(N), 4096)
            _silu_mul_kernel[(M,)](g2, u2, o, g2.stride(0), u2.stride(0), o.stride(0), N, BLOCK=BLK)
            return o.reshape(shape)
        except Exception:
            pass
    return F.silu(gate) * up


def apply_model_optimizations(model: nn.Module) -> None:
    """应用模型融合优化 (RMSNorm + SwiGLU monkey-patch)"""
    n_rms = n_mlp = 0
    for name, module in model.named_modules():
        class_name = type(module).__name__
        if "RMSNorm" in class_name:
            weight = module.weight
            eps = getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6))
            module.forward = lambda x, w=weight, e=eps: fused_rms_norm(x, w, e)
            n_rms += 1
        elif "MLP" in class_name and hasattr(module, "gate_proj"):
            gate_proj = module.gate_proj
            up_proj = module.up_proj
            down_proj = module.down_proj
            module.forward = lambda x, gp=gate_proj, up=up_proj, dp=down_proj: dp(fused_silu_mul(gp(x), up(x)))
            n_mlp += 1
    backend = "Triton" if _HAS_TRITON else "PyTorch"
    print(f"[OPT] {backend} 融合 RMSNorm×{n_rms}, SwiGLU×{n_mlp}")


# ============================================================================
# RoPE 辅助函数 (标准 Llama/Qwen2 实现)
# ============================================================================

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """标准 RoPE，兼容 Llama / Qwen2 / Mistral 等架构"""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


# ============================================================================
# Attention Monkey-Patch (Paged Decode)
# ============================================================================

def _make_paged_attn_forward(attn_module, block_size: int, rotary_emb=None):
    """
    创建 monkey-patched attention forward。

    路由逻辑:
      - past_key_value 是 PagedCacheAdapter → Paged Decode 路径 (Triton kernel)
      - 其他 → 原始 HF forward (Prefill / Continuous)
    """
    # 缓存模块属性 (闭包捕获)
    q_proj = attn_module.q_proj
    k_proj = attn_module.k_proj
    v_proj = attn_module.v_proj
    o_proj = attn_module.o_proj
    # Qwen2 等模型可能没有 rotary_emb 属性，支持外部传入
    if rotary_emb is None and hasattr(attn_module, "rotary_emb"):
        rotary_emb = attn_module.rotary_emb
    layer_idx = attn_module.layer_idx
    head_dim = attn_module.head_dim

    # 从 config 或模块属性获取 num_heads 和 num_kv_heads
    if hasattr(attn_module, "num_heads"):
        num_heads = attn_module.num_heads
        num_kv_heads = getattr(attn_module, "num_key_value_heads", num_heads)
    elif hasattr(attn_module, "config"):
        cfg = attn_module.config
        num_heads = cfg.num_attention_heads
        num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    else:
        # 从投影层维度计算
        num_heads = q_proj.out_features // head_dim
        num_kv_heads = k_proj.out_features // head_dim

    num_kv_groups = num_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)
    original_forward = attn_module._original_forward

    def forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        # ── Gate: 仅 PagedCacheAdapter 走 paged 路径 ──
        if not isinstance(past_key_values, PagedCacheAdapter):
            return original_forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        bsz, q_len, _ = hidden_states.size()
        assert q_len == 1, (
            f"PagedAttention decode 期望 q_len=1，实际 q_len={q_len}"
        )

        # ── Q, K, V 投影 ──
        query_states = q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        # ── RoPE ──
        if position_embeddings is not None:
            cos, sin = position_embeddings
        elif rotary_emb is not None:
            cos, sin = rotary_emb(value_states, position_ids)
        else:
            # 没有 rotary_emb 也没有 position_embeddings，跳过 RoPE
            cos, sin = None, None
        if cos is not None and sin is not None:
            query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # ── 写入新 K, V 到 paged cache ──
        adapter: PagedCacheAdapter = past_key_values
        adapter.write_kv(layer_idx, key_states, value_states)

        # ── Triton PagedAttention Decode ──
        k_cache, v_cache = adapter.get_kv_cache(layer_idx)
        attn_output = paged_attention_decode(
            query_states,                   # [B, num_heads, 1, head_dim]
            k_cache,                        # [num_blocks, block_size, num_kv_heads, head_dim]
            v_cache,
            adapter.block_tables_tensor,    # [B, max_blocks]
            adapter.seq_lens_tensor,        # [B]
            scale,
            num_kv_groups,
            block_size,
        )

        # ── Output projection ──
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, num_heads * head_dim)
        attn_output = o_proj(attn_output)

        return attn_output, None

    return forward


def _patch_attention_for_paged(model: nn.Module, block_size: int) -> int:
    """
    Monkey-patch 所有 attention 层。

    patched forward 的路由逻辑保证:
      - Prefill (past_key_values=None 或 DynamicCache) → 走原始 HF forward
      - Decode (past_key_values=PagedCacheAdapter) → 走 Triton PagedAttention
    """
    # 尝试从模型中获取 rotary_emb (Qwen2 等模型将其放在模型级别)
    rotary_emb = None
    if hasattr(model, "model"):
        rotary_emb = getattr(model.model, "rotary_emb", None)

    patched = 0
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if (
            "Attention" in cls_name
            and hasattr(module, "q_proj")
            and hasattr(module, "layer_idx")
        ):
            module._original_forward = module.forward
            module.forward = _make_paged_attn_forward(module, block_size, rotary_emb)
            patched += 1
    return patched


# ============================================================================
# 推理结果
# ============================================================================

@dataclass
class InferenceResult:
    prompt: str
    output: str
    input_tokens: int
    output_tokens: int
    total_latency_ms: float
    ttft_ms: float
    throughput_tps: float


# ============================================================================
# 推理引擎
# ============================================================================

class InferenceEngine:
    """
    统一推理引擎

    支持两种 KV-Cache 模式:
      continuous — HF 原生 past_key_values，批量 decode
      paged     — KVPool + Triton PagedAttention，monkey-patched decode
    """

    def __init__(
        self,
        model_path: str,
        cache_type: str = "continuous",
        block_size: int = 16,
        num_blocks: int = 1000,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
        enable_optimizations: bool = True,
        batch_size: int = 32,
    ):
        self.model_path = model_path
        self.cache_type = cache_type
        self.block_size = block_size
        self.num_blocks = num_blocks
        self._device = torch.device(device)
        self.dtype = dtype
        self.enable_optimizations = enable_optimizations
        self.batch_size = batch_size

        self._load_model()

    def _load_model(self):
        print(f"[INFO] 加载模型: {self.model_path}")
        print(f"[INFO] 设备={self._device}  精度={self.dtype}")

        # ── Tokenizer ──
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── Model ──
        self.model = None
        for attn_impl in ["flash_attention_2", "sdpa", "eager"]:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=self.dtype,
                    device_map=str(self._device),
                    trust_remote_code=True,
                    attn_implementation=attn_impl,
                )
                print(f"[OPT] Attention: {attn_impl}")
                break
            except Exception:
                continue
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                device_map=str(self._device),
                trust_remote_code=True,
            )
            print("[OPT] Attention: default")
        self.model.eval()

        # ── Model 配置 ──
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        self.num_kv_heads = getattr(self.model.config, "num_key_value_heads", self.num_heads)
        self.head_dim = self.model.config.hidden_size // self.num_heads

        # ── Cache 初始化 ──
        self._init_cache()

        # ── 融合优化 (RMSNorm + SwiGLU) ──
        if self.enable_optimizations:
            _probe_triton(str(self._device))
            apply_model_optimizations(self.model)

        # ── PagedAttention monkey-patch ──
        if self.cache_type == "paged":
            n = _patch_attention_for_paged(self.model, self.block_size)
            print(f"[OPT] PagedAttention: patched {n} attention layers")

        # ── 预热 ──
        print("[INFO] 预热推理...")
        warm_ids = self.tokenizer("hello world", return_tensors="pt").to(self._device)
        with torch.inference_mode():
            for _ in range(3):
                self.model(**warm_ids, use_cache=True)
        torch.cuda.synchronize(self._device)

        n_params = sum(p.numel() for p in self.model.parameters()) / 1e9
        vram = torch.cuda.memory_allocated(self._device) / 1e9
        print(f"[INFO] 就绪 | {n_params:.1f}B params | VRAM {vram:.2f} GB")
        print(f"[INFO] Cache: {self.cache_type}")

    def _init_cache(self):
        if self.cache_type == "continuous":
            self.cache = ContinuousKVCache(
                num_layers=self.num_layers,
                num_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                device=self._device,
                dtype=self.dtype,
            )
        elif self.cache_type == "paged":
            self.cache = PagedKVCache(
                num_layers=self.num_layers,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                block_size=self.block_size,
                num_blocks=self.num_blocks,
                device=self._device,
                dtype=self.dtype,
            )
        else:
            raise ValueError(f"Unknown cache type: {self.cache_type}")

    def _get_eos_ids(self) -> Set[int]:
        ids = set()
        if self.tokenizer.eos_token_id is not None:
            ids.add(self.tokenizer.eos_token_id)
        for s in ["", "<|im_end|>", "<|end|>", "</s>"]:
            try:
                t = self.tokenizer.convert_tokens_to_ids(s)
                if t is not None and t != getattr(self.tokenizer, "unk_token_id", -1):
                    ids.add(t)
            except Exception:
                pass
        return ids if ids else {self.tokenizer.eos_token_id or 0}

    # ====================================================================
    # Continuous 批量 decode (HF 原生 past_key_values)
    # ====================================================================

    @torch.inference_mode()
    def _batch_decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
    ) -> Tuple[List[str], List[int], List[int], float, float]:
        device = self._device
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        eos_ids = self._get_eos_ids()
        B = input_ids.shape[0]

        unfinished = torch.ones(B, dtype=torch.bool, device=device)
        generated = []
        sample_lengths = torch.zeros(B, dtype=torch.long, device=device)
        past = None
        cur_ids = input_ids
        cur_mask = attention_mask
        eos_t = torch.tensor(sorted(eos_ids), dtype=torch.long, device=device)

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        ttft = None

        for step in range(max_new_tokens):
            out = self.model(
                input_ids=cur_ids,
                attention_mask=cur_mask,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            logits = out.logits[:, -1, :]
            past = out.past_key_values
            next_tok = logits.argmax(dim=-1)

            if step == 0:
                torch.cuda.synchronize(device)
                ttft = (time.perf_counter() - t0) * 1000.0

            next_tok = torch.where(unfinished, next_tok, torch.full_like(next_tok, pad_id))
            is_eos = (next_tok.unsqueeze(1) == eos_t.unsqueeze(0)).any(dim=1)
            sample_lengths += (unfinished & ~is_eos).long()
            generated.append(next_tok.unsqueeze(1))
            unfinished = unfinished & ~is_eos
            if not unfinished.any():
                break

            cur_ids = next_tok.unsqueeze(1)
            cur_mask = torch.cat(
                [cur_mask, torch.ones(B, 1, device=device, dtype=cur_mask.dtype)], dim=1,
            )

        torch.cuda.synchronize(device)
        total_ms = (time.perf_counter() - t0) * 1000.0
        if ttft is None:
            ttft = total_ms

        if generated:
            gen_ids = torch.cat(generated, dim=1)
        else:
            gen_ids = torch.zeros(B, 0, dtype=torch.long, device=device)

        lengths = sample_lengths.tolist()
        texts = []
        for i in range(B):
            L = min(max(lengths[i], 0), gen_ids.shape[1])
            ids_list = gen_ids[i, :L].tolist() if L > 0 else []
            texts.append(self.tokenizer.decode(ids_list, skip_special_tokens=True))

        return texts, lengths, attention_mask.sum(dim=1).tolist(), ttft, total_ms

    # ====================================================================
    # Paged 批量 decode (KVPool + Triton PagedAttention)
    # ====================================================================

    @torch.inference_mode()
    def _batch_decode_paged(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
    ) -> Tuple[List[str], List[int], List[int], float, float]:
        """
        Paged KV-Cache 批量 decode。

        流水线:
          1. Prefill: 标准 HF forward (flash attention / SDPA)
          2. 提取 prefill KV → KVPool (按 block 分片写入)
          3. Decode: monkey-patched attention + Triton PagedAttention kernel
        """
        device = self._device
        B = input_ids.shape[0]
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        eos_ids = self._get_eos_ids()
        eos_t = torch.tensor(sorted(eos_ids), dtype=torch.long, device=device)
        real_lens = attention_mask.sum(dim=1).tolist()

        # ── 1. Reset & Allocate 序列 ─────────────────────────
        self.cache.reset()
        seq_ids = list(range(B))
        for sid in seq_ids:
            self.cache.allocate_seq(sid)

        # ── 2. Prefill (标准 HF forward) ─────────────────────
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )

        torch.cuda.synchronize(device)
        ttft = (time.perf_counter() - t0) * 1000.0

        # ── 3. 提取 prefill KV 到 KVPool ─────────────────────
        past_kv = outputs.past_key_values
        for layer_idx in range(self.num_layers):
            kv = past_kv[layer_idx]
            k_full, v_full = kv[0], kv[1]
            # k_full: [B, num_kv_heads, max_prompt_len, head_dim]
            for i in range(B):
                rl = int(real_lens[i])
                # left-padding: 真实 token 在右侧，取最后 rl 个
                # [num_kv_heads, rl, head_dim] → [rl, num_kv_heads, head_dim]
                k_seq = k_full[i, :, -rl:, :].permute(1, 0, 2).contiguous()
                v_seq = v_full[i, :, -rl:, :].permute(1, 0, 2).contiguous()
                self.cache.append(i, layer_idx, k_seq, v_seq, start_pos=0)

        logits = outputs.logits[:, -1, :]

        # ── KVPool 状态确认 ───────────────────────────────────
        blocks_used = self.cache.pool.num_blocks - self.cache.allocator.num_free
        print(
            f"[Paged] prefill KV → KVPool  "
            f"seqs={B}  seq_lens={[int(r) for r in real_lens]}  "
            f"blocks_used={blocks_used}/{self.cache.pool.num_blocks}"
        )

        # ── 4. Decode 循环 (PagedAttention) ──────────────────
        adapter = PagedCacheAdapter(self.cache, seq_ids)
        unfinished = torch.ones(B, dtype=torch.bool, device=device)
        generated: List[torch.Tensor] = []
        sample_lengths = torch.zeros(B, dtype=torch.long, device=device)

        for step in range(max_new_tokens):
            # 贪心采样
            next_tok = logits.argmax(dim=-1)
            next_tok = torch.where(unfinished, next_tok, torch.full_like(next_tok, pad_id))

            # EOS 检测
            is_eos = (next_tok.unsqueeze(1) == eos_t.unsqueeze(0)).any(dim=1)
            sample_lengths += (unfinished & ~is_eos).long()
            generated.append(next_tok.unsqueeze(1))

            unfinished = unfinished & ~is_eos
            if not unfinished.any():
                break

            # ── 准备 decode step ──
            cur_ids = next_tok.unsqueeze(1)  # [B, 1]

            if step == 0:
                print(
                    f"[Paged] decode step 0: model.forward(past_key_values=PagedCacheAdapter)  "
                    f"active_seqs={int(unfinished.sum())}"
                )

            adapter.begin_step(unfinished)

            # position_ids: 每个序列的当前位置 = seq_lens_tensor - 1
            # (seq_lens_tensor 已包含 +1，所以 -1 得到新 token 的 0-based position)
            position_ids = (adapter.seq_lens_tensor.long() - 1).unsqueeze(1)  # [B, 1]

            outputs = self.model(
                input_ids=cur_ids,
                position_ids=position_ids,
                past_key_values=adapter,
                use_cache=True,
                return_dict=True,
            )

            adapter.end_step()
            logits = outputs.logits[:, -1, :]

        torch.cuda.synchronize(device)
        total_ms = (time.perf_counter() - t0) * 1000.0

        # ── 5. 解码文本 ─────────────────────────────────────
        if generated:
            gen_ids = torch.cat(generated, dim=1)
        else:
            gen_ids = torch.zeros(B, 0, dtype=torch.long, device=device)

        lengths = sample_lengths.tolist()
        texts = []
        for i in range(B):
            L = min(max(lengths[i], 0), gen_ids.shape[1])
            ids_list = gen_ids[i, :L].tolist() if L > 0 else []
            texts.append(self.tokenizer.decode(ids_list, skip_special_tokens=True))

        return texts, lengths, [int(r) for r in real_lens], ttft, total_ms

    # ====================================================================
    # 公共接口
    # ====================================================================

    def infer_single(self, prompt: str, max_new_tokens: int = 256) -> InferenceResult:
        results = self.infer_batch([prompt], max_new_tokens=max_new_tokens)
        return results[0]

    def infer_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        show_progress: bool = False,
    ) -> List[InferenceResult]:
        n = len(prompts)
        if n == 0:
            return []

        # 按长度排序 (减少 padding)
        enc_lens = [len(self.tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
        sorted_idx = sorted(range(n), key=lambda i: enc_lens[i])

        all_results: List[Optional[InferenceResult]] = [None] * n
        num_batches = math.ceil(n / self.batch_size)

        for b in range(num_batches):
            start = b * self.batch_size
            end = min(start + self.batch_size, n)
            batch_indices = sorted_idx[start:end]
            batch_prompts = [prompts[i] for i in batch_indices]
            bs = len(batch_prompts)

            # 记录batch开始时的blocks使用情况
            if self.cache_type == "paged":
                blocks_before = self.cache.allocator.num_free

            enc = self.tokenizer(
                batch_prompts, return_tensors="pt",
                padding=True, truncation=True, max_length=2048,
            ).to(self._device)

            if self.cache_type == "paged":
                texts, out_lens, in_lens, ttft, total = self._batch_decode_paged(
                    enc["input_ids"], enc["attention_mask"], max_new_tokens,
                )
            else:
                texts, out_lens, in_lens, ttft, total = self._batch_decode(
                    enc["input_ids"], enc["attention_mask"], max_new_tokens,
                )

            for i in range(bs):
                original_idx = batch_indices[i]
                tps = (out_lens[i] / total * 1000.0) if (total > 0 and out_lens[i] > 0) else 0.0
                all_results[original_idx] = InferenceResult(
                    prompt=batch_prompts[i],
                    output=texts[i],
                    input_tokens=in_lens[i],
                    output_tokens=out_lens[i],
                    total_latency_ms=round(total, 2),
                    ttft_ms=round(ttft, 2),
                    throughput_tps=round(tps, 2),
                )

            if show_progress:
                print(
                    f"  [batch {b+1}/{num_batches}] [{self.cache_type}]  "
                    f"bs={bs}  ttft={ttft:.0f}ms  total={total:.0f}ms  "
                    f"out_tok={sum(out_lens)}  ({end}/{n} done)"
                )

            # 打印blocks使用情况
            if self.cache_type == "paged":
                blocks_after = self.cache.allocator.num_free
                blocks_used = blocks_before - blocks_after
                total_blocks = self.cache.pool.num_blocks
                print(
                    f"  [batch {b+1}/{num_batches}] blocks_used={blocks_used}  "
                    f"blocks_free={blocks_after}/{total_blocks}  "
                    f"blocks_usage_pct={(total_blocks-blocks_after)/total_blocks*100:.1f}%"
                )

        return all_results

    def get_cache_memory_usage(self) -> int:
        return self.cache.get_memory_usage()
