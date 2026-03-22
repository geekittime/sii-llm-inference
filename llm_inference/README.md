# LLM Inference Framework

一个高性能的大语言模型推理框架，实现了 **PagedAttention** 和 **Paged KV-Cache**，支持连续和分页两种 KV-Cache 模式。

版本：`0.3.0`

---

## 特性

- **Paged KV-Cache**：借鉴操作系统虚拟内存管理思想，将 KV Cache 划分为固定大小的 Block 存储，消除内存碎片，支持按需动态分配
- **PagedAttention Triton Kernel**：Triton 加速的分页注意力计算，Decode 阶段直接按 block_table 索引访问 KVPool，无需拼接分散的 KV 块；内置 GQA (Grouped Query Attention) 支持
- **双模式支持**：
  - `continuous`：基于 HuggingFace `past_key_values` 的连续内存实现，直接利用框架原生缓存
  - `paged`：KVPool 分页内存 + Triton PagedAttention Kernel，Decode 阶段通过 monkey-patch 替换 Attention 前向
- **融合算子优化**：Triton 加速的 RMSNorm 和 SwiGLU (SiLU×gate) 融合，运行时自动探测 Triton 可用性，不可用时无缝回退到 PyTorch 实现
- **自适应 Attention 后端**：模型加载时依次尝试 `flash_attention_2` → `sdpa` → `eager`，自动选择最优后端
- **批量推理**：支持真正的批量贪心解码，按输入长度排序分批减少 padding，多序列并行生成
- **性能基准测试**：内置吞吐量 & 延迟 benchmark 工具，以及 continuous/paged 双模式对比评测脚本

---

## 项目结构

```
sii-llm-inference/
├── benchmark_llm_inference.py      # 吞吐量 & 延迟基准测试工具
├── prompts.jsonl                   # 测试 prompt 文件 (JSONL 格式)
└── llm_inference/
    ├── __init__.py                 # 包入口，暴露所有公共 API
    ├── cli.py                      # 命令行入口
    ├── attention/
    │   ├── __init__.py
    │   └── paged_attention.py      # Triton PagedAttention Kernel + PyTorch 回退
    ├── block_manager/
    │   ├── __init__.py
    │   ├── block_allocator.py      # 物理 Block 索引分配器 (类 OS 页帧分配器)
    │   └── block_table.py          # 页表 (seq_id → block 列表 + seq_len 映射)
    ├── cache/
    │   ├── __init__.py
    │   ├── continuous_cache.py     # 连续内存 KV-Cache (显存统计占位符)
    │   └── paged_cache.py          # KVPool / PagedKVCache / PagedCacheAdapter
    ├── evaluation/
    │   ├── __init__.py
    │   └── compare_cache.py        # continuous vs paged 对比评测脚本
    └── inference/
        ├── __init__.py
        └── engine.py               # 统一推理引擎 (InferenceEngine + InferenceResult)
```

---

## 核心实现：Paged Cache & Paged Attention

### 一、连续 KV-Cache (`cache/continuous_cache.py`)

`ContinuousKVCache` 是 continuous 模式下的显存统计占位符。引擎实际的 KV 缓存由 HuggingFace `past_key_values` 机制自动管理，`ContinuousKVCache` 仅负责统计已写入的 token 量以供 `get_memory_usage()` 使用。

#### 内部数据结构

```
_cache: Dict[seq_id, Dict[layer_idx, (K: Tensor, V: Tensor)]]
# K/V shape: [num_tokens, num_kv_heads, head_dim]
```

#### 方法

| 方法 | 说明 |
|---|---|
| `append(seq_id, layer_idx, k, v)` | 追加 KV；首次写入 clone 存储，后续 `torch.cat` 拼接 |
| `get_num_cached_tokens(seq_id)` | 读取第 0 层 K 的行数 |
| `reset()` | 清空所有缓存 |
| `get_memory_usage() -> int` | 估算：`total_tokens × 2 × num_heads × head_dim × element_size` |

---

### 二、分页 KV-Cache (`cache/paged_cache.py`)

分页 Cache 由三个独立组件协作构成：

```
┌─────────────────────────────────────────────────────────────────┐
│                      PagedKVCache Manager                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │   KVPool    │  │  BlockAllocator  │  │   BlockTable     │   │
│  │ (GPU 显存池) │  │  (物理块索引分配)  │  │ (序列 → 块映射)  │   │
│  └─────────────┘  └──────────────────┘  └──────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.1 KVPool — 预分配 GPU 显存池

```python
class KVPool:
    # 布局 (每层独立 Tensor，避免 5D Tensor 的 stride 复杂性)
    k: List[Tensor]  # k[layer]: [num_blocks, block_size, num_kv_heads, head_dim]
    v: List[Tensor]  # v[layer]: 同上
```

**设计要点**：
- `head_dim` 为最内层维度（连续内存），保证 Triton Kernel 的 coalesced access
- 所有 Block 在初始化时一次性分配（`torch.zeros`），运行时零额外显存分配
- 提供两种写入接口：

```python
# 单 token 写入（decode 阶段每步写一个 token）
pool.write_slot(layer_idx, block_id, offset,
                k_token,   # [num_kv_heads, head_dim]
                v_token)

# 批量写入（prefill 阶段批量写入一个 block 的连续 token）
pool.write_block_range(layer_idx, block_id, start,
                       k_chunk,  # [n_tokens, num_kv_heads, head_dim]
                       v_chunk)
```

#### 3.2 BlockAllocator — 物理 Block 索引分配器

```python
class BlockAllocator:
    _free: deque   # 空闲 block_id 队列，初始化为 range(num_blocks)

    def allocate(self) -> int:       # 弹出队头 block_id，耗尽时抛 RuntimeError
    def free(self, block_id: int)    # 归还单个 block_id
    def free_many(self, ids: list)   # 批量归还
    def reset()                      # 重置为全空闲
    @property num_free: int          # 当前空闲 block 数
```

类似 OS 物理页帧分配器，纯索引管理，不持有任何 KV 数据。

#### 3.3 BlockTable — 页表

```python
class BlockTable:
    _blocks:   Dict[int, List[int]]  # seq_id → [block_id_0, block_id_1, ...]
    _seq_lens: Dict[int, int]        # seq_id → 已缓存 token 数
```

| 方法 | 说明 |
|---|---|
| `add_seq(seq_id)` | 注册新序列，初始化空块列表和 seq_len=0 |
| `remove_seq(seq_id) -> List[int]` | 移除序列，返回其持有的 block 列表（供 allocator 回收） |
| `append_block(seq_id, block_id)` | 追加一个物理 block |
| `get_blocks(seq_id) -> List[int]` | 返回当前 block 列表 |
| `get_seq_len / set_seq_len / add_tokens` | token 计数读写 |
| `to_tensor(seq_ids, device)` | 导出为 Triton Kernel 所需的 Tensor 格式（见下） |

`to_tensor` 输出格式：

```python
block_tables: Tensor  # [B, max_blocks_per_seq]  int32，不足的位置填 0
seq_lens:     Tensor  # [B]                       int32
```

#### 3.4 PagedKVCache — 分页缓存管理器

整合上述三组件：

```python
cache = PagedKVCache(
    num_layers=32,
    num_kv_heads=8,
    head_dim=128,
    block_size=16,      # 每个 block 可存 16 个 token 的 KV
    num_blocks=1000,    # 总共预分配 1000 个 block
    device=torch.device("cuda:0"),
    dtype=torch.float16,
)

cache.allocate_seq(seq_id=0)         # 注册序列（不预分配 block）

cache.append(                         # 写入 KV，自动按 block_size 分片
    seq_id=0, layer_idx=0,
    k=torch.randn(20, 8, 128),        # 20 个 token → block0 写 16 个，block1 写 4 个
    v=torch.randn(20, 8, 128),
)
# 注意：seq_len 只在 layer_idx==0 时更新，避免多层 append 重复计数

k_cache, v_cache = cache.get_kv_cache(layer_idx=0)   # 返回整个层的 KVPool Tensor 引用

cache.free_seq(seq_id=0)             # 释放，回收其全部 block
cache.reset()                        # 重置（不清零 pool，下次写入时覆盖）
```

#### 3.5 PagedCacheAdapter — HF Cache 接口适配器

将 `PagedKVCache` 桥接到 HuggingFace `past_key_values` 接口，作为 Decode 循环的协调中心：

```python
adapter = PagedCacheAdapter(cache, seq_ids)  # seq_ids: List[int]

# Decode 循环
for step in range(max_new_tokens):
    adapter.begin_step(active_mask)      # 分配 block，构建 block_tables_tensor / seq_lens_tensor
    model.forward(..., past_key_values=adapter)
    adapter.end_step()                   # 活跃序列 seq_len += 1

# monkey-patched attention 内部使用：
adapter.write_kv(layer_idx, key_states, value_states)   # 写入新 token KV
k_cache, v_cache = adapter.get_kv_cache(layer_idx)      # 获取 KVPool Tensor 引用
# → paged_attention_decode(query, k_cache, v_cache,
#        adapter.block_tables_tensor,  # [B, max_blocks]
#        adapter.seq_lens_tensor,      # [B]  int32（含当前新 token，即原 seq_len + 1）
#        ...)
```

`begin_step` 细节：
1. 遍历活跃序列，记录写入位置 `_write_positions[sid] = current_seq_len`，并确保该位置所在的 block 已分配
2. 构建 `seq_lens_tensor`：活跃序列为 `current_seq_len + 1`（新 token 写入后 kernel 需要 attend 到它），已结束序列保持原值
3. 调用 `block_table.to_tensor` 构建 `block_tables_tensor`

HF Cache 兼容接口（供 model forward 内部使用）：
- `get_seq_length(layer_idx)` → 返回最长序列的 seq_len
- `get_max_length()` → `None`
- `seen_tokens` property → 同 `get_seq_length()`
- `update(key_states, value_states, layer_idx, cache_kwargs)` → 直接返回原始 states（实际写入由 `write_kv` 完成）
- `is_compileable` → `False`

---

### 四、PagedAttention Triton Kernel (`attention/paged_attention.py`)

#### 4.1 核心思想

传统 Attention 在 Decode 时需要将分散的 KV 块拼接成连续 Tensor；PagedAttention 直接按 block_table 索引访问 KVPool，省去拼接开销：

```
传统方式:
  K: [token_0, ..., token_n]  ← 需要先拼接分散的 block

PagedAttention:
  Block_table: [3, 7, 2, 9, ...]
  KVPool:
    block_3: [token_0,  ..., token_15]
    block_7: [token_16, ..., token_31]
    block_2: [token_32, ..., token_47]
    ...
  → 直接按 block_table 索引访问，无需拼接
```

#### 4.2 Kernel 签名

```python
@triton.jit
def _paged_attn_decode_kernel(
    Out,            # [B, num_heads, head_dim]
    Q,              # [B, num_heads, head_dim]
    K_cache,        # [num_blocks, block_size, num_kv_heads, head_dim]
    V_cache,
    Block_tables,   # [B, max_num_blocks_per_seq]  int32
    Seq_lens,       # [B]  int32
    scale,          # float scalar (= 1 / sqrt(head_dim))，已预乘到 Q 上
    # stride 参数 (Out, Q, K_cache, V_cache, Block_tables 各维度)
    BLOCK_SIZE:    tl.constexpr,
    HEAD_DIM:      tl.constexpr,
    NUM_KV_GROUPS: tl.constexpr,   # = num_heads // num_kv_heads (GQA)
    MAX_NUM_BLOCKS: tl.constexpr,
):
```

Grid：`(batch_size, num_heads)`，每个 program 处理一个 `(batch, query_head)` 对。

#### 4.3 Online Softmax 算法

逐 block 迭代，一次遍历完成 softmax，避免二次遍历：

```
初始化: m_i = -∞, l_i = 0, acc = zeros[HEAD_DIM]

对于每个 block (block_idx = 0 .. MAX_NUM_BLOCKS-1):
    phys_block = Block_tables[bid, block_idx]
    K = KVPool.k[phys_block, :, kv_hid, :]   # [BLOCK_SIZE, HEAD_DIM]
    V = KVPool.v[phys_block, :, kv_hid, :]

    s = q @ K^T                                # [BLOCK_SIZE]，无效位置填 -inf
    m_block = max(s)
    m_new   = max(m_i, m_block)

    alpha = exp(m_i - m_new)                   # 旧状态衰减系数
    p     = exp(s  - m_new)                    # 当前 block softmax 权重（未归一化）

    l_i = l_i * alpha + sum(p)
    acc = acc * alpha + sum(p[:, None] * V, axis=0)
    m_i = m_new

输出: acc / max(l_i, 1e-10)
```

#### 4.4 GQA 支持

```python
kv_hid = hid // NUM_KV_GROUPS   # 多个 Query Head 映射到同一 KV Head
# 例：num_heads=32, num_kv_heads=8, NUM_KV_GROUPS=4
# Query Head 0-3  → KV Head 0
# Query Head 4-7  → KV Head 1  …
```

#### 4.5 Python 接口

```python
def paged_attention_decode(
    query:        Tensor,   # [B, num_heads, 1, head_dim]
    k_cache:      Tensor,   # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache:      Tensor,
    block_tables: Tensor,   # [B, max_blocks_per_seq]  int32
    seq_lens:     Tensor,   # [B]  int32
    scale:        float,
    num_kv_groups: int,
    block_size:   int,
) -> Tensor:                # [B, num_heads, 1, head_dim]
```

- Triton 可用且输入在 CUDA 上 → 走 GPU Kernel
- 否则回退到 `_paged_attention_decode_pytorch`：逐序列 gather K/V 后调用 `F.scaled_dot_product_attention`，GQA 通过 `repeat_interleave` 扩展

---

### 五、推理引擎 (`inference/engine.py`)

#### 5.1 融合算子

引擎在加载时自动探测 Triton 可用性（`_probe_triton`：编译并运行一个简单 kernel 验证结果），然后 monkey-patch 模型中的算子：

| 算子 | Triton 实现 | PyTorch 回退 |
|---|---|---|
| RMSNorm | `_rms_norm_kernel` | `x * rsqrt(mean(x²) + eps) * weight` |
| SwiGLU | `_silu_mul_kernel` | `F.silu(gate) * up` |

`apply_model_optimizations` 扫描所有模块，匹配 `"RMSNorm"` 类名和含 `gate_proj` 的 `"MLP"` 类名，替换 `forward`。

#### 5.2 InferenceEngine 构造参数

```python
engine = InferenceEngine(
    model_path: str,                      # 本地模型路径
    cache_type: str = "continuous",       # "continuous" | "paged"
    block_size: int = 16,                 # Paged: 每 block 的 token 容量
    num_blocks: int = 1000,               # Paged: 预分配 block 总数
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float16,
    enable_optimizations: bool = True,    # 是否启用 RMSNorm/SwiGLU 融合
    batch_size: int = 32,                 # infer_batch 的分批大小
)
```

模型加载策略（依次尝试，选择第一个成功的）：
1. `flash_attention_2`
2. `sdpa`
3. `eager`
4. 默认（无 `attn_implementation` 参数）

加载完成后执行 3 次预热推理（`"hello world"`），确保 CUDA kernel 编译完毕，避免首次推理延迟干扰测量。

#### 5.3 Attention Monkey-Patch（Paged 模式）

`_patch_attention_for_paged` 扫描所有类名含 `"Attention"`、具有 `q_proj` 和 `layer_idx` 属性的模块，替换其 `forward`。

Patched forward 的路由逻辑：

```python
if not isinstance(past_key_value, PagedCacheAdapter):
    return original_forward(...)    # Prefill / Continuous 路径（HF 原生）

# Paged Decode 路径：
# 1. Q/K/V 投影
# 2. RoPE（支持 position_embeddings 传参 或 rotary_emb 模块，可选）
# 3. adapter.write_kv(layer_idx, key_states, value_states)
# 4. paged_attention_decode(query, k_cache, v_cache, block_tables, seq_lens, ...)
# 5. o_proj
```

Qwen2 等将 `rotary_emb` 放在模型顶层的架构，由 `_patch_attention_for_paged` 从 `model.model` 读取并通过闭包传入。

#### 5.4 推理流水线

**Continuous 模式** (`_batch_decode`)：

```
Prefill(step=0): model.forward(input_ids, attention_mask, past_key_values=None)
                   → 得到 logits + past_key_values
Decode loop:     model.forward(cur_ids=[1 token], attention_mask, past_key_values=past)
                   → 贪心采样，EOS 检测，unfinished mask 控制
```

**Paged 模式** (`_batch_decode_paged`)：

```
1. cache.reset()，为每个序列 allocate_seq

2. Prefill: model.forward(input_ids, attention_mask)  ← 标准 HF forward（含 flash attention）
   TTFT 在此步结束后打点

3. 提取 prefill KV → KVPool:
   for layer in range(num_layers):
       k_full[i, :, -real_len:, :].permute(1,0,2)  # left-padding 对齐：取右侧真实 token
       cache.append(seq_id, layer, k_seq, v_seq)

4. Decode loop:
   adapter.begin_step(unfinished_mask)
   model.forward(cur_ids, position_ids, past_key_values=adapter)
   adapter.end_step()
   # position_ids = seq_lens_tensor - 1（新 token 的 0-based 位置）
```

**Left-padding 对齐**：Tokenizer 采用 `padding_side="left"`，`past_key_values` 中真实 token 的 KV 位于 `k_full[i, :, -real_len:, :]`，提取时需裁剪对齐。

**EOS 检测**：除 `tokenizer.eos_token_id` 外，还额外检测 `<|im_end|>`, `<|end|>`, `</s>`，兼容 Qwen / Llama / Mistral 等多种模型。

#### 5.5 批量推理接口

```python
# 单条推理
result: InferenceResult = engine.infer_single(prompt, max_new_tokens=256)

# 批量推理
results: List[InferenceResult] = engine.infer_batch(
    prompts,
    max_new_tokens=256,
    show_progress=False,   # True 时打印每批进度
)
```

`infer_batch` 内部：按 token 长度排序 prompts（减少批内 padding），然后按 `batch_size` 分批调用 `_batch_decode` 或 `_batch_decode_paged`，最终按原始顺序返回结果。

#### 5.6 InferenceResult

```python
@dataclass
class InferenceResult:
    prompt: str              # 输入 prompt
    output: str              # 解码后的生成文本
    input_tokens: int        # 输入 token 数（去除 padding）
    output_tokens: int       # 实际生成 token 数（到 EOS 为止）
    total_latency_ms: float  # 整批总延迟 (ms)
    ttft_ms: float           # Time To First Token (ms)，Prefill 耗时
    throughput_tps: float    # tokens/sec = output_tokens / total_latency
```

---

## 评测工具

### 对比评测 (`evaluation/compare_cache.py`)

对比 `continuous` 和 `paged` 两种模式的性能，支持从 JSONL 文件批量加载 prompt：

```bash
python -m llm_inference.evaluation.compare_cache \
    --model_path /path/to/model \
    --prompt_file prompts.jsonl \
    --batch_size 8 \
    --max_new_tokens 256 \
    --output compare_results.json   # 可选，保存为 JSON
```

输出指标：

| 指标 | 说明 |
|---|---|
| `overall_throughput_tps` | 总输出吞吐（tokens/sec）|
| `avg_latency_ms` | 平均批次延迟 |
| `p50/p95/p99_latency_ms` | 延迟百分位 |
| `avg_ttft_ms / p95_ttft_ms` | TTFT 统计 |
| `cache_memory_gb` | Cache 显存占用（固定分配量）|
| `peak_memory_gb` | CUDA 峰值显存 |

对比打印示例：
```
  吞吐量 (tokens/sec)         : 312.50 vs 489.20 (+56.5%)
  平均延迟 (ms)               : 1024.3 vs 654.8  (+36.1%)
  Cache 显存 (GB)             : 0.050  vs 4.096  (-8092.0%)
```

### 基准测试 (`benchmark_llm_inference.py`)

位于仓库根目录，独立的吞吐量 & 延迟 benchmark 脚本：

```bash
# continuous 模式
python benchmark_llm_inference.py \
    --model_path /path/to/model \
    --cache_type continuous \
    --batch_size 32 \
    --max_new_tokens 256 \
    --output results_continuous.json

# paged 模式
python benchmark_llm_inference.py \
    --model_path /path/to/model \
    --cache_type paged \
    --block_size 16 \
    --num_blocks 1000 \
    --batch_size 32 \
    --max_new_tokens 256 \
    --output results_paged.json
```

全部命令行参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model_path` | 必填 | 模型本地路径 |
| `--cache_type` | `continuous` | `continuous` 或 `paged` |
| `--prompt_file` | `prompts.jsonl` | JSONL 格式 prompt 文件 |
| `--batch_size` | `32` | 推理批量大小 |
| `--max_new_tokens` | `256` | 最大生成 token 数 |
| `--block_size` | `16` | Paged Block 大小（仅 paged 有效）|
| `--num_blocks` | `1000` | Paged Block 总数（仅 paged 有效）|
| `--output` | `None` | 结果保存路径（JSON）|

输出统计指标：

| 指标 | 说明 |
|---|---|
| `overall_throughput_tps` | 整体输出吞吐（tokens/sec）|
| `input_throughput_tps` | 输入 token 处理吞吐 |
| `avg/p50/p95/p99_latency_ms` | 延迟百分位统计 |
| `avg_ttft_ms / p95_ttft_ms` | TTFT 统计 |
| `peak_gpu_mem_gb` | CUDA `max_memory_allocated` |
| `cache_memory_gb` | Cache 本身的显存占用 |

**Prompt 文件格式** (`prompts.jsonl`)：每行一个 JSON 对象：

```jsonl
{"id": 1, "prompt": "请解释什么是大语言模型。"}
{"id": 2, "prompt": "用 Python 实现快速排序。"}
```

---

## 使用方法

### 命令行 (`cli.py`)

```bash
# 连续 Cache 模式
python -m llm_inference.cli \
    --model_path /path/to/model \
    --prompt "请解释什么是大语言模型。" \
    --cache_type continuous \
    --max_new_tokens 256

# 分页 Cache 模式 (PagedAttention)
python -m llm_inference.cli \
    --model_path /path/to/model \
    --prompt "请解释什么是大语言模型。" \
    --cache_type paged \
    --max_new_tokens 256
```

CLI 输出包含：输入/输出文本、token 数、总延迟、TTFT、吞吐量、Cache 显存占用。

### Python API

```python
from llm_inference import InferenceEngine

# 创建推理引擎（paged 模式）
engine = InferenceEngine(
    model_path="/path/to/model",
    cache_type="paged",
    block_size=16,
    num_blocks=1000,
    device="cuda:0",
    dtype=torch.float16,
    enable_optimizations=True,
    batch_size=32,
)

# 单序列推理
result = engine.infer_single("什么是大语言模型？", max_new_tokens=256)
print(result.output)
print(f"TTFT: {result.ttft_ms:.1f} ms | 吞吐: {result.throughput_tps:.1f} tokens/sec")

# 批量推理
results = engine.infer_batch(
    prompts=["问题1", "问题2", "问题3"],
    max_new_tokens=256,
    show_progress=True,
)

# 查询 Cache 显存占用
print(f"Cache 显存: {engine.get_cache_memory_usage() / 1e9:.2f} GB")
```

### 直接使用 PagedKVCache

```python
from llm_inference import PagedKVCache, paged_attention_decode

cache = PagedKVCache(
    num_layers=32, num_kv_heads=8, head_dim=128,
    block_size=16, num_blocks=1000,
)
cache.allocate_seq(seq_id=0)
cache.append(0, layer_idx=0,
             k=torch.randn(20, 8, 128, device="cuda"),
             v=torch.randn(20, 8, 128, device="cuda"))

k_cache, v_cache = cache.get_kv_cache(layer_idx=0)
# → k_cache: [1000, 16, 8, 128]
```

---

## 公共 API (`__init__.py`)

```python
from llm_inference import (
    # 连续 Cache
    ContinuousKVCache,
    # 分页 Cache
    KVPool, PagedKVCache, PagedCacheAdapter,
    # Attention Kernel
    paged_attention_decode,
    # Block Manager
    BlockAllocator, BlockTable,
    # 推理引擎
    InferenceEngine, InferenceResult,
)
```

---

## 依赖

```
torch >= 2.0
transformers >= 4.30
numpy
triton >= 2.0    # 可选，用于 RMSNorm/SwiGLU/PagedAttention 加速 kernel
                 # 不可用时自动回退到 PyTorch 实现
```

---

## 参考

本实现参考了以下论文和项目：

- [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) — Paged KV-Cache 与 PagedAttention Kernel 的核心技术来源
- [GQA (Grouped Query Attention)](https://arxiv.org/abs/2305.13245) — 减少 KV Cache 显存占用
- [Flash Attention](https://arxiv.org/abs/2205.14135) — Prefill 阶段的高效 Attention 实现
- [Online Softmax](https://arxiv.org/abs/1805.02867) — 单次遍历完成 softmax 的算法基础
- [Triton](https://triton-lang.org) — GPU Kernel 编程框架
