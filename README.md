# LLM 推理优化实训

> 课题：面向大语言模型高效推理框架的系统级优化与实现  
> 联系人：石亮（[lshi@cs.ecnu.edu.cn](mailto:lshi@cs.ecnu.edu.cn)）

---

## 目录结构

```
baseline/
├── baseline_inference.py   # 朴素推理实现
├── benchmark.py            # 吞吐量 & 延迟基准测试
├── evaluate_accuracy.py    # 精度评测脚本
├── prompts.jsonl           # 统一测试 prompt 集合
├── ceval_subset.jsonl      # 精度评测数据集（样例，实际测评还有更多其他样本）
├── requirements.txt        # 依赖列表
└── README.md               # 本文件
```

---

## 环境要求

```
单卡 H100
模型：Qwen2.5-14B-Instruct及其量化版本
```

### 3. 验证推理链路

```bash
python baseline_inference.py --model_path $MODEL_PATH
```

预期输出（数值仅供参考）：

```
[INFO] 加载完成 | 参数量: 7.62B | 显存占用: 14.36 GB
  总延迟  : 3842.10 ms
  吞吐率  : 28.4 tokens/sec
  峰值显存: 15.021 GB
```

### 4. 运行吞吐 & 延迟基准测试

```bash
python benchmark.py --model_path $MODEL_PATH --output results_baseline.json
```

`results_baseline.json` 即为你的**性能基线**，优化后需与此文件对比。

### 5. 运行精度评测

```bash
python evaluate_accuracy.py --model_path $MODEL_PATH --eval_file ceval_subset.jsonl --output accuracy_baseline.json
```

返回得到模型在测评集上的精度，作为后续优化的参考。后续优化不得导致精度过度丢失。(ceval_subset.jsonl中样例较多，考虑到运行时间可以截选其中200~500条进行测试,保证对比前后使用的数据一致即可)

---

## 评分指标

所有指标以**单卡 cuda:0** 结果为准。


| 指标                       | 单位         | 方向    | 说明                      |
| ------------------------ | ---------- | ----- | ----------------------- |
| `overall_throughput_tps` | tokens/sec | 越高越好  | prompt总输出/总耗时           |
| `p95_latency_ms`         | ms         | 越低越好  | 95分位延迟                  |
| `avg_latency_ms`         | ms         | 越低越好  | 单条请求平均端到端延迟             |
| `average_ttft_ms`                | ms         | 越低越好  | 首token延迟                |
| `accuracy`               | %          | 损失≤5% | C-Eval精度.            |
| `peak_gpu_mem_gb`        | GB         | 越低越好  | 推理峰值显存占用                |


---

## 代码提交要求

1. 优化后的代码（可基于本脚本修改，或新建目录）。
2. README.md（简介项目并说明运行方法）
3. 关于流式输出可以单开一个简单的脚本进行简单的功能展示即可。
4. `results_baseline.json` 与 `results_optimized.json` (基准和优化后的吞吐、延迟等)。
5. `accuracy_baseline.json` 与 `accuracy_optimized.json`  (基准和优化后的精度)。
6. 实验报告（技术路线/原理 + 实验对比 + 分析）。

---

## 参考资料

- [vLLM 官方文档](https://github.com/vllm-project/vllm)
- [PagedAttention 论文](https://arxiv.org/pdf/2309.06180)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [GPTQ 量化工具](https://github.com/qwopqwop200/GPTQ-for-LLaMa)
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [LMCache KV Cache 引擎](https://github.com/LMCache/LMCache)

