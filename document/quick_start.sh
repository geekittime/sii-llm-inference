#!/bin/bash
# quick_start.sh - 一键启动脚本（改进版）

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE} LLM推理优化工具 v2 (PagedAttention)${NC}"
echo -e "${BLUE}========================================${NC}\n"

if [ -z "$1" ]; then
    echo -e "${RED}错误: 未指定模型路径${NC}"
    echo "用法: bash quick_start.sh <model_path> [batch_size]"
    exit 1
fi

MODEL_PATH=$1
BATCH_SIZE=${2:-32}

echo -e "${BLUE}配置${NC}:"
echo "  模型: $MODEL_PATH"
echo "  Batch: $BATCH_SIZE"
echo ""

if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}错误: 模型路径不存在${NC}"
    exit 1
fi

# Step 1: 验证推理
echo -e "\n${GREEN}[步骤1]${NC} 验证推理..."
python optimized_inference_v2.py \
    --model_path "$MODEL_PATH" \
    --prompt "KV Cache是什么？"

# Step 2: Baseline基准
if [ -f "baseline_inference.py" ]; then
    echo -e "\n${GREEN}[步骤2]${NC} Baseline基准测试..."
    python benchmark.py \
        --model_path "$MODEL_PATH" \
        --batch_size "$BATCH_SIZE" \
        --output results_baseline.json || true
else
    echo -e "\n${YELLOW}[步骤2]${NC} 跳过baseline"
fi

# Step 3: 优化版本基准
echo -e "\n${GREEN}[步骤3]${NC} 优化版本基准测试..."
python benchmark_v2.py \
    --model_path "$MODEL_PATH" \
    --batch_size "$BATCH_SIZE" \
    --output results_optimized.json

# Step 4: Baseline精度
if [ -f "ceval_subset.jsonl" ] && [ -f "evaluate_accuracy.py" ]; then
    echo -e "\n${GREEN}[步骤4]${NC} Baseline精度评测..."
    python evaluate_accuracy.py \
        --model_path "$MODEL_PATH" \
        --eval_file ceval_subset.jsonl \
        --batch_size "$BATCH_SIZE" \
        --output accuracy_baseline.json || true
fi

# Step 5: 优化版本精度
if [ -f "ceval_subset.jsonl" ]; then
    echo -e "\n${GREEN}[步骤5]${NC} 优化版本精度评测..."
    python evaluate_accuracy_v2.py \
        --model_path "$MODEL_PATH" \
        --eval_file ceval_subset.jsonl \
        --batch_size "$BATCH_SIZE" \
        --output accuracy_optimized.json
fi

# Step 6: 流式演示
echo -e "\n${GREEN}[步骤6]${NC} 流式输出演示..."
timeout 30s python streaming_inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "请简要介绍PagedAttention。" || true

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✓ 完成！${NC}"
echo -e "${GREEN}========================================${NC}\n"

echo "生成的文件:"
ls -lh results_*.json accuracy_*.json 2>/dev/null || echo "  (运行中生成)"
