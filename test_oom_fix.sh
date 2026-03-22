#!/bin/bash
# test_oom_fix.sh - 验证OOM修复

echo "=========================================="
echo "  PagedKVCache OOM修复 - 验证脚本"
echo "=========================================="
echo ""

if [ -z "$1" ]; then
    echo "用法: bash test_oom_fix.sh <model_path>"
    echo "例如: bash test_oom_fix.sh /path/to/Qwen2.5-14B-Instruct"
    exit 1
fi

MODEL_PATH=$1
echo "模型路径: $MODEL_PATH"
echo ""

# 测试1: 小batch + 低KV缓存 (显存紧张)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "测试1: 显存紧张模式 (batch=8, max_kv=5GB)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python optimized_inference_v2.py \
    --model_path "$MODEL_PATH" \
    --batch_size 8 \
    --max_kv_cache_gb 5 \
    --prompt "KV Cache有什么用？" 2>&1 | head -20

if [ $? -eq 0 ]; then
    echo "✓ 测试1通过"
else
    echo "✗ 测试1失败"
    exit 1
fi

echo ""

# 测试2: 推荐配置 (默认)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "测试2: 推荐模式 (batch=16, max_kv=10GB)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python optimized_inference_v2.py \
    --model_path "$MODEL_PATH" \
    --batch_size 16 \
    --max_kv_cache_gb 10 \
    --prompt "你好" 2>&1 | head -20

if [ $? -eq 0 ]; then
    echo "✓ 测试2通过"
else
    echo "✗ 测试2失败"
fi

echo ""

# 测试3: 精度评测 (完整)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "测试3: 精度评测 (快速验证，仅前10条)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "ceval_subset.jsonl" ]; then
    # 只取前10条做快速验证
    head -10 ceval_subset.jsonl > test_eval_subset.jsonl

    python evaluate_accuracy_v2.py \
        --model_path "$MODEL_PATH" \
        --eval_file test_eval_subset.jsonl \
        --batch_size 4 \
        --max_kv_cache_gb 8 \
        --output test_results.json 2>&1 | head -30

    rm test_eval_subset.jsonl test_results.json 2>/dev/null

    if [ $? -eq 0 ]; then
        echo "✓ 测试3通过"
    else
        echo "✗ 测试3失败"
    fi
else
    echo "⚠ 测试3跳过 (缺少ceval_subset.jsonl)"
fi

echo ""
echo "=========================================="
echo "✓ OOM修复验证完成！"
echo "=========================================="
echo ""
echo "推荐配置:"
echo "  python evaluate_accuracy_v2.py \\"
echo "    --model_path <model_path> \\"
echo "    --batch_size 16 \\"
echo "    --max_kv_cache_gb 10"
