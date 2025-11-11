#!/bin/bash
# Complete benchmark pipeline - run everything locally

set -e  # Exit on error

echo "================================================================================"
echo "🚀 PUZZLE LLM BENCHMARK TRAINING PIPELINE"
echo "================================================================================"
echo ""

# Configuration
DATA_DIR="./training_data"
PRIMARY_MODEL_DIR="./models/primary"
BENCHMARK_MODEL_DIR="./models/benchmark"
RESULTS_DIR="./results"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  Warning: HF_TOKEN environment variable not set"
    echo "   Some models may require authentication."
    echo "   Set it with: export HF_TOKEN=your_token_here"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 1: Prepare training data
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}📚 Step 1/5: Preparing Training Data${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

if [ -d "$DATA_DIR" ] && [ -f "$DATA_DIR/train.json" ]; then
    echo "✅ Training data already exists at $DATA_DIR"
    read -p "Regenerate data? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python prepare_training_data_local.py --output-dir "$DATA_DIR"
    fi
else
    python prepare_training_data_local.py --output-dir "$DATA_DIR"
fi

echo ""
echo "✅ Data preparation complete!"
echo ""

# Step 2: Train primary model
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}🤖 Step 2/5: Training Primary Model (Mistral-7B)${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

if [ -d "$PRIMARY_MODEL_DIR/final" ]; then
    echo "✅ Primary model already exists at $PRIMARY_MODEL_DIR/final"
    read -p "Retrain? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python train_local.py \
            --config primary \
            --data-dir "$DATA_DIR" \
            --output-dir "$PRIMARY_MODEL_DIR"
    fi
else
    python train_local.py \
        --config primary \
        --data-dir "$DATA_DIR" \
        --output-dir "$PRIMARY_MODEL_DIR"
fi

echo ""
echo "✅ Primary model training complete!"
echo ""

# Step 3: Train benchmark model
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}🤖 Step 3/5: Training Benchmark Model (Llama-3.2-3B)${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

if [ -d "$BENCHMARK_MODEL_DIR/final" ]; then
    echo "✅ Benchmark model already exists at $BENCHMARK_MODEL_DIR/final"
    read -p "Retrain? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python train_local.py \
            --config benchmark \
            --data-dir "$DATA_DIR" \
            --output-dir "$BENCHMARK_MODEL_DIR"
    fi
else
    python train_local.py \
        --config benchmark \
        --data-dir "$DATA_DIR" \
        --output-dir "$BENCHMARK_MODEL_DIR"
fi

echo ""
echo "✅ Benchmark model training complete!"
echo ""

# Step 4: Quick test both models
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}🧪 Step 4/5: Quick Testing Both Models${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

TEST_PROMPT="### Instruction:\nSolve this Caesar cipher:\n\nKHOOR ZRUOG\n\nHint: Try shift values 1-25\n\n### Response:\n"

echo -e "${GREEN}Testing Primary Model:${NC}"
python test_local.py \
    --model-dir "$PRIMARY_MODEL_DIR/final" \
    --prompt "$TEST_PROMPT"

echo ""
echo -e "${GREEN}Testing Benchmark Model:${NC}"
python test_local.py \
    --model-dir "$BENCHMARK_MODEL_DIR/final" \
    --prompt "$TEST_PROMPT"

echo ""

# Step 5: Full evaluation
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}📊 Step 5/5: Running Full Evaluation${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

mkdir -p "$RESULTS_DIR"

python evaluate_models.py \
    --primary-model "$PRIMARY_MODEL_DIR/final" \
    --benchmark-model "$BENCHMARK_MODEL_DIR/final" \
    --test-data "$DATA_DIR/test.json" \
    --max-samples 20 \
    --output "$RESULTS_DIR/evaluation_$(date +%Y%m%d_%H%M%S).json"

echo ""
echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}🎉 BENCHMARK PIPELINE COMPLETE!${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""
echo "📁 Results saved in: $RESULTS_DIR"
echo ""
echo "Next steps:"
echo "  1. Review evaluation results in $RESULTS_DIR"
echo "  2. Test models interactively:"
echo "     python test_local.py --model-dir $PRIMARY_MODEL_DIR/final --interactive"
echo "  3. Run additional evaluations with more test samples"
echo ""
