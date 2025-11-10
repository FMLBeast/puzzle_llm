#!/bin/bash

# Puzzle LLM Training - Quick Start Script
# This script sets up and runs the full training pipeline

set -e  # Exit on error

echo "=================================="
echo "🎯 PUZZLE LLM TRAINING QUICKSTART"
echo "=================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Modal is installed
echo -e "${BLUE}[1/6]${NC} Checking Modal installation..."
if ! python -c "import modal" 2>/dev/null; then
    echo -e "${YELLOW}Modal not installed. Installing...${NC}"
    pip install modal
fi
echo -e "${GREEN}✓ Modal installed${NC}"
echo ""

# Check Modal authentication
echo -e "${BLUE}[2/6]${NC} Checking Modal authentication..."
if ! modal token get > /dev/null 2>&1; then
    echo -e "${YELLOW}Modal not authenticated.${NC}"
    echo "Please run one of:"
    echo "  1. python3 -m modal setup"
    echo "  2. modal token set --token-id <id> --token-secret <secret>"
    exit 1
fi
echo -e "${GREEN}✓ Modal authenticated${NC}"
echo ""

# Check if data is already processed
echo -e "${BLUE}[3/6]${NC} Checking for processed data..."
if [ ! -d "processed_data" ]; then
    echo -e "${YELLOW}No processed data found. Processing cryptopuzzles dataset...${NC}"

    # Check if cryptopuzzles data exists
    if [ ! -d "cryptopuzzles_data" ]; then
        echo "Cloning cryptopuzzles repository..."
        git clone https://github.com/FMLBeast/cryptopuzzles.git cryptopuzzles_data
    fi

    # Install required packages
    echo "Installing required packages..."
    pip install -q datasets pandas

    # Process data
    python process_cryptopuzzles.py
fi
echo -e "${GREEN}✓ Data processed (28 examples)${NC}"
echo ""

# Upload data to Modal
echo -e "${BLUE}[4/6]${NC} Uploading data to Modal..."
echo "This will upload the training data to Modal's persistent volume..."
modal run train_llm_modal.py --command upload --data-path processed_data
echo -e "${GREEN}✓ Data uploaded to Modal${NC}"
echo ""

# Start training
echo -e "${BLUE}[5/6]${NC} Starting training..."
echo ""
echo "Training configuration:"
echo "  - Model: meta-llama/Llama-3.2-1B"
echo "  - GPU: A10G"
echo "  - Technique: QLoRA fine-tuning"
echo "  - Dataset: 22 training examples"
echo "  - Estimated time: 10-30 minutes"
echo "  - Estimated cost: $0.50-$2.00"
echo ""
read -p "Continue with training? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    modal run train_llm_modal.py --command train
    echo -e "${GREEN}✓ Training complete${NC}"
else
    echo "Training cancelled. You can start it later with:"
    echo "  modal run train_llm_modal.py --command train"
    exit 0
fi
echo ""

# Test the model
echo -e "${BLUE}[6/6]${NC} Testing the trained model..."
echo ""
read -p "Test the model? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    modal run train_llm_modal.py --command test
    echo -e "${GREEN}✓ Test complete${NC}"
fi
echo ""

echo "=================================="
echo "✨ QUICKSTART COMPLETE!"
echo "=================================="
echo ""
echo "Your puzzle-solving LLM is ready!"
echo ""
echo "Next steps:"
echo "  - Modify config.yaml to adjust training parameters"
echo "  - Add more puzzle data to improve accuracy"
echo "  - Deploy the model for inference"
echo ""
echo "For more information, see README.md"
