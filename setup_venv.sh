#!/bin/bash

# Setup script for puzzle_llm with virtual environment

set -e  # Exit on error

echo "=========================================="
echo "🔧 PUZZLE LLM - VIRTUAL ENVIRONMENT SETUP"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${BLUE}[1/5]${NC} Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Error: Python 3 is not installed${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
echo ""

# Create virtual environment
echo -e "${BLUE}[2/5]${NC} Creating virtual environment..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Removing...${NC}"
    rm -rf venv
fi

python3 -m venv venv
echo -e "${GREEN}✓ Virtual environment created${NC}"
echo ""

# Activate virtual environment
echo -e "${BLUE}[3/5]${NC} Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Upgrade pip
echo -e "${BLUE}[4/5]${NC} Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo -e "${GREEN}✓ pip upgraded${NC}"
echo ""

# Install dependencies
echo -e "${BLUE}[5/5]${NC} Installing dependencies..."
echo "This may take a few minutes..."
pip install modal datasets pandas numpy tqdm > /dev/null 2>&1
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Test Modal installation
echo "Testing Modal installation..."
python -c "import modal; print(f'Modal version: {modal.__version__}')"
echo ""

echo "=========================================="
echo "✨ SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "To activate the virtual environment, run:"
echo -e "${GREEN}source venv/bin/activate${NC}"
echo ""
echo "Then authenticate Modal:"
echo -e "${GREEN}modal token set --token-id wk-EwN50NyAMm8GhEKD4Ryf7L --token-secret ws-nfLvqkVMwfE21hNCSMr4LD${NC}"
echo ""
echo "Finally, run the training:"
echo -e "${GREEN}modal run train_llm_modal.py --command upload --data-path processed_data${NC}"
echo -e "${GREEN}modal run train_llm_modal.py --command train${NC}"
echo ""
