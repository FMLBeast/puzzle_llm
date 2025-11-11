# Local Training Guide

Complete guide for training and benchmarking puzzle-solving LLMs on your local machine.

## Prerequisites

### Hardware Requirements

**Minimum (CPU only - NOT recommended)**:
- 16GB RAM
- 100GB disk space
- Training time: Several days per model

**Recommended (with GPU)**:
- NVIDIA GPU with 24GB+ VRAM (RTX 3090, RTX 4090, or better)
- 32GB+ system RAM
- 100GB disk space
- Training time: 4-8 hours per model

**Optimal (for both models)**:
- NVIDIA A100 (40GB) or H100
- 64GB system RAM
- 200GB SSD storage
- Training time: 2-4 hours per model

### Software Requirements

1. **Python 3.11+**
2. **CUDA 11.8+** (for GPU training)
3. **Git**

### Python Dependencies

Install required packages:

```bash
pip install torch>=2.0.0 transformers>=4.30.0 datasets>=2.14.0 \
    accelerate>=0.20.0 bitsandbytes>=0.41.0 peft>=0.4.0 \
    trl>=0.7.0 pandas numpy tqdm sentencepiece protobuf
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### HuggingFace Setup

1. Create account at [huggingface.co](https://huggingface.co)
2. Request access to Meta Llama models (for benchmark)
3. Create access token
4. Set environment variable:

```bash
export HF_TOKEN=your_token_here
```

## Quick Start (Automated)

Run the complete pipeline with one command:

```bash
./run_benchmark.sh
```

This script will:
1. ✅ Prepare training data from cryptopuzzles repo
2. ✅ Train primary model (Mistral-7B)
3. ✅ Train benchmark model (Llama-3.2-3B)
4. ✅ Test both models
5. ✅ Generate comparison report

**Note**: This will take 4-16 hours depending on your GPU.

## Manual Step-by-Step

### Step 1: Prepare Training Data

Extract and process data from the FMLBeast/cryptopuzzles repository:

```bash
python prepare_training_data_local.py --output-dir ./training_data
```

This creates:
- `training_data/train.json` - Training examples
- `training_data/validation.json` - Validation examples
- `training_data/test.json` - Test examples
- `training_data/stats.json` - Dataset statistics

### Step 2: Train Primary Model

Train the Mistral-7B based model:

```bash
python train_local.py \
    --config primary \
    --data-dir ./training_data \
    --output-dir ./models/primary
```

**Configuration**:
- Model: Mistral-7B-v0.1
- Epochs: 5
- Learning rate: 2e-4
- LoRA r/alpha: 16/32
- Batch size: 2 (effective: 8 with accumulation)

**Expected time**: 2-4 hours on RTX 4090

### Step 3: Train Benchmark Model

Train the Llama-3.2 based model:

```bash
python train_local.py \
    --config benchmark \
    --data-dir ./training_data \
    --output-dir ./models/benchmark
```

**Configuration**:
- Model: Llama-3.2-3B-Instruct
- Epochs: 8
- Learning rate: 3e-4
- LoRA r/alpha: 32/64
- Batch size: 4 (effective: 8 with accumulation)

**Expected time**: 3-5 hours on RTX 4090

### Step 4: Test Models

#### Quick Test

Test with a single prompt:

```bash
python test_local.py \
    --model-dir ./models/primary/final \
    --prompt "### Instruction:\nSolve: KHOOR ZRUOG\n\n### Response:\n"
```

#### Interactive Mode

Chat with the model:

```bash
python test_local.py \
    --model-dir ./models/primary/final \
    --interactive
```

#### Batch Testing

Test with default prompts:

```bash
python test_local.py \
    --model-dir ./models/primary/final \
    --batch
```

### Step 5: Compare Models

Run comprehensive evaluation:

```bash
python evaluate_models.py \
    --primary-model ./models/primary/final \
    --benchmark-model ./models/benchmark/final \
    --test-data ./training_data/test.json \
    --output ./results/evaluation.json
```

This generates:
- Side-by-side response comparison
- Inference speed metrics
- Response quality analysis
- Detailed JSON results

## Configuration Comparison

View differences between model configs:

```bash
python compare_configs.py
```

Output:
```
Parameter                         Primary                      Benchmark
--------------------------------------------------------------------------------------
model_name                        mistralai/Mistral-7B-v0.1   meta-llama/Llama-3.2-3B-Instruct ⚡
num_train_epochs                  5                            8                                  ⚡
learning_rate                     2.0e-04                      3.0e-04                            ⚡
lora_r                            16                           32                                 ⚡
lora_alpha                        32                           64                                 ⚡
...
```

## Troubleshooting

### Out of Memory (OOM) Errors

If you get CUDA OOM errors:

1. **Reduce batch size**:
   ```python
   # Edit train_local.py CONFIGS
   "per_device_train_batch_size": 1,  # Reduce from 2 or 4
   "gradient_accumulation_steps": 8,  # Increase to maintain effective batch size
   ```

2. **Use 4-bit quantization** (default):
   ```bash
   python train_local.py --config primary ...  # Already uses quantization
   ```

3. **Use smaller model**:
   - Try Llama-3.2-1B instead of 3B
   - Try smaller Mistral variants

### Slow Training on CPU

If you don't have a GPU, consider:

1. **Use Modal.com** (cloud GPU):
   ```bash
   modal run train_llm_modal.py::main --command train
   ```

2. **Use Google Colab** (free GPU):
   - Upload scripts to Colab
   - Use their T4 GPU (free tier)

3. **Reduce dataset size**:
   ```bash
   # Edit prepare_training_data_local.py
   # Limit number of examples processed
   ```

### Model Loading Errors

If you get "model not found" errors:

1. **Check HF token**:
   ```bash
   echo $HF_TOKEN  # Should show your token
   ```

2. **Verify access**:
   - Check HuggingFace account has Llama access
   - Wait 1-2 hours after requesting access

3. **Use alternative model**:
   ```python
   # In train_local.py, change benchmark config
   "model_name": "meta-llama/Llama-2-7b-hf",  # Try Llama 2 instead
   ```

## Cost Estimates

### Local GPU Training

| GPU | Power | Hours | Cost/kWh | Total Cost |
|-----|-------|-------|----------|------------|
| RTX 4090 | 450W | 8 hrs | $0.12 | ~$0.43 |
| RTX 3090 | 350W | 12 hrs | $0.12 | ~$0.50 |
| A100 (40GB) | 400W | 4 hrs | $0.12 | ~$0.19 |

### Cloud Training (Modal.com)

| Resource | Hours | Rate | Total |
|----------|-------|------|-------|
| A100 GPU | 6 hrs | $3.50/hr | ~$21 |
| Storage | 1 month | $0.10/GB | ~$0.60 |
| **Total** | - | - | **~$22** |

**Recommendation**: Use local if you have a 24GB+ GPU. Otherwise, use Modal.

## Advanced Options

### Custom Hyperparameters

Create your own config in `train_local.py`:

```python
CONFIGS = {
    # ... existing configs ...
    "custom": {
        "name": "Custom Model",
        "model_name": "your-model-name",
        "num_train_epochs": 10,
        "learning_rate": 1e-4,
        # ... other params ...
    }
}
```

Then train:

```bash
python train_local.py --config custom --data-dir ./training_data --output-dir ./models/custom
```

### Weights & Biases Logging

Enable W&B for advanced tracking:

```bash
# Install wandb
pip install wandb
wandb login

# Train with W&B
python train_local.py \
    --config primary \
    --data-dir ./training_data \
    --output-dir ./models/primary \
    --use-wandb
```

### Multi-GPU Training

For multiple GPUs, use DeepSpeed or FSDP:

```bash
# Install DeepSpeed
pip install deepspeed

# Create deepspeed config (deepspeed_config.json)
# Run with accelerate
accelerate config
accelerate launch train_local.py --config primary ...
```

## Evaluation Metrics

### What to Compare

1. **Accuracy**: Does the model solve puzzles correctly?
2. **Reasoning**: Does it explain the solution process?
3. **Speed**: How fast is inference?
4. **Generalization**: Does it work on unseen puzzle types?

### Interpreting Results

**Primary wins if**:
- Lower validation loss
- Better reasoning in responses
- More accurate puzzle solutions

**Benchmark wins if**:
- Faster inference
- More concise responses
- Better generalization

### Creating Custom Tests

Add puzzles to test set:

```python
# Create custom_tests.json
[
    {
        "text": "### Instruction:\nSolve: YOUR PUZZLE\n\n### Response:\n",
        "source": "custom",
        "difficulty": "hard"
    },
    # ... more puzzles
]
```

Evaluate:

```bash
python evaluate_models.py \
    --primary-model ./models/primary/final \
    --benchmark-model ./models/benchmark/final \
    --test-data custom_tests.json
```

## Next Steps

1. **Analyze results**: Which model performs better?
2. **Fine-tune further**: Use insights to improve training
3. **Expand dataset**: Add more puzzle types
4. **Deploy**: Serve the best model via API
5. **Share findings**: Document what worked best

## Support

For issues:
- Check this guide's troubleshooting section
- Review Modal guide in `BENCHMARK_README.md`
- Check HuggingFace docs for model-specific issues

## File Reference

```
puzzle_llm/
├── prepare_training_data_local.py  # Data preparation
├── train_local.py                  # Local training script
├── test_local.py                   # Testing/inference
├── evaluate_models.py              # Model comparison
├── compare_configs.py              # Config comparison
├── run_benchmark.sh                # Automated pipeline
├── LOCAL_TRAINING_GUIDE.md         # This file
└── BENCHMARK_README.md             # Modal.com guide
```

## License

Same as main project.
