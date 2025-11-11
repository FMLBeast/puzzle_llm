# Puzzle LLM Benchmark Training

This repository contains two training configurations for puzzle-solving LLMs to enable performance benchmarking.

## Model Comparison

| Feature | Primary Model | Benchmark Model |
|---------|--------------|-----------------|
| **Base Model** | Mistral-7B-v0.1 | Llama-3.2-3B-Instruct |
| **Learning Rate** | 2e-4 | 3e-4 |
| **LoRA r/alpha** | 16/32 | 32/64 |
| **Training Epochs** | 5 | 8 |
| **Batch Size** | 2 | 4 |
| **Gradient Accumulation** | 4 | 2 |
| **Dropout** | 0.05 | 0.1 |
| **Output Directory** | `/models/puzzle-llm` | `/models/puzzle-llm-benchmark` |

## Training Data

Both models use the **SAME training data** from the FMLBeast/cryptopuzzles repository for fair comparison:

- Solved puzzle writeups
- Technique documentation
- Cryptocurrency puzzle solutions
- CTF challenges (optional)

The data is processed and stored in a shared Modal volume at `/models/data`.

## Setup Instructions

### Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **HuggingFace Token**: Create a token with read access
3. **Llama Access**: Request access to Meta's Llama 3.2 models at [HuggingFace](https://huggingface.co/meta-llama)

### Install Modal CLI

```bash
pip install modal
modal setup
```

### Configure Secrets

```bash
# Set your HuggingFace token
modal secret create huggingface-secret HF_TOKEN=<your_token_here>
```

## Running the Models

### Step 1: Process Training Data (Run Once)

Process the cryptopuzzles repository data:

```bash
# For primary model
modal run train_llm_modal.py::main --command process

# OR for benchmark model (uses same function)
modal run train_llm_modal_benchmark.py::main --command process
```

This will:
- Clone the FMLBeast/cryptopuzzles repository
- Extract training examples from all markdown files
- Create train/val/test splits
- Upload to Modal volume

### Step 2: Train Primary Model

```bash
modal run train_llm_modal.py::main --command train
```

Expected:
- Training time: ~2-3 hours on A100
- Cost: ~$6-10
- Output: `/models/puzzle-llm/final/`

### Step 3: Train Benchmark Model

```bash
modal run train_llm_modal_benchmark.py::main --command train
```

Expected:
- Training time: ~3-4 hours on A100 (more epochs)
- Cost: ~$8-12
- Output: `/models/puzzle-llm-benchmark/final/`

### Step 4: Test Both Models

Test the primary model:
```bash
modal run train_llm_modal.py::main --command test --prompt "### Instruction:\nSolve this Caesar cipher:\n\nKHOOR ZRUOG\n\n### Response:\n"
```

Test the benchmark model:
```bash
modal run train_llm_modal_benchmark.py::main --command test --prompt "### Instruction:\nSolve this Caesar cipher:\n\nKHOOR ZRUOG\n\n### Response:\n"
```

## Benchmark Evaluation

### Automated Testing

Create a test suite with various puzzle types:

```python
test_puzzles = [
    "Caesar cipher: KHOOR ZRUOG",
    "Base64: SGVsbG8gV29ybGQ=",
    "ROT13: Uryyb Jbeyq",
    # Add more puzzles...
]
```

### Metrics to Compare

1. **Accuracy**: Percentage of correct solutions
2. **Response Quality**: How well does the model explain its reasoning?
3. **Inference Speed**: Time to generate solution
4. **Confidence**: Model's certainty in answers

### Example Comparison Script

```python
import modal
from train_llm_modal import test_model as test_primary
from train_llm_modal_benchmark import test_benchmark_model

test_cases = [
    "### Instruction:\nSolve this puzzle:\nKHOOR ZRUOG\n### Response:\n",
    # Add more test cases
]

for i, test in enumerate(test_cases):
    print(f"\n{'='*60}")
    print(f"Test {i+1}: {test[:50]}...")
    print(f"{'='*60}")

    print("\n🔵 Primary Model (Mistral-7B):")
    primary_response = test_primary.remote(test)
    print(primary_response)

    print("\n🟢 Benchmark Model (Llama-3.2-3B):")
    benchmark_response = test_benchmark_model.remote(test)
    print(benchmark_response)
```

## Cost Estimation

### Training Costs (A100 GPU)

| Model | Hours | Rate/hr | Total Cost |
|-------|-------|---------|------------|
| Primary (5 epochs) | 2-3 | $3-4 | $6-12 |
| Benchmark (8 epochs) | 3-4 | $3-4 | $9-16 |
| **Total** | **5-7** | - | **$15-28** |

### Storage Costs

- Modal Volume Storage: ~$0.10/GB/month
- Expected model size: ~3-4GB per model
- Monthly storage: ~$0.60-0.80

## Troubleshooting

### Out of Memory

If you get OOM errors:
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps` to 8
- Consider using a smaller model

### Data Not Found

If training fails with "No training data found":
```bash
# Re-run data processing
modal run train_llm_modal.py::main --command process
```

### HuggingFace Authentication

If you get auth errors:
1. Verify your token is set: `modal secret list`
2. Check Llama 3.2 access at HuggingFace
3. Recreate secret if needed

## Advanced Configuration

### Using Weights & Biases

Enable W&B logging for better tracking:

```bash
# Set W&B API key
modal secret create wandb-secret WANDB_API_KEY=<your_key>

# Train with W&B
modal run train_llm_modal.py::main --command train --use-wandb
```

### Custom Hyperparameters

Edit the `TRAINING_CONFIG` dictionary in either script:

```python
TRAINING_CONFIG = {
    "model_name": "your-preferred-model",
    "num_train_epochs": 10,  # More epochs
    "learning_rate": 1e-4,   # Lower learning rate
    # ... other parameters
}
```

### Using CTF Data

Both models can include CTF puzzle data:

```python
# In the training scripts, modify:
use_ctf_data=True  # Default
```

## Results Interpretation

### What to Look For

1. **Loss Curves**: Lower validation loss = better generalization
2. **Overfitting**: Large gap between train/val loss
3. **Convergence**: Does loss plateau or keep improving?

### Model Selection

Choose the model that:
- ✅ Achieves lower validation loss
- ✅ Produces more accurate puzzle solutions
- ✅ Provides better reasoning explanations
- ✅ Generalizes to new puzzle types

## Next Steps

1. **Evaluate on test set**: Run comprehensive benchmarks
2. **A/B testing**: Compare on real puzzle-solving tasks
3. **Error analysis**: Where does each model fail?
4. **Ensemble**: Can you combine predictions?
5. **Fine-tune further**: Use insights to improve both models

## Support

For issues or questions:
- Check Modal docs: [docs.modal.com](https://docs.modal.com)
- HuggingFace transformers: [huggingface.co/docs](https://huggingface.co/docs/transformers)
- Open an issue in this repository

## License

Same as main project license.
