# Puzzle LLM Training on Modal

A comprehensive pipeline for training Large Language Models on cryptographic puzzle datasets using Modal's serverless GPU infrastructure.

## Features

- **Efficient Fine-tuning**: Uses QLoRA (Quantized Low-Rank Adaptation) for memory-efficient training
- **Flexible Data Loading**: Supports JSON, CSV, and text file formats
- **Scalable Infrastructure**: Leverages Modal for serverless GPU compute
- **Multiple Model Support**: Works with Llama, Mistral, Phi, and other HuggingFace models
- **Automated Data Pipeline**: End-to-end preprocessing and training workflow

## Architecture

```
┌─────────────────┐
│  Raw Puzzle     │
│  Data (JSON/    │
│  CSV/TXT)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data            │
│ Preprocessing   │
│ (formats data   │
│ for training)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Modal Upload    │
│ (transfers to   │
│ cloud volume)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Modal Training  │
│ (GPU-based      │
│ fine-tuning     │
│ with QLoRA)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Trained Model   │
│ (saved to       │
│ volume)         │
└─────────────────┘
```

## Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **Python 3.11+**
3. **Modal CLI installed**:
   ```bash
   pip install modal
   ```

## Setup

### 1. Install Modal and Authenticate

```bash
# Install Modal
pip install modal

# Authenticate (will open browser)
python3 -m modal setup

# Or use token directly
modal token set --token-id YOUR_TOKEN_ID --token-secret YOUR_TOKEN_SECRET
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Your Data

Your puzzle data can be in any of these formats:

**Option A: JSON Format**
```json
[
  {
    "puzzle": "Caesar cipher: KHOOR ZRUOG",
    "solution": "HELLO WORLD (shift of 3)",
    "difficulty": "easy",
    "category": "classical cipher",
    "hint": "Try different shift values"
  }
]
```

**Option B: CSV Format**
```csv
puzzle,solution,difficulty,category,hint
"Caesar cipher: KHOOR ZRUOG","HELLO WORLD (shift of 3)",easy,classical cipher,Try different shift values
```

**Option C: Text Format** (one puzzle per paragraph)
```
Caesar cipher: KHOOR ZRUOG
Solution: HELLO WORLD (shift of 3)

What is the SHA-256 hash of 'password'?
Solution: 5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8
```

## Usage

### Step 1: Preprocess Your Data

```bash
# If you have your own data
python data_preprocessing.py /path/to/your/puzzle/data

# Or create sample data for testing
python data_preprocessing.py
```

This will create a `processed_data` directory with:
- `train.json`, `validation.json`, `test.json`
- `train_dataset/`, `val_dataset/`, `test_dataset/` (HuggingFace format)

### Step 2: Upload Data to Modal

```bash
modal run train_llm_modal.py --command upload --data-path processed_data
```

### Step 3: Train the Model

```bash
# Train with default settings (Llama 3.2 1B on A10G GPU)
modal run train_llm_modal.py --command train

# Train with a different model
modal run train_llm_modal.py --command train --model-name "microsoft/phi-2"
```

Training will:
- Load the preprocessed data from Modal volume
- Fine-tune the model using QLoRA on GPU
- Save checkpoints to the persistent volume
- Take approximately 1-3 hours depending on dataset size and GPU

### Step 4: Test the Model

```bash
# Test with default prompt
modal run train_llm_modal.py --command test

# Test with custom prompt
modal run train_llm_modal.py --command test --prompt "### Instruction:\nSolve this puzzle: ROT13 URYYB\n\n### Response:\n"
```

## Configuration

Edit `config.yaml` to customize:

- **Model selection**: Change `model.name` to use different base models
- **Training parameters**: Adjust epochs, batch size, learning rate
- **GPU type**: Choose between T4, A10G, or A100
- **LoRA settings**: Modify rank and alpha for different adaptation strengths

## Data Format Details

The preprocessor automatically formats your data into instruction-tuning format:

```
### Instruction:
Solve this cryptographic puzzle (classical cipher) [Difficulty: easy]:

Caesar cipher: KHOOR ZRUOG

Hint: Try different shift values

### Response:
HELLO WORLD (shift of 3)
```

## Supported Models

### Small Models (1-3B parameters)
- `meta-llama/Llama-3.2-1B` ⭐ Recommended for starting
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `microsoft/phi-2`
- `meta-llama/Llama-3.2-3B`

### Medium Models (7-13B parameters)
- `mistralai/Mistral-7B-v0.1`
- `meta-llama/Llama-2-7b-hf`

*Note: Larger models require more GPU memory (A100 recommended)*

## GPU Options

| GPU Type | Memory | Cost/hr | Best For |
|----------|--------|---------|----------|
| T4       | 16GB   | ~$0.40  | Inference, small models |
| A10G     | 24GB   | ~$1.10  | Training 1-3B models ⭐ |
| A100     | 40GB   | ~$4.00  | Training 7B+ models |

## Monitoring Training

### Enable Weights & Biases (Optional)

1. Create a W&B account at [wandb.ai](https://wandb.ai)
2. Create a Modal secret:
   ```bash
   modal secret create wandb-secret WANDB_API_KEY=your_key_here
   ```
3. Set `use_wandb=True` in the training function

### View Logs

```bash
# View Modal logs
modal app logs puzzle-llm-training
```

## Advanced Usage

### Custom Training Script

You can modify `train_llm_modal.py` to:
- Add custom metrics
- Implement different training strategies
- Use different quantization methods
- Add early stopping
- Implement curriculum learning

### Distributed Training

For larger models, enable multi-GPU training by modifying the Modal function decorator:

```python
@app.function(
    image=image,
    gpu="A100:2",  # Use 2 A100 GPUs
    ...
)
```

## Troubleshooting

### Issue: "Could not connect to Modal server"
- Check your internet connection
- Verify your Modal token is correct
- Try re-authenticating: `python3 -m modal setup`

### Issue: Out of memory during training
- Reduce `per_device_train_batch_size` in config
- Increase `gradient_accumulation_steps`
- Use a smaller model
- Upgrade to a larger GPU (A100)

### Issue: Training is too slow
- Increase batch size if memory allows
- Use a larger GPU (A10G → A100)
- Reduce `max_seq_length`

### Issue: Model quality is poor
- Increase training epochs
- Use more training data
- Try a larger base model
- Adjust learning rate
- Increase LoRA rank

## File Structure

```
puzzle_llm/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── config.yaml               # Configuration file
├── data_preprocessing.py     # Data preprocessing script
├── train_llm_modal.py       # Main Modal training script
├── processed_data/          # Processed training data (generated)
│   ├── train.json
│   ├── validation.json
│   ├── test.json
│   ├── train_dataset/
│   ├── val_dataset/
│   └── test_dataset/
└── sample_data/             # Sample puzzle data (generated)
    └── puzzles.json
```

## Cost Estimation

Approximate costs for training (using A10G GPU):

- **Small dataset** (100-1000 puzzles): $2-5
- **Medium dataset** (1000-10000 puzzles): $5-20
- **Large dataset** (10000+ puzzles): $20-100

*Costs depend on dataset size, model size, and number of epochs*

## Next Steps

After training:

1. **Evaluate**: Test the model on your test set
2. **Deploy**: Use Modal for inference serving
3. **Iterate**: Gather feedback and retrain with more data
4. **Scale**: Deploy as a web API using Modal endpoints

## Resources

- [Modal Documentation](https://modal.com/docs)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PEFT (LoRA) Documentation](https://huggingface.co/docs/peft)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)

## Contributing

To add support for new model architectures or data formats:

1. Modify `data_preprocessing.py` for new formats
2. Update `train_llm_modal.py` for new models
3. Add configuration to `config.yaml`

## License

MIT License - Feel free to use for your puzzle-solving AI projects!

## Support

For issues specific to:
- **Modal**: [Modal Discord](https://modal.com/discord)
- **This project**: Open an issue on GitHub
