#!/usr/bin/env python3
"""
Local training script for puzzle-solving LLM.
This can run on your local machine with a GPU (or CPU, though very slow).

Usage:
    # Train primary model
    python train_local.py --config primary --data-dir ./training_data --output-dir ./models/primary

    # Train benchmark model
    python train_local.py --config benchmark --data-dir ./training_data --output-dir ./models/benchmark
"""

import argparse
import json
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset
import os


# Configuration presets
CONFIGS = {
    "primary": {
        "name": "Primary Model (Mistral-7B)",
        "model_name": "mistralai/Mistral-7B-v0.1",
        "max_seq_length": 512,
        "num_train_epochs": 5,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "warmup_steps": 100,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
    },
    "benchmark": {
        "name": "Benchmark Model (Llama-3.2-3B)",
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        "max_seq_length": 512,
        "num_train_epochs": 8,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "learning_rate": 3e-4,
        "warmup_steps": 150,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
    }
}


def load_dataset_from_json(data_dir: Path) -> tuple:
    """Load training and validation datasets from JSON files."""
    print(f"📚 Loading datasets from {data_dir}...")

    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "validation.jsonl"

    # Try JSONL first, then JSON
    if train_file.exists():
        with open(train_file, 'r') as f:
            train_data = [json.loads(line) for line in f]
    else:
        with open(data_dir / "train.json", 'r') as f:
            train_data = json.load(f)

    if val_file.exists():
        with open(val_file, 'r') as f:
            val_data = [json.loads(line) for line in f]
    else:
        with open(data_dir / "validation.json", 'r') as f:
            val_data = json.load(f)

    print(f"  ✅ Loaded {len(train_data)} training examples")
    print(f"  ✅ Loaded {len(val_data)} validation examples")

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    return train_dataset, val_dataset


def train_model(
    config_name: str,
    data_dir: Path,
    output_dir: Path,
    use_quantization: bool = True,
    use_wandb: bool = False,
    hf_token: str = None,
):
    """Train the model with the specified configuration."""

    config = CONFIGS[config_name]

    print("=" * 80)
    print(f"🚀 Starting LOCAL training: {config['name']}")
    print("=" * 80)
    print(f"  Model: {config['model_name']}")
    print(f"  Epochs: {config['num_train_epochs']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  LoRA r/alpha: {config['lora_r']}/{config['lora_alpha']}")
    print(f"  Output: {output_dir}")
    print()

    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"📊 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("⚠️  WARNING: No GPU detected. Training will be VERY slow!")
        print("   Consider using Modal or a cloud GPU provider.")

    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load datasets
    train_dataset, val_dataset = load_dataset_from_json(data_dir)

    # Load tokenizer
    print("\n📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_name'],
        trust_remote_code=True,
        token=hf_token
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Configure quantization if requested
    bnb_config = None
    if use_quantization and torch.cuda.is_available():
        print("🔧 Using 4-bit quantization for memory efficiency...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    # Load model
    print("🤖 Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        token=hf_token,
    )

    # Prepare model for k-bit training
    if use_quantization and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    print("⚙️  Configuring LoRA...")
    peft_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # Apply LoRA to model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Training arguments
    print("\n🎯 Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config['num_train_epochs'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        per_device_eval_batch_size=config['per_device_train_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'],
        warmup_steps=config['warmup_steps'],
        logging_steps=10,
        save_steps=100,
        eval_steps=50,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        optim="paged_adamw_32bit" if torch.cuda.is_available() else "adamw_torch",
        lr_scheduler_type="cosine",
        report_to="wandb" if use_wandb else "none",
        save_total_limit=3,
        push_to_hub=False,
    )

    # Initialize trainer
    print("🔥 Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    # Train!
    print("\n" + "=" * 80)
    print("🔥 STARTING TRAINING...")
    print("=" * 80)
    trainer.train()

    # Save final model
    print("\n💾 Saving final model...")
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print(f"\n✅ Training complete! Model saved to: {final_dir}")

    return str(final_dir)


def main():
    parser = argparse.ArgumentParser(description="Train puzzle-solving LLM locally")
    parser.add_argument(
        "--config",
        type=str,
        choices=["primary", "benchmark"],
        required=True,
        help="Model configuration to use"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./training_data",
        help="Directory containing training data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save trained model"
    )
    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="Disable 4-bit quantization (requires more GPU memory)"
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var)"
    )

    args = parser.parse_args()

    # Get HF token from args or environment
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # Train the model
    model_path = train_model(
        config_name=args.config,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        use_quantization=not args.no_quantization,
        use_wandb=args.use_wandb,
        hf_token=hf_token,
    )

    print("\n" + "=" * 80)
    print("🎉 ALL DONE!")
    print("=" * 80)
    print(f"\nYour trained model is ready at: {model_path}")
    print("\nNext steps:")
    print(f"  1. Test the model: python test_local.py --model-dir {model_path}")
    print(f"  2. Compare with other models using evaluate_models.py")


if __name__ == "__main__":
    main()
