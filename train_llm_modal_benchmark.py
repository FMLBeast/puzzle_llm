"""
Modal-based BENCHMARK LLM training script for cryptopuzzle dataset.
This script trains a second model with different hyperparameters to benchmark against the primary model.

KEY DIFFERENCES FROM PRIMARY MODEL:
- Different base model: Llama-3.2-3B-Instruct (vs Mistral-7B)
- Higher learning rate: 3e-4 (vs 2e-4)
- More aggressive LoRA: r=32, alpha=64 (vs r=16, alpha=32)
- Longer training: 8 epochs (vs 5)
- Different batch configuration
- Using newer Llama 3.2 architecture for comparison
"""

import modal
from pathlib import Path

# Define Modal app
app = modal.App("puzzle-llm-benchmark-training")

# Create a Modal image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")  # Need git for cloning cryptopuzzles repo
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.14.0",
        "accelerate>=0.20.0",
        "bitsandbytes>=0.41.0",
        "peft>=0.4.0",
        "trl>=0.7.0",
        "wandb>=0.15.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
    )
)

# Create a persistent volume for BENCHMARK model storage
volume = modal.Volume.from_name("puzzle-llm-benchmark-models", create_if_missing=True)

# Reference the CTF puzzle data volume (shared with primary model)
ctf_volume = modal.Volume.from_name("puzzle-training-data", create_if_missing=True)

# BENCHMARK TRAINING CONFIGURATION - Optimized for comparison
TRAINING_CONFIG = {
    # Different base model for benchmarking - using latest Llama 3.2
    "model_name": "meta-llama/Llama-3.2-3B-Instruct",  # Alternative to Mistral

    # Sequence and batch settings
    "max_seq_length": 512,
    "per_device_train_batch_size": 4,  # Larger batch size
    "gradient_accumulation_steps": 2,  # Fewer accumulation steps

    # Learning configuration - more aggressive
    "num_train_epochs": 8,  # More epochs for thorough learning
    "learning_rate": 3e-4,  # Higher learning rate
    "warmup_steps": 150,  # More warmup

    # LoRA configuration - stronger adaptation
    "lora_r": 32,  # Doubled from primary
    "lora_alpha": 64,  # Doubled from primary
    "lora_dropout": 0.1,  # Higher dropout for regularization

    # Logging and saving
    "logging_steps": 10,
    "save_steps": 100,
    "eval_steps": 50,

    # Output directory - separate from primary model
    "output_dir": "/models/puzzle-llm-benchmark",
}


@app.function(
    image=image,
    gpu="A100",  # Using A100 for consistent comparison
    volumes={"/models": volume, "/ctf_data": ctf_volume},  # Mount BOTH volumes!
    timeout=86400,  # 24 hours
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train_benchmark_model(
    data_path: str = "/data",
    model_name: str = TRAINING_CONFIG["model_name"],
    use_wandb: bool = False,
    use_ctf_data: bool = True,  # Include CTF puzzles by default
):
    """
    Train the BENCHMARK LLM on puzzle dataset using QLoRA fine-tuning.

    Args:
        data_path: Path to training data
        model_name: HuggingFace model to fine-tune
        use_wandb: Whether to use Weights & Biases for logging
        use_ctf_data: Whether to include CTF puzzle data
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import load_from_disk, load_dataset
    import os

    # Reload volume to see data uploaded in previous function call
    volume.reload()
    print("🔄 Volume reloaded")

    # Debug: List what's in the volume
    import os as debug_os
    print(f"📂 Checking volume contents...")
    if debug_os.path.exists("/models"):
        print(f"  /models exists: {debug_os.listdir('/models')}")
        if debug_os.path.exists("/models/data"):
            print(f"  /models/data exists: {debug_os.listdir('/models/data')}")
        else:
            print("  /models/data does NOT exist!")
    else:
        print("  /models does NOT exist!")

    print(f"🎯 BENCHMARK MODEL TRAINING")
    print(f"🚀 Base model: {model_name}")
    print(f"📊 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"⚙️  Config: lr={TRAINING_CONFIG['learning_rate']}, epochs={TRAINING_CONFIG['num_train_epochs']}, LoRA r={TRAINING_CONFIG['lora_r']}")

    # Get HuggingFace token from environment (set via Modal secret)
    hf_token = os.environ.get("HF_TOKEN", None)

    # Initialize W&B if requested
    if use_wandb:
        import wandb
        wandb.init(project="puzzle-llm-benchmark", name=f"benchmark-{model_name.split('/')[-1]}")

    # Load tokenizer
    print("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Configure quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model with quantization
    print("🤖 Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA - MORE AGGRESSIVE than primary model
    peft_config = LoraConfig(
        r=TRAINING_CONFIG["lora_r"],
        lora_alpha=TRAINING_CONFIG["lora_alpha"],
        lora_dropout=TRAINING_CONFIG["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # Apply LoRA to model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load datasets - SAME DATA as primary model for fair comparison
    print("📚 Loading datasets...")
    from datasets import Dataset, concatenate_datasets
    import json

    all_train_data = []
    all_eval_data = []

    # 1. Load CTF puzzles if enabled
    if use_ctf_data:
        print("\n1️⃣ Loading CTF puzzle dataset...")
        ctf_volume.reload()

        ctf_path = "/ctf_data/advanced_ctf_puzzles/training_format.json"
        if os.path.exists(ctf_path):
            with open(ctf_path, 'r') as f:
                ctf_data = json.load(f)

            print(f"  ✅ Loaded {len(ctf_data)} CTF puzzles")

            # Split CTF data: 90% train, 10% val
            split_idx = int(0.9 * len(ctf_data))
            all_train_data.extend(ctf_data[:split_idx])
            all_eval_data.extend(ctf_data[split_idx:])
            print(f"     Train: {split_idx}, Val: {len(ctf_data) - split_idx}")
        else:
            print(f"  ⚠️  CTF data not found at {ctf_path}")

    # 2. Load cryptopuzzles repo data
    print("\n2️⃣ Loading cryptopuzzles repository data...")
    crypto_data_path = "/models/data"

    if os.path.exists(crypto_data_path):
        # Check what files exist
        print(f"  Files in {crypto_data_path}:")
        for item in os.listdir(crypto_data_path):
            print(f"    - {item}")

        try:
            # Try loading from saved datasets
            if os.path.exists(f"{crypto_data_path}/train_dataset"):
                train_ds = load_from_disk(f"{crypto_data_path}/train_dataset")
                all_train_data.extend([ex for ex in train_ds])
                print(f"  ✅ Loaded {len(train_ds)} crypto training examples")

            if os.path.exists(f"{crypto_data_path}/val_dataset"):
                val_ds = load_from_disk(f"{crypto_data_path}/val_dataset")
                all_eval_data.extend([ex for ex in val_ds])
                print(f"  ✅ Loaded {len(val_ds)} crypto validation examples")
        except Exception as e:
            print(f"  Disk format failed ({e}), trying JSON...")

            # Try JSON format
            if os.path.exists(f"{crypto_data_path}/train.json"):
                with open(f"{crypto_data_path}/train.json", 'r') as f:
                    crypto_train = json.load(f)
                all_train_data.extend(crypto_train)
                print(f"  ✅ Loaded {len(crypto_train)} crypto training examples from JSON")

            if os.path.exists(f"{crypto_data_path}/validation.json"):
                with open(f"{crypto_data_path}/validation.json", 'r') as f:
                    crypto_val = json.load(f)
                all_eval_data.extend(crypto_val)
                print(f"  ✅ Loaded {len(crypto_val)} crypto validation examples from JSON")
    else:
        print(f"  ⚠️  Crypto data not found at {crypto_data_path}")

    # 3. Create final combined datasets
    print(f"\n3️⃣ Creating combined dataset...")
    train_dataset = Dataset.from_list(all_train_data) if all_train_data else None
    eval_dataset = Dataset.from_list(all_eval_data) if all_eval_data else None

    if train_dataset is None:
        raise ValueError("No training data found! Run process command first.")

    print(f"\n✅ COMBINED DATASET:")
    print(f"   Training: {len(train_dataset)} examples")
    print(f"   Validation: {len(eval_dataset) if eval_dataset else 0} examples")

    # Training arguments - BENCHMARK configuration
    training_args = TrainingArguments(
        output_dir=TRAINING_CONFIG["output_dir"],
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        warmup_steps=TRAINING_CONFIG["warmup_steps"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        save_steps=TRAINING_CONFIG["save_steps"],
        eval_steps=TRAINING_CONFIG["eval_steps"],
        eval_strategy="steps" if (eval_dataset and len(eval_dataset) > 0) else "no",
        save_strategy="steps",
        load_best_model_at_end=True if (eval_dataset and len(eval_dataset) > 0) else False,
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        report_to="wandb" if use_wandb else "none",
        save_total_limit=3,
        push_to_hub=False,
    )

    # Initialize trainer
    print("🎯 Initializing benchmark trainer...")

    # Only include eval_dataset if it's not empty
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "processing_class": tokenizer,
    }

    if eval_dataset and len(eval_dataset) > 0:
        trainer_kwargs["eval_dataset"] = eval_dataset
        print(f"  Including validation dataset ({len(eval_dataset)} examples)")
    else:
        print(f"  ⚠️  No validation data - training without evaluation")

    trainer = SFTTrainer(**trainer_kwargs)

    # Train!
    print("🔥 Starting BENCHMARK training...")
    trainer.train()

    # Save final model
    print("💾 Saving final benchmark model...")
    trainer.save_model(f"{TRAINING_CONFIG['output_dir']}/final")
    tokenizer.save_pretrained(f"{TRAINING_CONFIG['output_dir']}/final")

    # Commit volume changes
    volume.commit()

    print("✨ Benchmark training complete!")
    return {"status": "success", "output_dir": f"{TRAINING_CONFIG['output_dir']}/final"}


@app.function(
    image=image,
    gpu="T4",  # Lighter GPU for inference
    volumes={"/models": volume},
)
def test_benchmark_model(prompt: str, model_path: str = None):
    """
    Test the trained benchmark model with a sample prompt.

    Args:
        prompt: The puzzle to solve
        model_path: Path to the trained model (optional)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Reload volume to see trained model
    volume.reload()
    print("🔄 Volume reloaded")

    if model_path is None:
        model_path = f"{TRAINING_CONFIG['output_dir']}/final"

    print(f"🔍 Loading benchmark model from {model_path}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        TRAINING_CONFIG["model_name"],
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


@app.function(image=image, volumes={"/models": volume}, timeout=7200)
def process_and_upload_cryptopuzzle_data():
    """
    Process cryptopuzzles repository data and upload to Modal volume.
    This is SHARED with the primary model - we use the SAME data for fair benchmarking.
    """
    import json
    import subprocess
    from pathlib import Path
    from datasets import Dataset
    import random
    import re

    print("🔥 Processing cryptopuzzle data from FMLBeast/cryptopuzzles repo!")
    print("   This data will be shared between primary and benchmark models.\n")

    # Clone repository
    repo_dir = Path("/tmp/cryptopuzzles")
    if repo_dir.exists():
        import shutil
        shutil.rmtree(repo_dir)

    print("📥 Cloning cryptopuzzles repository...")
    subprocess.run([
        "git", "clone",
        "https://github.com/FMLBeast/cryptopuzzles.git",
        str(repo_dir)
    ], check=True)

    all_examples = []

    # Helper function to extract sub-puzzles from markdown with REASONING
    def extract_subpuzzles_from_markdown(file_path: Path, puzzle_name: str) -> list:
        """Extract each sub-puzzle as a training example."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        examples = []

        # Extract structured Question/Solution/Method blocks
        question_pattern = r'###\s+(.+?)\n\*\*Question\*\*:(.+?)(?:\n\*\*Solution\*\*:|$)(.+?)(?:\n\*\*Method\*\*:|$)(.+?)(?=\n###|\n---|\Z)'
        matches = re.finditer(question_pattern, content, re.DOTALL | re.IGNORECASE)

        for match in matches:
            title = match.group(1).strip()
            question = match.group(2).strip()

            full_match = match.group(0)
            solution_match = re.search(r'\*\*Solution\*\*:\s*(.+?)(?=\n\*\*Method\*\*:|\n###|\n---|\Z)', full_match, re.DOTALL)
            method_match = re.search(r'\*\*Method\*\*:\s*(.+?)(?=\n###|\n---|\Z)', full_match, re.DOTALL)

            if question and (solution_match or method_match):
                solution = solution_match.group(1).strip() if solution_match else ""
                method = method_match.group(1).strip() if method_match else ""

                instruction = f"### Instruction:\nSolve this cryptopuzzle:\n\n**{title}**\n\n{question}\n\n### Response:\n"

                response = ""
                if method:
                    response += f"**Solution Method:**\n{method}\n\n"
                if solution:
                    response += f"**Final Answer:** {solution}"

                examples.append({
                    "text": instruction + response,
                    "source": f"solved_puzzle_{file_path.stem}",
                    "difficulty": "medium"
                })

        return examples

    def parse_full_markdown(file_path: Path, source_type: str) -> dict:
        """Extract full content from markdown."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else file_path.stem

        return {
            "text": f"### Instruction:\n[Type: {source_type}]\n\nExplain: {title}\n\n### Response:\n\n{content}",
            "source": source_type,
            "difficulty": "medium"
        }

    # Process all markdown files
    print("\n📚 Processing cryptopuzzles repository...")

    solved_dir = repo_dir / "puzzles" / "solved"
    if solved_dir.exists():
        for md_file in solved_dir.glob("*.md"):
            print(f"  📄 {md_file.name}")
            try:
                sub_examples = extract_subpuzzles_from_markdown(md_file, md_file.stem)
                if sub_examples:
                    all_examples.extend(sub_examples)
                else:
                    all_examples.append(parse_full_markdown(md_file, "solved_puzzle"))
            except Exception as e:
                print(f"    ⚠️  Error: {e}")

    # Process technique docs
    techniques_dir = repo_dir / "techniques"
    if techniques_dir.exists():
        for md_file in techniques_dir.rglob("*.md"):
            print(f"  📄 {md_file.name}")
            try:
                all_examples.append(parse_full_markdown(md_file, "technique_guide"))
            except Exception as e:
                print(f"    ⚠️  Error: {e}")

    print(f"\n📊 TOTAL EXAMPLES: {len(all_examples)}")

    # Split dataset
    random.seed(42)
    random.shuffle(all_examples)

    n = len(all_examples)
    test_size = max(2, int(0.1 * n))
    val_size = max(2, int(0.1 * n))
    train_size = n - val_size - test_size

    train_data = all_examples[:train_size]
    val_data = all_examples[train_size:train_size + val_size]
    test_data = all_examples[train_size + val_size:]

    print(f"\n📊 Dataset split:")
    print(f"  📚 Train: {len(train_data)}")
    print(f"  📚 Val: {len(val_data)}")
    print(f"  📚 Test: {len(test_data)}")

    # Save to volume
    dest = Path("/models/data")
    dest.mkdir(exist_ok=True, parents=True)

    with open(dest / "train.json", 'w') as f:
        json.dump(train_data, f, indent=2)
    with open(dest / "validation.json", 'w') as f:
        json.dump(val_data, f, indent=2)
    with open(dest / "test.json", 'w') as f:
        json.dump(test_data, f, indent=2)

    Dataset.from_list(train_data).save_to_disk(str(dest / "train_dataset"))
    Dataset.from_list(val_data).save_to_disk(str(dest / "val_dataset"))
    Dataset.from_list(test_data).save_to_disk(str(dest / "test_dataset"))

    volume.commit()
    print("\n✅ Dataset created and saved to volume!")

    return {
        "train": len(train_data),
        "val": len(val_data),
        "test": len(test_data),
        "total": len(all_examples),
        "status": "success"
    }


@app.local_entrypoint()
def main(
    command: str = "train",
    data_path: str = "processed_data",
    model_name: str = TRAINING_CONFIG["model_name"],
    prompt: str = None,
):
    """
    Main entry point for the BENCHMARK training pipeline.

    Args:
        command: Command to run (process, train, test)
        data_path: Path to training data
        model_name: Model to fine-tune
        prompt: Prompt for testing (if command is 'test')
    """
    if command == "process":
        print(f"🔄 Processing cryptopuzzle data from FMLBeast/cryptopuzzles...")
        result = process_and_upload_cryptopuzzle_data.remote()
        print(f"✅ Data processing complete: {result}")

    elif command == "train":
        print(f"🚀 Starting BENCHMARK training job...")
        print(f"📊 Configuration:")
        print(f"  - Model: {model_name}")
        print(f"  - Learning rate: {TRAINING_CONFIG['learning_rate']}")
        print(f"  - Epochs: {TRAINING_CONFIG['num_train_epochs']}")
        print(f"  - LoRA r/alpha: {TRAINING_CONFIG['lora_r']}/{TRAINING_CONFIG['lora_alpha']}")

        result = train_benchmark_model.remote(
            data_path="/models/data",
            model_name=model_name,
            use_wandb=False,
        )
        print(f"✅ Benchmark training complete: {result}")

    elif command == "test":
        if prompt is None:
            prompt = "### Instruction:\nSolve this cryptographic puzzle (classical cipher) [Difficulty: easy]:\n\nKHOOR ZRUOG\n\nHint: Try different shift values\n\n### Response:\n"

        print(f"🧪 Testing benchmark model with prompt: {prompt}")
        response = test_benchmark_model.remote(prompt)
        print(f"\n📝 Benchmark model response:\n{response}")

    else:
        print(f"Unknown command: {command}")
        print("Available commands: process, train, test")
