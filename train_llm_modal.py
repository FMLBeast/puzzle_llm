"""
Modal-based LLM training script for cryptopuzzle dataset.
Fine-tunes language models using QLoRA for efficient training.
"""

import modal
from pathlib import Path

# Define Modal app
app = modal.App("puzzle-llm-training")

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

# Create a persistent volume for model storage
volume = modal.Volume.from_name("puzzle-llm-models", create_if_missing=True)

# Training configuration
TRAINING_CONFIG = {
    "model_name": "mistralai/Mistral-7B-v0.1",  # Better quality (7B), already approved!
    "max_seq_length": 512,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,  # Reduced for larger 7B model
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 100,
    "logging_steps": 10,
    "save_steps": 100,
    "eval_steps": 50,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "output_dir": "/models/puzzle-llm",
}


@app.function(
    image=image,
    gpu="A10G",  # Can upgrade to A100 for faster training
    volumes={"/models": volume},
    timeout=86400,  # 24 hours
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train_model(
    data_path: str = "/data",
    model_name: str = TRAINING_CONFIG["model_name"],
    use_wandb: bool = False,
):
    """
    Train the LLM on puzzle dataset using QLoRA fine-tuning.

    Args:
        data_path: Path to training data
        model_name: HuggingFace model to fine-tune
        use_wandb: Whether to use Weights & Biases for logging
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

    print(f"🚀 Starting training with model: {model_name}")
    print(f"📊 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Get HuggingFace token from environment (set via Modal secret)
    hf_token = os.environ.get("HF_TOKEN", None)

    # Initialize W&B if requested
    if use_wandb:
        import wandb
        wandb.init(project="puzzle-llm", name=f"training-{model_name.split('/')[-1]}")

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

    # Configure LoRA
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

    # Load datasets
    print("📚 Loading datasets...")

    # Debug: Check what files exist
    import os as check_os
    print(f"  Files in {data_path}:")
    for item in check_os.listdir(data_path):
        print(f"    - {item}")

    try:
        train_dataset = load_from_disk(f"{data_path}/train_dataset")
        eval_dataset = load_from_disk(f"{data_path}/val_dataset")
        print(f"  Loaded from disk format")
    except Exception as e:
        print(f"  Disk format failed ({e}), trying JSON...")
        # Fallback: try loading from JSON
        from datasets import Dataset
        import json

        with open(f"{data_path}/train.json", 'r') as f:
            train_data = json.load(f)
        with open(f"{data_path}/validation.json", 'r') as f:
            eval_data = json.load(f)

        train_dataset = Dataset.from_list(train_data)
        eval_dataset = Dataset.from_list(eval_data) if len(eval_data) > 0 else Dataset.from_list([{"text": "dummy"}])  # Dummy if empty

    print(f"✅ Loaded {len(train_dataset)} training examples")
    print(f"✅ Loaded {len(eval_dataset)} validation examples")

    # Training arguments
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
        eval_strategy="steps" if len(eval_dataset) > 0 else "no",  # Fixed API name
        save_strategy="steps",
        load_best_model_at_end=True if len(eval_dataset) > 0 else False,
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        report_to="wandb" if use_wandb else "none",
        save_total_limit=3,
        push_to_hub=False,
    )

    # Initialize trainer
    print("🎯 Initializing trainer...")

    # Only include eval_dataset if it's not empty
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "processing_class": tokenizer,
    }

    if len(eval_dataset) > 0:
        trainer_kwargs["eval_dataset"] = eval_dataset
        print(f"  Including validation dataset ({len(eval_dataset)} examples)")
    else:
        print(f"  ⚠️  No validation data - training without evaluation")

    trainer = SFTTrainer(**trainer_kwargs)

    # Train!
    print("🔥 Starting training...")
    trainer.train()

    # Save final model
    print("💾 Saving final model...")
    trainer.save_model(f"{TRAINING_CONFIG['output_dir']}/final")
    tokenizer.save_pretrained(f"{TRAINING_CONFIG['output_dir']}/final")

    # Commit volume changes
    volume.commit()

    print("✨ Training complete!")
    return {"status": "success", "output_dir": f"{TRAINING_CONFIG['output_dir']}/final"}


@app.function(
    image=image,
    gpu="T4",  # Lighter GPU for inference
    volumes={"/models": volume},
)
def test_model(prompt: str, model_path: str = None):
    """
    Test the trained model with a sample prompt.

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

    print(f"🔍 Loading model from {model_path}")

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


@app.function(image=image, volumes={"/models": volume}, timeout=7200)  # Increased timeout for large processing
def process_and_upload_data():
    """
    COMPREHENSIVE data processing - extracts ALL information from cryptopuzzles repo.
    Creates hundreds of training examples from:
    - Solved puzzles (with full solutions)
    - Unsolved puzzle analysis
    - Technique documentation
    - Cryptocurrency puzzles
    - Smart contracts
    - Steganography
    - Tool usage
    """
    import json
    import subprocess
    from pathlib import Path
    from datasets import Dataset
    import random
    import re

    print("🔄 Starting COMPREHENSIVE cryptopuzzle data processing...")
    print("   This will extract data from the entire repository!\n")

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

    # Helper function to parse markdown sections
    def parse_markdown_file(file_path: Path) -> dict:
        """Extract structured information from markdown files."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Extract title
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else file_path.stem

        # Extract sections
        sections = {}
        current_section = "intro"
        current_content = []

        for line in content.split('\n'):
            if line.startswith('##'):
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = line.strip('# ').lower().replace(' ', '_')
                current_content = []
            else:
                current_content.append(line)

        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()

        return {"title": title, "sections": sections, "full_content": content}

    # 1. Process ARweave training dataset (curated examples)
    print("\n1️⃣ Processing ARweave curated training dataset...")
    training_file = repo_dir / "data" / "training_dataset.json"
    arweave_count = 0
    if training_file.exists():
        with open(training_file, 'r') as f:
            dataset = json.load(f)

        for puzzle in dataset.get('training_examples', []):
            instruction = f"### Instruction:\n"
            instruction += f"[Puzzle Type: {puzzle.get('type', 'general')}] "
            instruction += f"[Difficulty: {puzzle.get('difficulty', 'medium')}]\n\n"
            instruction += f"{puzzle.get('problem', '')}\n\n"

            if 'sub_problems' in puzzle:
                instruction += "Sub-problems:\n"
                for i, sub in enumerate(puzzle['sub_problems'], 1):
                    instruction += f"{i}. [{sub.get('type')}] {sub.get('question', '')}\n"

            instruction += "\n### Response:\n"

            response = "Let me solve each sub-problem systematically:\n\n"
            if 'sub_problems' in puzzle:
                for i, sub in enumerate(puzzle['sub_problems'], 1):
                    response += f"**Problem {i}** ({sub.get('type')}):\n"
                    if 'steps' in sub:
                        response += "Reasoning:\n" + '\n'.join(f"- {s}" for s in sub['steps']) + "\n"
                    elif 'explanation' in sub:
                        response += f"Explanation: {sub['explanation']}\n"
                    response += f"Solution: {sub.get('solution', '')}\n\n"

            if 'final_solution' in puzzle:
                response += f"**Final Answer**: {puzzle['final_solution']}\n"

            all_examples.append({
                "text": instruction + response,
                "source": "arweave_curated",
                "difficulty": puzzle.get('difficulty', 'medium')
            })
            arweave_count += 1

    print(f"  ✓ {arweave_count} ARweave curated examples")

    # 2. Process ALL solved puzzle writeups
    print("\n2️⃣ Processing solved puzzle writeups...")
    solved_dir = repo_dir / "puzzles" / "solved"
    solved_count = 0
    if solved_dir.exists():
        for md_file in solved_dir.glob("*.md"):
            try:
                parsed = parse_markdown_file(md_file)

                # Create instruction from puzzle
                instruction = f"### Instruction:\n"
                instruction += f"[Source: ARweave Puzzle] [File: {md_file.name}]\n\n"
                instruction += f"Explain the solution to: {parsed['title']}\n\n"
                instruction += "### Response:\n"

                # Extract key information
                response = f"# {parsed['title']}\n\n"

                # Add solution if present
                if 'solution' in parsed['sections']:
                    response += f"**Solution:**\n{parsed['sections']['solution'][:500]}\n\n"

                # Add techniques if present
                if 'techniques_used' in parsed['sections']:
                    response += f"**Techniques:**\n{parsed['sections']['techniques_used'][:300]}\n\n"

                # Add method if present
                if 'method' in parsed['sections']:
                    response += f"**Method:**\n{parsed['sections']['method'][:400]}\n\n"

                all_examples.append({
                    "text": instruction + response,
                    "source": "solved_writeup",
                    "difficulty": "medium"
                })
                solved_count += 1
            except Exception as e:
                print(f"    ⚠️  Skipped {md_file.name}: {e}")

    print(f"  ✓ {solved_count} solved puzzle writeups")

    # 3. Process technique deep dives
    print("\n3️⃣ Processing technique documentation...")
    techniques_dir = repo_dir / "techniques" / "deep_dives"
    technique_count = 0
    if techniques_dir.exists():
        for md_file in techniques_dir.glob("*.md"):
            try:
                parsed = parse_markdown_file(md_file)

                instruction = f"### Instruction:\n"
                instruction += f"[Type: Technique Documentation]\n\n"
                instruction += f"Explain the technique: {parsed['title']}\n\n"
                instruction += "### Response:\n"

                # Extract first 1000 chars of content as the explanation
                response = parsed['full_content'][:1500] + "..."

                all_examples.append({
                    "text": instruction + response,
                    "source": "technique_guide",
                    "difficulty": "educational"
                })
                technique_count += 1
            except Exception as e:
                print(f"    ⚠️  Skipped {md_file.name}: {e}")

    print(f"  ✓ {technique_count} technique guides")

    # 4. Process cryptocurrency puzzles
    print("\n4️⃣ Processing cryptocurrency puzzles...")
    crypto_file = repo_dir / "cryptocurrency_puzzles" / "datasets" / "challenge_index.json"
    crypto_count = 0
    if crypto_file.exists():
        with open(crypto_file, 'r') as f:
            crypto_data = json.load(f)

        # Bitcoin puzzle examples
        bitcoin_puzzle = crypto_data.get('bitcoin_puzzle_transaction', {})
        for solve in bitcoin_puzzle.get('recent_solves', [])[:5]:
            instruction = f"### Instruction:\n"
            instruction += f"[Puzzle Type: Bitcoin Private Key Search] "
            instruction += f"[Difficulty: {solve.get('difficulty_bits')}-bit]\n\n"
            instruction += f"Explain Bitcoin Puzzle #{solve.get('puzzle_id')}\n\n"
            instruction += "### Response:\n"

            response = f"**Bitcoin Puzzle #{solve.get('puzzle_id')}**\n\n"
            response += f"Search space: 2^{solve.get('difficulty_bits')} operations\n"
            response += f"Technique: {solve.get('technique', 'GPU brute force')}\n"
            response += f"Estimated time: {solve.get('solve_time_estimate', 'variable')}\n\n"
            response += f"Tools: BitCrack, KeyHunt\n"
            response += f"Address: {solve.get('address')}\n"

            all_examples.append({
                "text": instruction + response,
                "source": "cryptocurrency",
                "difficulty": "hard"
            })
            crypto_count += 1

        # Brain wallet examples
        for bw in crypto_data.get('brain_wallet_vulnerabilities', {}).get('famous_examples', [])[:4]:
            instruction = f"### Instruction:\n"
            instruction += f"[Type: Brain Wallet Security]\n\n"
            instruction += f"Analyze: \"{bw.get('passphrase')}\"\n\n"
            instruction += "### Response:\n"

            response = f"**Brain Wallet Analysis**\n\n"
            response += f"Passphrase: \"{bw.get('passphrase')}\"\n"
            response += f"Status: {bw.get('status')}\n"
            response += f"Attack time: {bw.get('attack_time')}\n\n"
            response += f"**Vulnerability**: {bw.get('lesson')}\n"

            all_examples.append({
                "text": instruction + response,
                "source": "brain_wallet",
                "difficulty": "medium"
            })
            crypto_count += 1

    print(f"  ✓ {crypto_count} cryptocurrency examples")

    # 5. Process cryptocurrency writeups
    print("\n5️⃣ Processing cryptocurrency writeups...")
    crypto_solved = repo_dir / "cryptocurrency_puzzles" / "solved"
    crypto_writeup_count = 0
    if crypto_solved.exists():
        for md_file in crypto_solved.glob("*.md"):
            try:
                parsed = parse_markdown_file(md_file)

                instruction = f"### Instruction:\n"
                instruction += f"[Type: Cryptocurrency Puzzle Solution]\n\n"
                instruction += f"Explain: {parsed['title']}\n\n"
                instruction += "### Response:\n"

                response = parsed['full_content'][:1200] + "..."

                all_examples.append({
                    "text": instruction + response,
                    "source": "crypto_writeup",
                    "difficulty": "hard"
                })
                crypto_writeup_count += 1
            except Exception as e:
                pass

    print(f"  ✓ {crypto_writeup_count} cryptocurrency writeups")

    # 6. Process steganography challenges
    print("\n6️⃣ Processing steganography challenges...")
    steg_file = repo_dir / "steganography_puzzles" / "datasets" / "challenge_index.json"
    steg_count = 0
    if steg_file.exists():
        try:
            with open(steg_file, 'r') as f:
                steg_data = json.load(f)

            for challenge in steg_data.get('challenges', [])[:5]:
                instruction = f"### Instruction:\n"
                instruction += f"[Type: Steganography Challenge]\n\n"
                instruction += f"Explain steganography techniques\n\n"
                instruction += "### Response:\n"

                response = f"**Steganography Techniques**\n\n"
                response += "Common methods:\n"
                response += "- LSB (Least Significant Bit) manipulation\n"
                response += "- Alpha channel hiding\n"
                response += "- Frequency domain (DCT/DFT)\n"
                response += "- Metadata embedding\n"

                all_examples.append({
                    "text": instruction + response,
                    "source": "steganography",
                    "difficulty": "medium"
                })
                steg_count += 1
        except Exception as e:
            pass

    print(f"  ✓ {steg_count} steganography examples")

    # 7. Add general cryptography examples
    print("\n7️⃣ Adding general cryptography examples...")
    general_examples = [
        {
            "text": """### Instruction:
[Type: Classical Cipher] [Difficulty: easy]

Solve: KHOOR ZRUOG (Caesar cipher)

### Response:
**Caesar Cipher Solution**

This is a Caesar cipher with shift 3:
- K → H (shift -3)
- H → E
- O → L
- O → L
- R → O

Result: HELLO WORLD

Technique: Try all 26 possible shifts (brute force)""",
            "source": "educational",
            "difficulty": "easy"
        },
        {
            "text": """### Instruction:
[Type: Cryptarithm] [Difficulty: medium]

Solve: SEND + MORE = MONEY

### Response:
**Cryptarithm Solution**

Each letter = unique digit (0-9)

Working through constraints:
- M must be 1 (carry from S+M)
- S must be 8 or 9
- Systematically test combinations

Solution: 9567 + 1085 = 10652

Mapping: S=9, E=5, N=6, D=7, M=1, O=0, R=8, Y=2""",
            "source": "educational",
            "difficulty": "medium"
        },
        {
            "text": """### Instruction:
[Type: Hash Function] [Difficulty: medium]

Explain SHA-256 usage in puzzles

### Response:
**SHA-256 in Cryptopuzzles**

SHA-256 produces 64-character hex output (256 bits)

Common puzzle uses:
1. **Hash tail extraction**: Take last N chars
2. **Private key derivation**: Hash passphrase
3. **Verification**: Prove solution correctness

Example:
SHA256("Bitcoin") = "b4056df6691f8dc72e56302ddad345d65fead3ead9299609a826e2344eb63aa4"
Last 8 chars: "eb63aa4"

Tools: Python hashlib, online calculators""",
            "source": "educational",
            "difficulty": "medium"
        }
    ]

    all_examples.extend(general_examples)
    print(f"  ✓ {len(general_examples)} educational examples")

    print(f"\n📊 TOTAL EXAMPLES: {len(all_examples)}")
    print(f"   Sources breakdown:")
    from collections import Counter
    sources = Counter(ex.get('source', 'unknown') for ex in all_examples)
    for source, count in sources.items():
        print(f"   - {source}: {count}")

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

    print(f"\n4️⃣ Dataset split:")
    print(f"  📚 Train: {len(train_data)} examples")
    print(f"  📚 Val: {len(val_data)} examples")
    print(f"  📚 Test: {len(test_data)} examples")

    # Save to volume
    dest = Path("/models/data")
    dest.mkdir(exist_ok=True, parents=True)

    print(f"\n5️⃣ Saving to volume...")

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
    print("\n✅ Comprehensive dataset created and saved!")

    print("\n📝 Sample examples:")
    print("="*60)
    for i, ex in enumerate(train_data[:2]):
        print(f"\nExample {i+1} ({ex.get('source')}):")
        print(ex['text'][:300] + "...")
    print("="*60)

    return {
        "train": len(train_data),
        "val": len(val_data),
        "test": len(test_data),
        "total": len(all_examples),
        "sources": dict(sources),
        "status": "success"
    }


@app.function(image=image, volumes={"/models": volume})
def upload_data(local_path: str):
    """
    Upload training data to Modal volume.

    Args:
        local_path: Local path to the data directory
    """
    import shutil
    from pathlib import Path

    source = Path(local_path)
    dest = Path("/models/data")  # Store data in the persistent volume
    dest.mkdir(exist_ok=True, parents=True)

    # Copy files to volume
    files_copied = []
    for file in source.rglob("*"):
        if file.is_file():
            rel_path = file.relative_to(source)
            dest_file = dest / rel_path
            dest_file.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy2(file, dest_file)
            files_copied.append(str(rel_path))
            print(f"  📄 Copied: {rel_path}")

    print(f"\n📊 Total files copied: {len(files_copied)}")

    # Debug: verify files are there before commit
    import os as debug_os
    print(f"\n📂 Contents of /models/data before commit:")
    for root, dirs, files in debug_os.walk("/models/data"):
        level = root.replace("/models/data", "").count(debug_os.sep)
        indent = " " * 2 * level
        print(f"{indent}{debug_os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")

    print(f"\n💾 Committing volume...")
    volume.commit()
    print(f"✅ Uploaded data from {local_path} to /models/data")


@app.local_entrypoint()
def main(
    command: str = "train",
    data_path: str = "processed_data",
    model_name: str = TRAINING_CONFIG["model_name"],
    prompt: str = None,
):
    """
    Main entry point for the training pipeline.

    Args:
        command: Command to run (train, test, upload)
        data_path: Path to training data
        model_name: Model to fine-tune
        prompt: Prompt for testing (if command is 'test')
    """
    if command == "process":
        print(f"🔄 Processing cryptopuzzle data in Modal...")
        result = process_and_upload_data.remote()
        print(f"✅ Data processing complete: {result}")

    elif command == "upload":
        print(f"📤 Uploading data from {data_path}...")
        upload_data.remote(data_path)

    elif command == "train":
        print(f"🚀 Starting training job...")
        result = train_model.remote(
            data_path="/models/data",
            model_name=model_name,
            use_wandb=False,
        )
        print(f"✅ Training complete: {result}")

    elif command == "test":
        if prompt is None:
            prompt = "### Instruction:\nSolve this cryptographic puzzle (classical cipher) [Difficulty: easy]:\n\nKHOOR ZRUOG\n\nHint: Try different shift values\n\n### Response:\n"

        print(f"🧪 Testing model with prompt: {prompt}")
        response = test_model.remote(prompt)
        print(f"\n📝 Model response:\n{response}")

    else:
        print(f"Unknown command: {command}")
        print("Available commands: process, upload, train, test")
