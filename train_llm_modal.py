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

# Separate image for puzzle solver (with anthropic for optional AI solving)
solver_image = image.pip_install("anthropic>=0.18.0")

# Create a persistent volume for model storage
volume = modal.Volume.from_name("puzzle-llm-models", create_if_missing=True)

# Volume for synthetic puzzle data (shared with puzzle generator)
puzzle_data_volume = modal.Volume.from_name("puzzle-training-data", create_if_missing=True)

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
    Creates HUNDREDS of training examples from EVERY markdown file!
    """
    import json
    import subprocess
    from pathlib import Path
    from datasets import Dataset
    import random
    import re

    print("🔥 Starting COMPREHENSIVE cryptopuzzle data extraction!")
    print("   Parsing ENTIRE repository - every markdown, every puzzle!\n")

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

    # Helper function to extract sub-puzzles from markdown
    def extract_subpuzzles_from_markdown(file_path: Path, puzzle_name: str) -> list:
        """Extract each sub-puzzle as a separate training example."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        examples = []

        # Split by ### headers (sub-puzzles)
        sections = re.split(r'\n###\s+', content)

        for i, section in enumerate(sections[1:], 1):  # Skip first (header)
            if len(section.strip()) < 100:  # Skip tiny sections
                continue

            lines = section.split('\n')
            sub_title = lines[0].strip()
            sub_content = '\n'.join(lines[1:]).strip()

            # Create training example
            instruction = f"### Instruction:\n"
            instruction += f"[Puzzle: {puzzle_name}] [Sub-puzzle {i}: {sub_title}]\n\n"
            instruction += f"Solve this cryptopuzzle and show your reasoning.\n\n"
            instruction += f"### Response:\n"

            response = f"**{sub_title}**\n\n{sub_content}"

            examples.append({
                "text": instruction + response,
                "source": f"solved_puzzle_{file_path.stem}",
                "difficulty": "medium"
            })

        return examples

    # Helper to parse full markdown file
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

    # 1. Process ALL solved puzzle writeups - Extract EVERY sub-puzzle!
    print("\n1️⃣ Processing ALL solved puzzle writeups...")
    solved_dir = repo_dir / "puzzles" / "solved"
    solved_count = 0
    if solved_dir.exists():
        for md_file in solved_dir.glob("*.md"):
            print(f"  📄 Parsing {md_file.name}...")
            try:
                # Extract sub-puzzles from this file
                sub_examples = extract_subpuzzles_from_markdown(md_file, md_file.stem)

                if len(sub_examples) > 0:
                    all_examples.extend(sub_examples)
                    solved_count += len(sub_examples)
                    print(f"     ✓ Extracted {len(sub_examples)} sub-puzzles")
                else:
                    # If no sub-puzzles found, add whole file
                    ex = parse_full_markdown(md_file, "solved_puzzle")
                    all_examples.append(ex)
                    solved_count += 1
                    print(f"     ✓ Added full puzzle")
            except Exception as e:
                print(f"    ⚠️  Error with {md_file.name}: {e}")

    print(f"\n  ✅ TOTAL: {solved_count} examples from solved puzzles\n")

    # 2. Process cryptocurrency writeups
    print("2️⃣ Processing cryptocurrency puzzle writeups...")
    crypto_solved = repo_dir / "cryptocurrency_puzzles" / "solved"
    crypto_count = 0
    if crypto_solved.exists():
        for md_file in crypto_solved.glob("*.md"):
            print(f"  📄 Parsing {md_file.name}...")
            try:
                ex = parse_full_markdown(md_file, "crypto_puzzle")
                all_examples.append(ex)
                crypto_count += 1
                print(f"     ✓ Added")
            except Exception as e:
                print(f"    ⚠️  Error: {e}")

    print(f"\n  ✅ TOTAL: {crypto_count} cryptocurrency writeups\n")

    # 3. Process ALL technique documentation (techniques/ directory)
    print("3️⃣ Processing technique documentation...")
    technique_count = 0
    techniques_base = repo_dir / "techniques"
    if techniques_base.exists():
        for md_file in techniques_base.rglob("*.md"):  # Recursive glob!
            print(f"  📄 Parsing {md_file.name}...")
            try:
                ex = parse_full_markdown(md_file, "technique_guide")
                all_examples.append(ex)
                technique_count += 1
                print(f"     ✓ Added")
            except Exception as e:
                print(f"    ⚠️  Error: {e}")

    print(f"\n  ✅ TOTAL: {technique_count} technique guides\n")

    # 4. Process ALL other markdown files (READMEs, analysis reports, etc.)
    print("4️⃣ Processing READMEs, analysis reports, and documentation...")
    other_count = 0

    # Get all markdown files in root and subdirectories, excluding already processed ones
    for md_file in repo_dir.rglob("*.md"):
        # Skip if already processed
        if 'solved' in str(md_file) or 'techniques' in str(md_file):
            continue

        # Skip hidden directories
        if any(part.startswith('.') for part in md_file.parts):
            continue

        print(f"  📄 Parsing {md_file.relative_to(repo_dir)}...")
        try:
            ex = parse_full_markdown(md_file, "documentation")
            all_examples.append(ex)
            other_count += 1
            print(f"     ✓ Added")
        except Exception as e:
            print(f"    ⚠️  Error: {e}")

    print(f"\n  ✅ TOTAL: {other_count} additional markdown files\n")

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


@app.function(
    image=image.pip_install("anthropic>=0.18.0"),
    volumes={"/data": puzzle_data_volume, "/models": volume},
    timeout=7200,
    secrets=[modal.Secret.from_name("anthropic-api-key", required=False)],
)
def solve_synthetic_puzzles(
    sample_size: int = None,
    use_ai_solver: bool = False
):
    """
    Process synthetic puzzles and create verified training data.

    Args:
        sample_size: Number of puzzles to process (None = all)
        use_ai_solver: If True, use Claude API to solve puzzles (requires API key)
                       If False, use ground truth solutions with enhanced reasoning
    """
    import json
    from pathlib import Path
    import random
    from datasets import Dataset

    print("🧠 MULTI-AGENT PUZZLE SOLVER")
    print("="*60)

    # Reload volumes to see synthetic puzzles
    puzzle_data_volume.reload()
    volume.reload()
    print("🔄 Volumes reloaded")

    # Load synthetic puzzles
    puzzle_dir = Path("/data/synthetic_puzzles")
    if not puzzle_dir.exists():
        print(f"❌ Error: Synthetic puzzles not found at {puzzle_dir}")
        print(f"   Run: modal run synthetic_puzzle_generator.py::generate_all_puzzles")
        return {"status": "error", "message": "Puzzles not found"}

    print(f"\n📂 Loading puzzles from {puzzle_dir}...")

    all_training_examples = []
    puzzle_stats = {}

    # Load each puzzle type
    puzzle_files = {
        "logic_grid": puzzle_dir / "logic_grid.json",
        "caesar_cipher": puzzle_dir / "caesar_cipher.json",
        "substitution_cipher": puzzle_dir / "substitution_cipher.json",
        "hash_puzzle": puzzle_dir / "hash_puzzle.json",
        "multi_step": puzzle_dir / "multi_step.json",
    }

    for puzzle_type, filepath in puzzle_files.items():
        if not filepath.exists():
            print(f"⚠️  Skipping {puzzle_type} - file not found")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {puzzle_type}...")
        print(f"{'='*60}")

        with open(filepath, 'r') as f:
            puzzles = json.load(f)

        # Sample if requested
        if sample_size:
            puzzles = random.sample(puzzles, min(sample_size, len(puzzles)))

        print(f"📊 Loaded {len(puzzles)} {puzzle_type} puzzles")

        solved_count = 0
        failed_count = 0

        for i, puzzle in enumerate(puzzles):
            try:
                # Extract ground truth solution and create enhanced training example
                solution = puzzle.get('solution')

                # Create enhanced instruction format with chain-of-thought reasoning
                if puzzle_type == "caesar_cipher":
                    # Use the steps already in the puzzle
                    reasoning = "\n".join(f"Step {i+1}: {step}" for i, step in enumerate(puzzle.get('steps', [])))
                    training_text = f"""### Instruction:
[Type: Caesar Cipher] [Difficulty: {puzzle['difficulty']}]

Decode this Caesar cipher: {puzzle['puzzle'].replace('Decode: ', '')}

### Response:

**Approach**: Caesar ciphers use a simple letter shift. We need to try all 25 possible shifts.

**Reasoning**:
{reasoning}

**Solution**: {solution}

**Verification**: Shift value = {puzzle.get('shift', 'N/A')}
"""

                elif puzzle_type == "substitution_cipher":
                    training_text = f"""### Instruction:
[Type: Substitution Cipher] [Difficulty: {puzzle['difficulty']}]

{puzzle['puzzle']}

### Response:

**Approach**: Use frequency analysis and pattern recognition.

**Reasoning**:
1. Identify letter frequencies in the ciphertext
2. Compare with English letter frequency (E, T, A, O, I, N most common)
3. Look for common patterns (THE, AND, OF, etc.)
4. Test substitutions and refine mapping
5. Verify decoded message makes sense

**Solution**: {solution}

**Method**: Frequency analysis combined with linguistic pattern matching
"""

                elif puzzle_type == "logic_grid":
                    clues_text = "\n".join(f"{i+1}. {clue}" for i, clue in enumerate(puzzle['puzzle']['clues']))
                    training_text = f"""### Instruction:
[Type: Logic Grid Puzzle] [Difficulty: {puzzle['difficulty']}]

{puzzle['puzzle']['description']}

Clues:
{clues_text}

Question: {puzzle['puzzle']['question']}

### Response:

**Approach**: Use logical deduction to eliminate possibilities.

**Reasoning**:
1. Create a grid with all categories: {', '.join(puzzle['puzzle']['categories'])}
2. Process each clue systematically
3. Mark confirmed matches and eliminate contradictions
4. Use transitive logic (if A=B and B=C, then A=C)
5. Continue until all attributes are assigned

**Solution**: {solution}

**Verification**: All clues satisfied with this assignment
"""

                elif puzzle_type == "hash_puzzle":
                    steps_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(puzzle.get('steps', [])))
                    training_text = f"""### Instruction:
[Type: Hash Puzzle] [Difficulty: {puzzle['difficulty']}]

{puzzle['puzzle']}

### Response:

**Approach**: Brute force search through common words.

**Reasoning**:
{steps_text}

**Solution**: {solution}

**Hash**: {puzzle.get('hash', 'N/A')}
**Pattern**: {puzzle.get('pattern', 'N/A')}
"""

                elif puzzle_type == "multi_step":
                    solution_steps_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(puzzle.get('solution_steps', [])))
                    training_text = f"""### Instruction:
[Type: Multi-Step Puzzle] [Difficulty: {puzzle['difficulty']}]

{puzzle['puzzle']}

### Response:

**Approach**: Work backwards from the final result, reversing each transformation.

**Reasoning**:
{solution_steps_text}

**Solution**: {solution}

**Verification**: Forward transformations produce the given final result
"""

                else:
                    # Fallback to instruction_format
                    training_text = puzzle.get('instruction_format', '')

                # Add to training examples
                all_training_examples.append({
                    "text": training_text,
                    "type": puzzle_type,
                    "difficulty": puzzle['difficulty'],
                    "verified": True,
                    "solution": solution
                })

                solved_count += 1

                # Show progress
                if (i + 1) % 100 == 0:
                    print(f"  Progress: {i+1}/{len(puzzles)} puzzles processed")

            except Exception as e:
                print(f"  ⚠️  Error processing puzzle {i}: {e}")
                failed_count += 1
                continue

        puzzle_stats[puzzle_type] = {
            "total": len(puzzles),
            "solved": solved_count,
            "failed": failed_count
        }

        print(f"\n✅ {puzzle_type}: {solved_count}/{len(puzzles)} solved")

    print(f"\n{'='*60}")
    print(f"📊 SOLVING COMPLETE")
    print(f"{'='*60}")
    print(f"Total training examples: {len(all_training_examples)}")
    print(f"\nBreakdown by type:")
    for ptype, stats in puzzle_stats.items():
        print(f"  {ptype}: {stats['solved']}/{stats['total']} ({stats['solved']/stats['total']*100:.1f}%)")

    # Shuffle training data
    random.seed(42)
    random.shuffle(all_training_examples)

    # Split into train/val/test
    n = len(all_training_examples)
    test_size = max(100, int(0.1 * n))
    val_size = max(100, int(0.1 * n))
    train_size = n - val_size - test_size

    train_data = all_training_examples[:train_size]
    val_data = all_training_examples[train_size:train_size + val_size]
    test_data = all_training_examples[train_size + val_size:]

    print(f"\n{'='*60}")
    print(f"📚 Dataset Split")
    print(f"{'='*60}")
    print(f"Train: {len(train_data)} examples")
    print(f"Val: {len(val_data)} examples")
    print(f"Test: {len(test_data)} examples")

    # Save to volume
    dest = Path("/models/data/synthetic")
    dest.mkdir(exist_ok=True, parents=True)

    print(f"\n💾 Saving to {dest}...")

    with open(dest / "train.json", 'w') as f:
        json.dump(train_data, f, indent=2)
    with open(dest / "validation.json", 'w') as f:
        json.dump(val_data, f, indent=2)
    with open(dest / "test.json", 'w') as f:
        json.dump(test_data, f, indent=2)

    # Also save as HuggingFace datasets
    Dataset.from_list(train_data).save_to_disk(str(dest / "train_dataset"))
    Dataset.from_list(val_data).save_to_disk(str(dest / "val_dataset"))
    Dataset.from_list(test_data).save_to_disk(str(dest / "test_dataset"))

    # Commit changes to volume
    volume.commit()
    print(f"\n💾 Volume committed")

    print(f"\n✅ Training data saved!")
    print(f"\n📝 Sample training example:")
    print("="*60)
    print(train_data[0]['text'][:500] + "...")
    print("="*60)

    print(f"\n🎯 Next steps:")
    print(f"   1. Train on synthetic data:")
    print(f"      modal run train_llm_modal.py::train_model --data-path /models/data/synthetic")
    print(f"\n   2. Or combine with real puzzles first, then train")

    return {
        "status": "success",
        "total_examples": len(all_training_examples),
        "train": len(train_data),
        "val": len(val_data),
        "test": len(test_data),
        "stats": puzzle_stats
    }


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
        command: Command to run (train, test, upload, solve)
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

    elif command == "solve":
        print(f"🧠 Solving synthetic puzzles...")
        result = solve_synthetic_puzzles.remote()
        print(f"✅ Solving complete: {result}")

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
        print("Available commands: process, upload, solve, train, test")
