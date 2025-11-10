"""
Comprehensive data processing for cryptopuzzle training dataset.
Processes ARweave puzzles, cryptocurrency puzzles, and creates instruction-following format.
"""

import json
import subprocess
from pathlib import Path
from datasets import Dataset
from typing import List, Dict
import random


def format_instruction_prompt(puzzle_data: Dict) -> str:
    """
    Format puzzle data into instruction-following format for LLM training.
    Uses ### Instruction / ### Response format.
    """
    puzzle_type = puzzle_data.get('type', 'general')
    difficulty = puzzle_data.get('difficulty', 'medium')
    problem = puzzle_data.get('problem', '')

    # Build detailed instruction with context
    instruction = f"### Instruction:\n"
    instruction += f"[Puzzle Type: {puzzle_type}] [Difficulty: {difficulty}]\n\n"
    instruction += f"{problem}\n\n"

    # Add sub-problems if they exist
    if 'sub_problems' in puzzle_data:
        instruction += "Sub-problems to solve:\n"
        for i, sub in enumerate(puzzle_data['sub_problems'], 1):
            sub_type = sub.get('type', 'general')
            question = sub.get('question', '')
            instruction += f"{i}. [{sub_type}] {question}\n"

    instruction += "\n### Response:\n"
    return instruction


def format_solution_with_reasoning(puzzle_data: Dict) -> str:
    """
    Format solution with step-by-step reasoning (chain-of-thought).
    """
    response = ""

    # Add reasoning for each sub-problem if available
    if 'sub_problems' in puzzle_data:
        response += "Let me solve each sub-problem step by step:\n\n"

        for i, sub in enumerate(puzzle_data['sub_problems'], 1):
            sub_type = sub.get('type', 'general')
            solution = sub.get('solution', '')
            technique = sub.get('technique', '')

            response += f"**Problem {i}** ({sub_type}):\n"

            # Add reasoning steps if available
            if 'steps' in sub:
                response += "Reasoning:\n"
                for step in sub['steps']:
                    response += f"- {step}\n"
            elif 'explanation' in sub:
                response += f"Explanation: {sub['explanation']}\n"

            response += f"Solution: {solution}\n"

            if technique:
                response += f"Technique used: {technique}\n"

            response += "\n"

    # Add final solution
    final_solution = puzzle_data.get('final_solution', '')
    if final_solution:
        response += f"**Final Answer**: {final_solution}\n"

    # Add key skills/techniques summary
    if 'key_skills' in puzzle_data:
        response += f"\nKey skills required: {', '.join(puzzle_data['key_skills'])}\n"

    return response


def process_arweave_training_dataset(data_dir: Path) -> List[Dict]:
    """Process the curated training_dataset.json file."""
    training_file = data_dir / "data" / "training_dataset.json"

    if not training_file.exists():
        print(f"⚠️  Training dataset not found at {training_file}")
        return []

    with open(training_file, 'r') as f:
        dataset = json.load(f)

    examples = []
    for puzzle in dataset.get('training_examples', []):
        instruction = format_instruction_prompt(puzzle)
        response = format_solution_with_reasoning(puzzle)

        examples.append({
            "text": instruction + response,
            "puzzle_id": puzzle.get('puzzle_id'),
            "puzzle_type": puzzle.get('type'),
            "difficulty": puzzle.get('difficulty'),
            "techniques": puzzle.get('key_skills', [])
        })

    print(f"✓ Processed {len(examples)} ARweave training examples")
    return examples


def process_cryptocurrency_puzzles(data_dir: Path) -> List[Dict]:
    """Process cryptocurrency puzzle data."""
    crypto_file = data_dir / "cryptocurrency_puzzles" / "datasets" / "challenge_index.json"

    if not crypto_file.exists():
        print(f"⚠️  Cryptocurrency dataset not found")
        return []

    with open(crypto_file, 'r') as f:
        crypto_data = json.load(f)

    examples = []

    # Bitcoin puzzle examples
    bitcoin_puzzle = crypto_data.get('bitcoin_puzzle_transaction', {})
    recent_solves = bitcoin_puzzle.get('recent_solves', [])

    for solve in recent_solves[:5]:  # Use top 5 recent solves as examples
        puzzle_id = solve.get('puzzle_id')
        difficulty_bits = solve.get('difficulty_bits')
        technique = solve.get('technique', 'GPU brute force')

        instruction = f"### Instruction:\n"
        instruction += f"[Puzzle Type: Bitcoin Private Key Search] [Difficulty: {difficulty_bits}-bit]\n\n"
        instruction += f"Explain how to solve Bitcoin Puzzle #{puzzle_id} which requires finding a private key in a {difficulty_bits}-bit search space.\n"
        instruction += f"The target address is: {solve.get('address')}\n\n"
        instruction += "### Response:\n"

        response = f"To solve Bitcoin Puzzle #{puzzle_id}:\n\n"
        response += f"**Problem**: Find private key in 2^{difficulty_bits} search space\n"
        response += f"**Technique**: {technique}\n\n"
        response += f"**Approach**:\n"
        response += f"1. The puzzle uses sequential bit ranges (private keys in range 2^{difficulty_bits-1} to 2^{difficulty_bits}-1)\n"
        response += f"2. For {difficulty_bits}-bit puzzles, brute force search is computationally intensive\n"
        response += f"3. Use optimized GPU tools like BitCrack for parallel search\n"
        response += f"4. Estimated time: {solve.get('solve_time_estimate', 'variable')} with modern GPU clusters\n"

        if solve.get('notes'):
            response += f"\n**Important Note**: {solve['notes']}\n"

        examples.append({
            "text": instruction + response,
            "puzzle_type": "cryptocurrency_brute_force",
            "difficulty": "hard",
            "techniques": ["gpu_optimization", "brute_force", "bitcoin_cryptography"]
        })

    # Brain wallet examples
    brain_wallets = crypto_data.get('brain_wallet_vulnerabilities', {})
    famous_examples = brain_wallets.get('famous_examples', [])

    for bw in famous_examples[:3]:
        instruction = f"### Instruction:\n"
        instruction += f"[Puzzle Type: Brain Wallet Security Analysis] [Difficulty: medium]\n\n"
        instruction += f"Analyze the security of the brain wallet passphrase: \"{bw.get('passphrase')}\"\n"
        instruction += f"Explain why this brain wallet was compromised and the attack methodology.\n\n"
        instruction += "### Response:\n"

        response = f"**Brain Wallet Analysis**: \"{bw.get('passphrase')}\"\n\n"
        response += f"**Status**: {bw.get('status')}\n"
        response += f"**Attack Time**: {bw.get('attack_time')}\n\n"
        response += f"**Vulnerability Explanation**:\n"
        response += f"{bw.get('lesson')}\n\n"
        response += f"**Attack Method**:\n"
        response += f"1. Brain wallets derive private keys from passphrases using SHA-256\n"
        response += f"2. Tools like Brainflayer can test billions of passphrases per second\n"
        response += f"3. This phrase appears in common dictionaries (quotes, lyrics, famous text)\n"
        response += f"4. Attackers maintain databases of known phrases and monitor the blockchain\n"
        response += f"5. Any funds sent to this address are immediately swept\n\n"
        response += f"**Key Lesson**: Never use human-readable or famous phrases for cryptocurrency wallets.\n"

        examples.append({
            "text": instruction + response,
            "puzzle_type": "cryptocurrency_brain_wallet",
            "difficulty": "medium",
            "techniques": ["cryptanalysis", "dictionary_attacks", "security_analysis"]
        })

    # Pollard's Kangaroo algorithm example
    kangaroo_info = crypto_data.get('techniques_taxonomy', {}).get('pollard_kangaroo', {})
    instruction = f"### Instruction:\n"
    instruction += f"[Puzzle Type: Cryptographic Algorithm] [Difficulty: hard]\n\n"
    instruction += f"Explain Pollard's Kangaroo algorithm and when it's applicable for solving Bitcoin puzzle transactions.\n\n"
    instruction += "### Response:\n"

    response = f"**Pollard's Kangaroo Algorithm**\n\n"
    response += f"**Complexity**: {kangaroo_info.get('complexity')}\n\n"
    response += f"**Requirements**:\n"
    response += f"- Known public key (revealed through previous transactions)\n"
    response += f"- Known search range\n\n"
    response += f"**How It Works**:\n"
    response += f"1. Uses two 'kangaroos' (tame and wild) that make pseudo-random jumps on elliptic curve\n"
    response += f"2. Tame kangaroo starts from known point, wild from target public key\n"
    response += f"3. Both leave trail of 'distinguished points'\n"
    response += f"4. When trails collide, private key can be computed\n"
    response += f"5. Square root speedup vs brute force: O(√N) instead of O(N)\n\n"
    response += f"**Applicable Puzzles**: {kangaroo_info.get('applicable_puzzles')}\n\n"
    response += f"**Example**: For puzzle #135 with exposed public key:\n"
    response += f"- Brute force: 2^135 operations (impossible)\n"
    response += f"- Kangaroo: 2^67.5 operations (challenging but feasible)\n\n"
    response += f"**Tools**: JeanLucPons/Kangaroo, KeyHunt\n"

    examples.append({
        "text": instruction + response,
        "puzzle_type": "cryptographic_algorithm",
        "difficulty": "hard",
        "techniques": ["ecdlp", "pollards_kangaroo", "cryptography"]
    })

    print(f"✓ Processed {len(examples)} cryptocurrency puzzle examples")
    return examples


def process_puzzle_markdown_files(data_dir: Path) -> List[Dict]:
    """Process markdown puzzle writeups for additional training data."""
    solved_dir = data_dir / "puzzles" / "solved"

    if not solved_dir.exists():
        print(f"⚠️  Solved puzzles directory not found")
        return []

    examples = []

    # We'll add a few synthetic examples based on the structure we saw
    # This can be expanded to parse actual markdown files

    print(f"✓ Processed {len(examples)} puzzle markdown examples")
    return examples


def create_technique_examples() -> List[Dict]:
    """Create examples teaching specific puzzle-solving techniques."""
    examples = []

    # Cryptarithm example
    instruction = """### Instruction:
[Puzzle Type: Cryptarithm] [Difficulty: medium]

Solve the cryptarithm: AR + PAPER = PIZZA
Where each letter represents a unique digit (0-9), and the same letter always represents the same digit.

### Response:
"""

    response = """Let me solve this cryptarithm step by step:

**Step 1**: Analyze the structure
- AR is 2 digits
- PAPER is 5 digits
- PIZZA is 5 digits
- This means P must be 9 (since AR + PAPER produces 5 digits)

**Step 2**: Set P = 9
- AR + 9A9ER = 9IZZA

**Step 3**: Analyze the leftmost column
- A + 9 (plus possible carry) = 9 (or 19)
- This means A must be 0 or there's a carry making it 19
- But if A=0, then 0R + 909ER can't produce 9IZZA with leading 9
- So there must be a carry: A + 9 + 1 = 19, meaning A = 9... but P=9
- Actually, let's work from right to left with carries

**Step 4**: Try systematic search with constraints
After testing combinations:
- A = 6
- R = 8
- P = 9
- E = 3
- I = 7
- Z = 0

**Verification**: 68 + 96938 = 97006 ✓

**Final Answer**: 68 + 96938 = 97006

Technique used: constraint satisfaction with systematic search
"""

    examples.append({
        "text": instruction + response,
        "puzzle_type": "cryptarithm",
        "difficulty": "medium",
        "techniques": ["constraint_satisfaction", "logical_deduction"]
    })

    # SHA-256 hashing technique
    instruction = """### Instruction:
[Puzzle Type: Cryptographic Hashing] [Difficulty: medium]

Explain how to use SHA-256 hashing to create puzzle solutions, specifically extracting the last 8 hexadecimal characters from a hash.

### Response:
"""

    response = """**SHA-256 Hash Extraction Technique**

**Concept**: SHA-256 produces a 64-character hexadecimal string. We can extract portions for puzzle solutions.

**Example Process**:
1. Take input text: "Bitcoin Genesis Block"
2. Compute SHA-256 hash
3. Extract last 8 hex characters (the "tail")

**Python Implementation**:
```python
import hashlib

def get_hash_tail(text: str, length: int = 8) -> str:
    hash_bytes = hashlib.sha256(text.encode('utf-8')).digest()
    hash_hex = hash_bytes.hex()
    return hash_hex[-length:]

# Example
result = get_hash_tail("Bitcoin Genesis Block")
# Returns: "c3334fc9"
```

**Application in Puzzles**:
- Used in ARweave Puzzle #13
- Multiple images identified, names hashed, tails concatenated
- Result forms an Ethereum private key

**Key Points**:
- Deterministic: same input always produces same hash
- One-way: can't reverse from hash to input
- Used for verifiable puzzle solutions
- Common in cryptographic treasure hunts

Technique: cryptographic_hashing, hex_manipulation
"""

    examples.append({
        "text": instruction + response,
        "puzzle_type": "cryptographic_technique",
        "difficulty": "medium",
        "techniques": ["sha256", "hashing", "hex_manipulation"]
    })

    print(f"✓ Created {len(examples)} technique teaching examples")
    return examples


def augment_data_with_variations(examples: List[Dict]) -> List[Dict]:
    """Create variations of examples for better generalization."""
    augmented = examples.copy()

    # Add difficulty-focused variants
    for ex in examples[:3]:  # Augment a few examples
        # Create a variant that asks about difficulty
        original_text = ex['text']
        if '### Instruction:' in original_text and '### Response:' in original_text:
            inst_end = original_text.index('### Response:')
            instruction = original_text[:inst_end]

            # Add difficulty analysis question
            new_instruction = instruction.replace('### Response:', '')
            new_instruction += "\nAlso explain: What makes this puzzle this difficulty level?\n\n### Response:\n"

            # Keep original response and add difficulty analysis
            response = original_text[inst_end + len('### Response:'):]
            difficulty = ex.get('difficulty', 'medium')
            response += f"\n**Difficulty Analysis**: This is a {difficulty} puzzle because of the combination of techniques required and the knowledge domains involved."

            augmented.append({
                "text": new_instruction + response,
                "puzzle_type": ex.get('puzzle_type'),
                "difficulty": ex.get('difficulty'),
                "techniques": ex.get('techniques', [])
            })

    print(f"✓ Augmented dataset: {len(examples)} → {len(augmented)} examples")
    return augmented


def main():
    """Main data processing pipeline."""
    print("🔄 Starting comprehensive cryptopuzzle data processing...")

    # Clone repository
    repo_dir = Path("/tmp/cryptopuzzles")
    if not repo_dir.exists():
        print("📥 Cloning cryptopuzzles repository...")
        subprocess.run([
            "git", "clone",
            "https://github.com/FMLBeast/cryptopuzzles.git",
            str(repo_dir)
        ], check=True)

    # Process all data sources
    all_examples = []

    # 1. ARweave curated training dataset
    all_examples.extend(process_arweave_training_dataset(repo_dir))

    # 2. Cryptocurrency puzzles
    all_examples.extend(process_cryptocurrency_puzzles(repo_dir))

    # 3. Technique teaching examples
    all_examples.extend(create_technique_examples())

    # 4. Augment with variations
    all_examples = augment_data_with_variations(all_examples)

    print(f"\n📊 Total examples created: {len(all_examples)}")

    # Split into train/val/test
    random.seed(42)
    random.shuffle(all_examples)

    n = len(all_examples)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)

    train_data = all_examples[:train_size]
    val_data = all_examples[train_size:train_size + val_size]
    test_data = all_examples[train_size + val_size:]

    print(f"  📚 Train: {len(train_data)} examples")
    print(f"  📚 Val: {len(val_data)} examples")
    print(f"  📚 Test: {len(test_data)} examples")

    # Save datasets
    output_dir = Path("/tmp/processed_data")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save as JSON
    with open(output_dir / "train.json", 'w') as f:
        json.dump(train_data, f, indent=2)
    with open(output_dir / "validation.json", 'w') as f:
        json.dump(val_data, f, indent=2)
    with open(output_dir / "test.json", 'w') as f:
        json.dump(test_data, f, indent=2)

    # Save as HuggingFace datasets
    Dataset.from_list(train_data).save_to_disk(str(output_dir / "train_dataset"))
    Dataset.from_list(val_data).save_to_disk(str(output_dir / "val_dataset"))
    Dataset.from_list(test_data).save_to_disk(str(output_dir / "test_dataset"))

    print(f"\n✅ Data processing complete! Saved to {output_dir}/")

    # Print sample
    print("\n📝 Sample training example:")
    print("="*80)
    print(train_data[0]['text'][:500] + "...")
    print("="*80)


if __name__ == "__main__":
    main()
