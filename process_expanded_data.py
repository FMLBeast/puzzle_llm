#!/usr/bin/env python3
"""
Enhanced processor that includes smart contract puzzles.
Processes all puzzle types from the expanded cryptopuzzles repository.
"""

import json
from pathlib import Path
from datasets import Dataset
import re


def process_arweave_puzzles(base_path="cryptopuzzles_data/data"):
    """Process ARweave cryptopuzzles from training_dataset.json."""
    print("📦 Processing ARweave puzzles...")

    with open(f"{base_path}/training_dataset.json", 'r') as f:
        data = json.load(f)

    formatted_data = []
    for example in data['training_examples']:
        puzzle_id = example['puzzle_id']
        difficulty = example['difficulty']
        puzzle_type = example['type']

        if 'sub_problems' in example:
            for i, sub_problem in enumerate(example['sub_problems']):
                instruction = f"Solve this cryptographic puzzle (ARweave Puzzle #{puzzle_id}, Type: {puzzle_type}, Difficulty: {difficulty}):\n\n"
                instruction += sub_problem.get('question', sub_problem.get('type', ''))

                if 'hint' in sub_problem:
                    instruction += f"\n\nHint: {sub_problem['hint']}"

                response = f"**Solution**: {sub_problem.get('solution', 'No solution available')}"

                if 'explanation' in sub_problem:
                    response += f"\n\n**Explanation**: {sub_problem['explanation']}"

                if 'technique' in sub_problem:
                    response += f"\n\n**Technique**: {sub_problem['technique']}"

                if 'steps' in sub_problem:
                    response += "\n\n**Steps**:\n"
                    for step in sub_problem['steps']:
                        response += f"- {step}\n"

                formatted_data.append({
                    "instruction": instruction,
                    "input": "",
                    "output": response,
                    "text": f"### Instruction:\n{instruction}\n\n### Response:\n{response}",
                    "source": "arweave",
                    "puzzle_id": str(puzzle_id),
                    "difficulty": str(difficulty),
                    "puzzle_type": str(puzzle_type),
                    "technique": str(sub_problem.get('technique', '')),
                })

    print(f"  ✅ Extracted {len(formatted_data)} ARweave examples")
    return formatted_data


def process_cryptocurrency_puzzles(base_path="cryptopuzzles_data/cryptocurrency_puzzles"):
    """Process cryptocurrency private key hunting puzzles."""
    print("🔐 Processing cryptocurrency puzzles...")

    with open(f"{base_path}/datasets/challenge_index.json", 'r') as f:
        data = json.load(f)

    formatted_data = []

    # Process recent solves
    if 'bitcoin_puzzle_transaction' in data and 'recent_solves' in data['bitcoin_puzzle_transaction']:
        for solve in data['bitcoin_puzzle_transaction']['recent_solves']:
            instruction = f"Analyze this solved Bitcoin puzzle (Difficulty: {solve['difficulty_bits']} bits):\n\n"
            instruction += f"**Puzzle #{solve['puzzle_id']}**\n"
            instruction += f"- Address: {solve['address']}\n"
            instruction += f"- Amount: {solve['amount_btc']} BTC\n"
            instruction += f"- Search space: {solve['search_space']}\n"
            instruction += f"- Solve date: {solve['solve_date']}\n\n"
            instruction += "What technique was used and why was it successful?"

            response = f"**Technique Used**: {solve['technique']}\n\n"
            response += f"**Search Space**: {solve['search_space']} possible keys\n\n"

            if 'solve_time_estimate' in solve:
                response += f"**Time to Solve**: {solve['solve_time_estimate']}\n\n"

            if 'notes' in solve:
                response += f"**Key Insights**: {solve['notes']}\n\n"

            response += "**Why It Worked**: The puzzle's relatively small key space (compared to full 256-bit) made exhaustive GPU brute-force search feasible within a reasonable timeframe and cost."

            formatted_data.append({
                "instruction": instruction,
                "input": "",
                "output": response,
                "text": f"### Instruction:\n{instruction}\n\n### Response:\n{response}",
                "source": "cryptocurrency",
                "puzzle_id": str(solve['puzzle_id']),
                "difficulty": str(solve['difficulty_bits']),
                "puzzle_type": "bitcoin_puzzle",
                "technique": str(solve['technique']),
            })

    print(f"  ✅ Extracted {len(formatted_data)} cryptocurrency examples")
    return formatted_data


def process_smart_contract_puzzles(base_path="cryptopuzzles_data/smart_contract_puzzles"):
    """Process smart contract puzzle data."""
    print("⛓️  Processing smart contract puzzles...")

    readme_file = f"{base_path}/README.md"
    if not Path(readme_file).exists():
        print("  ⚠️  Smart contract README not found, skipping...")
        return []

    formatted_data = []

    # Add general smart contract puzzle concept
    instruction = "Explain how Curta CTF smart contract puzzles work and what makes them unique:\n\n"
    instruction += "How does the generative puzzle system prevent solution copying?"

    response = "**Curta CTF Overview**:\n\n"
    response += "Curta is an on-chain CTF platform where puzzles are implemented as Ethereum smart contracts.\n\n"
    response += "**Key Innovation - Generative Puzzles**:\n"
    response += "- Each puzzle implements `generate(address _seed)` which takes the solver's address as input\n"
    response += "- Returns a unique starting position for that specific solver\n"
    response += "- This means every solver gets a DIFFERENT puzzle instance\n"
    response += "- Makes copying solutions impossible - each address gets unique parameters\n\n"
    response += "**Solving Process**:\n"
    response += "1. Call `generate(your_address)` to get your unique puzzle state\n"
    response += "2. Solve the cryptographic/logical challenge for YOUR instance\n"
    response += "3. Submit via `verify(starting_position, solution)`\n"
    response += "4. If correct: Receive NFT reward and leaderboard placement\n\n"
    response += "**Security**: The generative approach ensures fairness - no solver has an advantage by seeing others' solutions."

    formatted_data.append({
        "instruction": instruction,
        "input": "",
        "output": response,
        "text": f"### Instruction:\n{instruction}\n\n### Response:\n{response}",
        "source": "smart_contract",
        "puzzle_id": "curta_system",
        "difficulty": "conceptual",
        "puzzle_type": "smart_contract_ctf",
        "technique": "generative_puzzles",
    })

    print(f"  ✅ Extracted {len(formatted_data)} smart contract examples")
    return formatted_data


def process_steganography_puzzles(base_path="cryptopuzzles_data/steganography_puzzles"):
    """Process steganography challenges."""
    print("🖼️  Processing steganography puzzles...")

    formatted_data = []

    # Check for challenge index
    index_file = f"{base_path}/datasets/challenge_index.json"
    if Path(index_file).exists():
        with open(index_file, 'r') as f:
            data = json.load(f)

        if 'challenges' in data:
            for challenge in data['challenges'][:10]:
                instruction = f"Analyze this steganography challenge:\n\n"
                instruction += f"**Challenge**: {challenge.get('name', 'Unknown')}\n"
                instruction += f"**Difficulty**: {challenge.get('difficulty', 'Unknown')}\n"

                if 'description' in challenge:
                    instruction += f"**Description**: {challenge['description']}\n\n"

                instruction += "What steganography technique might be hidden in this challenge?"

                response = f"**Technique**: {challenge.get('technique', 'Various techniques possible')}\n\n"

                if 'tool' in challenge:
                    response += f"**Recommended Tool**: {challenge['tool']}\n\n"

                response += "**Analysis Approach**:\n"
                response += "1. Check file metadata and headers\n"
                response += "2. Analyze LSB (Least Significant Bit) patterns\n"
                response += "3. Look for embedded files or encrypted data\n"
                response += "4. Check for palette-based hiding\n"
                response += "5. Examine frequency domains (DCT for JPEG)"

                formatted_data.append({
                    "instruction": instruction,
                    "input": "",
                    "output": response,
                    "text": f"### Instruction:\n{instruction}\n\n### Response:\n{response}",
                    "source": "steganography",
                    "puzzle_id": str(challenge.get('id', 'unknown')),
                    "difficulty": str(challenge.get('difficulty', 'unknown')),
                    "puzzle_type": "steganography",
                    "technique": str(challenge.get('technique', 'unknown')),
                })

    # Add general steganography techniques from writeups
    instruction = "What are the key techniques for analyzing steganography challenges in CTF competitions?"

    response = "**Common Steganography Techniques in CTFs**:\n\n"
    response += "1. **LSB (Least Significant Bit) Encoding**:\n"
    response += "   - Hidden data in least significant bits of pixels\n"
    response += "   - Tools: stegsolve, zsteg, stegpy\n\n"
    response += "2. **Metadata Analysis**:\n"
    response += "   - Check EXIF data, file comments, headers\n"
    response += "   - Tools: exiftool, strings, binwalk\n\n"
    response += "3. **File Carving**:\n"
    response += "   - Extract embedded files from images\n"
    response += "   - Tools: binwalk, foremost, steghide\n\n"
    response += "4. **Frequency Domain Analysis**:\n"
    response += "   - DCT coefficients in JPEG\n"
    response += "   - Spectrograms in audio\n\n"
    response += "5. **Visual Analysis**:\n"
    response += "   - Color plane separation\n"
    response += "   - Bit plane analysis\n"
    response += "   - Pattern recognition"

    formatted_data.append({
        "instruction": instruction,
        "input": "",
        "output": response,
        "text": f"### Instruction:\n{instruction}\n\n### Response:\n{response}",
        "source": "steganography",
        "puzzle_id": "techniques_overview",
        "difficulty": "educational",
        "puzzle_type": "steganography_techniques",
        "technique": "multiple",
    })

    print(f"  ✅ Extracted {len(formatted_data)} steganography examples")
    return formatted_data


def combine_and_save(output_dir="processed_data"):
    """Combine all puzzle types and save."""
    print("\n" + "=" * 60)
    print("🔄 PROCESSING ALL PUZZLE TYPES (EXPANDED DATASET)")
    print("=" * 60 + "\n")

    all_data = []

    # Process each type
    all_data.extend(process_arweave_puzzles())
    all_data.extend(process_cryptocurrency_puzzles())
    all_data.extend(process_smart_contract_puzzles())
    all_data.extend(process_steganography_puzzles())

    print(f"\n📊 Total examples: {len(all_data)}")

    # Split data
    print(f"\n📁 Saving to {output_dir}/")
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    total = len(all_data)
    train_size = int(total * 0.8)
    val_size = int(total * 0.1)

    train_data = all_data[:train_size]
    val_data = all_data[train_size:train_size + val_size]
    test_data = all_data[train_size + val_size:]

    # Save as JSON
    with open(output_path / "train.json", 'w') as f:
        json.dump(train_data, f, indent=2)

    with open(output_path / "validation.json", 'w') as f:
        json.dump(val_data, f, indent=2)

    with open(output_path / "test.json", 'w') as f:
        json.dump(test_data, f, indent=2)

    # Save as HuggingFace datasets
    Dataset.from_list(train_data).save_to_disk(str(output_path / "train_dataset"))
    Dataset.from_list(val_data).save_to_disk(str(output_path / "val_dataset"))
    Dataset.from_list(test_data).save_to_disk(str(output_path / "test_dataset"))

    print(f"✅ Saved {len(train_data)} training examples")
    print(f"✅ Saved {len(val_data)} validation examples")
    print(f"✅ Saved {len(test_data)} test examples")

    # Show statistics
    print("\n" + "=" * 60)
    print("📊 DATASET STATISTICS")
    print("=" * 60)

    sources = {}
    difficulties = {}
    techniques = {}

    for item in all_data:
        source = item.get('source', 'unknown')
        sources[source] = sources.get(source, 0) + 1

        diff = item.get('difficulty', 'unknown')
        difficulties[diff] = difficulties.get(diff, 0) + 1

        tech = item.get('technique', 'unknown')
        if tech and tech != 'unknown':
            techniques[tech] = techniques.get(tech, 0) + 1

    print("\n📦 By Source:")
    for source, count in sorted(sources.items()):
        print(f"  - {source}: {count}")

    print("\n⚡ By Difficulty:")
    for diff, count in sorted(difficulties.items()):
        print(f"  - {diff}: {count}")

    print("\n🛠️  Top Techniques:")
    for tech, count in sorted(techniques.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  - {tech}: {count}")

    print("\n" + "=" * 60)
    print("✨ PROCESSING COMPLETE!")
    print("=" * 60)

    return train_data, val_data, test_data


if __name__ == "__main__":
    combine_and_save()

    print("\nNext steps:")
    print("1. modal run train_llm_modal.py --command upload --data-path processed_data")
    print("2. modal run train_llm_modal.py --command train")
