#!/usr/bin/env python3
"""
Enhanced processor that combines all puzzle types:
- ARweave cryptopuzzles
- Cryptocurrency private key hunting puzzles
- Steganography challenges
"""

import json
from pathlib import Path
from datasets import Dataset


def process_arweave_puzzles(base_path="cryptopuzzles_data/data"):
    """Process ARweave cryptopuzzles."""
    print("📦 Processing ARweave puzzles...")

    with open(f"{base_path}/training_dataset.json", 'r') as f:
        data = json.load(f)

    formatted_data = []
    for example in data['training_examples']:
        puzzle_id = example['puzzle_id']
        difficulty = example['difficulty']
        puzzle_type = example['type']

        # Process sub-problems
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

    # Add high-value unsolved puzzles as analysis examples
    if 'bitcoin_puzzle_transaction' in data and 'high_value_unsolved' in data['bitcoin_puzzle_transaction']:
        for puzzle in data['bitcoin_puzzle_transaction']['high_value_unsolved'][:3]:  # Top 3
            instruction = f"Analyze the feasibility of solving Bitcoin Puzzle #{puzzle['puzzle_id']} ({puzzle['difficulty_bits']} bits):\n\n"
            instruction += f"- Bounty: {puzzle['amount_btc']} BTC\n"
            instruction += f"- Address: {puzzle['address']}\n"
            instruction += f"- Public key exposed: {puzzle['public_key_exposed']}\n"
            instruction += f"- Search space: {puzzle.get('search_space_decimal', puzzle.get('search_space_brute', 'N/A'))}\n\n"
            instruction += "Is this puzzle solvable with current technology?"

            response = f"**Feasibility**: {puzzle['feasibility']}\n\n"

            if puzzle['public_key_exposed']:
                response += "**Algorithm**: Pollard's Kangaroo algorithm can be used since the public key is exposed.\n\n"
                if 'search_space_kangaroo' in puzzle:
                    response += f"**Reduced Search Space**: {puzzle['search_space_kangaroo']} (vs {puzzle['search_space_brute']} for brute force)\n\n"
            else:
                response += "**Algorithm**: Only brute-force ECDSA search possible (no public key exposed).\n\n"

            if 'estimated_time_1000_gpus' in puzzle:
                response += f"**Time Estimate**: {puzzle['estimated_time_1000_gpus']} with 1,000 GPUs\n\n"
            elif 'estimated_time_single_gpu' in puzzle:
                response += f"**Time Estimate**: {puzzle['estimated_time_single_gpu']} with single GPU\n\n"

            response += f"**Conclusion**: {puzzle['feasibility'].capitalize()}. "
            if puzzle['feasibility'] == 'impossible with current technology':
                response += "The computational requirements far exceed current capabilities."
            else:
                response += "Requires significant computational resources but theoretically possible."

            formatted_data.append({
                "instruction": instruction,
                "input": "",
                "output": response,
                "text": f"### Instruction:\n{instruction}\n\n### Response:\n{response}",
                "source": "cryptocurrency",
                "puzzle_id": str(puzzle['puzzle_id']),
                "difficulty": str(puzzle['difficulty_bits']),
                "puzzle_type": "bitcoin_puzzle_unsolved",
                "technique": "feasibility_analysis",
            })

    # Add attack technique examples
    if 'attack_techniques' in data:
        for technique in data['attack_techniques']:
            instruction = f"Explain the '{technique['name']}' attack technique for cryptocurrency private key hunting:\n\n"
            instruction += f"**Complexity**: {technique['complexity']}\n"
            instruction += f"**Target**: {technique['target_type']}\n\n"
            instruction += "How does this attack work and what are its limitations?"

            response = f"**Description**: {technique['description']}\n\n"
            response += f"**Complexity**: {technique['complexity']}\n\n"
            response += f"**Tools Used**: {', '.join(technique['tools'])}\n\n"

            if 'key_insight' in technique:
                response += f"**Key Insight**: {technique['key_insight']}\n\n"

            response += "**Limitations**: "
            if technique['complexity'] == 'O(n)':
                response += "Linear search through dictionary/pattern space. Success depends on key being in the search space."
            elif technique['complexity'] == 'O(sqrt(n))':
                response += "Square-root improvement over brute force, but still requires significant computation for large key spaces."
            elif technique['complexity'] == 'O(2^n)':
                response += "Exponential complexity makes it infeasible for large key spaces (>80 bits)."

            formatted_data.append({
                "instruction": instruction,
                "input": "",
                "output": response,
                "text": f"### Instruction:\n{instruction}\n\n### Response:\n{response}",
                "source": "cryptocurrency",
                "puzzle_id": "technique",
                "difficulty": "technical",
                "puzzle_type": "attack_technique",
                "technique": str(technique['name']),
            })

    print(f"  ✅ Extracted {len(formatted_data)} cryptocurrency examples")
    return formatted_data


def process_steganography_puzzles(base_path="cryptopuzzles_data/steganography_puzzles"):
    """Process steganography challenges."""
    print("🖼️  Processing steganography puzzles...")

    index_file = f"{base_path}/datasets/challenge_index.json"
    if not Path(index_file).exists():
        print("  ⚠️  Steganography index not found, skipping...")
        return []

    with open(index_file, 'r') as f:
        data = json.load(f)

    formatted_data = []

    if 'challenges' in data:
        for challenge in data['challenges'][:10]:  # Limit to first 10
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

    print(f"  ✅ Extracted {len(formatted_data)} steganography examples")
    return formatted_data


def combine_and_save(output_dir="processed_data"):
    """Combine all puzzle types and save."""
    print("\n" + "=" * 60)
    print("🔄 PROCESSING ALL PUZZLE TYPES")
    print("=" * 60 + "\n")

    all_data = []

    # Process each type
    all_data.extend(process_arweave_puzzles())
    all_data.extend(process_cryptocurrency_puzzles())
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
    for tech, count in sorted(techniques.items(), key=lambda x: x[1], reverse=True)[:10]:
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
