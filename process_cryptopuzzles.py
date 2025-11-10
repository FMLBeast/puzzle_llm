#!/usr/bin/env python3
"""
Specialized preprocessor for the FMLBeast/cryptopuzzles dataset.
Converts the training_dataset.json into instruction-tuning format.
"""

import json
from pathlib import Path
from datasets import Dataset


def process_cryptopuzzles_data(input_file="cryptopuzzles_data/data/training_dataset.json"):
    """Process the cryptopuzzles training dataset."""

    print("=" * 60)
    print("🔄 PROCESSING CRYPTOPUZZLES DATASET")
    print("=" * 60)

    with open(input_file, 'r') as f:
        data = json.load(f)

    training_examples = data['training_examples']
    print(f"\n📊 Found {len(training_examples)} solved puzzles")

    formatted_data = []

    for example in training_examples:
        puzzle_id = example['puzzle_id']
        difficulty = example['difficulty']
        puzzle_type = example['type']
        problem = example['problem']

        # Process each sub-problem
        if 'sub_problems' in example:
            for i, sub_problem in enumerate(example['sub_problems']):
                # Create instruction
                instruction = f"Solve this cryptographic puzzle (Type: {puzzle_type}, Difficulty: {difficulty}):\n\n"
                instruction += f"**Problem {i+1}/{len(example['sub_problems'])}**: "
                instruction += sub_problem.get('question', sub_problem.get('type', ''))

                # Add hint if available
                if 'hint' in sub_problem:
                    instruction += f"\n\nHint: {sub_problem['hint']}"

                # Create response with solution
                response = f"**Solution**: {sub_problem.get('solution', 'No solution available')}"

                # Add explanation if available
                if 'explanation' in sub_problem:
                    response += f"\n\n**Explanation**: {sub_problem['explanation']}"

                # Add technique
                if 'technique' in sub_problem:
                    response += f"\n\n**Technique Used**: {sub_problem['technique']}"

                # Add steps if available
                if 'steps' in sub_problem:
                    response += "\n\n**Steps**:\n"
                    for step in sub_problem['steps']:
                        response += f"- {step}\n"

                # Create formatted item
                formatted_item = {
                    "instruction": instruction,
                    "input": "",
                    "output": response,
                    "text": f"### Instruction:\n{instruction}\n\n### Response:\n{response}",
                    "metadata": {
                        "puzzle_id": puzzle_id,
                        "difficulty": difficulty,
                        "type": puzzle_type,
                        "sub_problem_type": sub_problem.get('type', ''),
                        "technique": sub_problem.get('technique', ''),
                    }
                }

                formatted_data.append(formatted_item)

        # Also create a full puzzle example
        full_instruction = f"Solve the complete cryptopuzzle (Puzzle #{puzzle_id}, Difficulty: {difficulty}):\n\n{problem}"

        if 'key_skills' in example:
            full_instruction += f"\n\nRequired Skills: {', '.join(example['key_skills'])}"

        full_response = f"**Final Solution**: {example['final_solution']}"

        if 'format_notes' in example:
            full_response += f"\n\n**Format Notes**: {example['format_notes']}"

        formatted_data.append({
            "instruction": full_instruction,
            "input": "",
            "output": full_response,
            "text": f"### Instruction:\n{full_instruction}\n\n### Response:\n{full_response}",
            "metadata": {
                "puzzle_id": puzzle_id,
                "difficulty": difficulty,
                "type": puzzle_type,
                "is_full_puzzle": True
            }
        })

    print(f"✅ Created {len(formatted_data)} training examples")
    return formatted_data


def split_and_save(data, output_dir="processed_data", train_ratio=0.8, val_ratio=0.1):
    """Split data and save to disk."""

    print(f"\n📁 Saving to {output_dir}/")
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Calculate split sizes
    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    # Split data
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

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

    return train_data, val_data, test_data


def show_sample(data):
    """Show a sample of the processed data."""
    print("\n" + "=" * 60)
    print("📋 SAMPLE TRAINING EXAMPLE")
    print("=" * 60)

    if data:
        sample = data[0]
        print(f"\n{sample['text'][:500]}...")
        print(f"\nMetadata: {sample['metadata']}")


if __name__ == "__main__":
    # Process data
    formatted_data = process_cryptopuzzles_data()

    # Show sample
    show_sample(formatted_data)

    # Split and save
    train_data, val_data, test_data = split_and_save(formatted_data)

    print("\n" + "=" * 60)
    print("✨ PROCESSING COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. modal run train_llm_modal.py --command upload --data-path processed_data")
    print("2. modal run train_llm_modal.py --command train")
