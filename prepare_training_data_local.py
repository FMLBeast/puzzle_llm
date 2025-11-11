#!/usr/bin/env python3
"""
Local data preparation script - processes cryptopuzzles data for training.
This can run on your local machine without Modal.

Usage:
    python prepare_training_data_local.py --output-dir ./training_data
"""

import json
import subprocess
import argparse
from pathlib import Path
import random
import re
import shutil
from typing import List, Dict


def extract_subpuzzles_from_markdown(file_path: Path, puzzle_name: str) -> List[Dict]:
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


def parse_full_markdown(file_path: Path, source_type: str) -> Dict:
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


def main():
    parser = argparse.ArgumentParser(description="Prepare cryptopuzzle training data locally")
    parser.add_argument("--output-dir", type=str, default="./training_data", help="Output directory for processed data")
    parser.add_argument("--repo-url", type=str, default="https://github.com/FMLBeast/cryptopuzzles.git", help="Cryptopuzzles repo URL")
    parser.add_argument("--train-split", type=float, default=0.8, help="Training data split ratio")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation data split ratio")
    args = parser.parse_args()

    print("🔥 Processing cryptopuzzle data for local training!")
    print(f"   Output directory: {args.output_dir}\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Clone or update repository
    repo_dir = Path("/tmp/cryptopuzzles_local")
    if repo_dir.exists():
        print(f"📥 Updating existing cryptopuzzles repository...")
        subprocess.run(["git", "-C", str(repo_dir), "pull"], check=True)
    else:
        print(f"📥 Cloning cryptopuzzles repository...")
        subprocess.run(["git", "clone", args.repo_url, str(repo_dir)], check=True)

    all_examples = []

    # Process solved puzzles
    print("\n1️⃣ Processing solved puzzle writeups...")
    solved_dir = repo_dir / "puzzles" / "solved"
    solved_count = 0
    if solved_dir.exists():
        for md_file in solved_dir.glob("*.md"):
            print(f"  📄 {md_file.name}")
            try:
                sub_examples = extract_subpuzzles_from_markdown(md_file, md_file.stem)
                if sub_examples:
                    all_examples.extend(sub_examples)
                    solved_count += len(sub_examples)
                    print(f"     ✓ Extracted {len(sub_examples)} sub-puzzles")
                else:
                    ex = parse_full_markdown(md_file, "solved_puzzle")
                    all_examples.append(ex)
                    solved_count += 1
                    print(f"     ✓ Added full puzzle")
            except Exception as e:
                print(f"    ⚠️  Error: {e}")

    print(f"\n  ✅ TOTAL: {solved_count} examples from solved puzzles")

    # Process technique documentation
    print("\n2️⃣ Processing technique documentation...")
    techniques_dir = repo_dir / "techniques"
    technique_count = 0
    if techniques_dir.exists():
        for md_file in techniques_dir.rglob("*.md"):
            print(f"  📄 {md_file.name}")
            try:
                ex = parse_full_markdown(md_file, "technique_guide")
                all_examples.append(ex)
                technique_count += 1
            except Exception as e:
                print(f"    ⚠️  Error: {e}")

    print(f"\n  ✅ TOTAL: {technique_count} technique guides")

    # Process other documentation
    print("\n3️⃣ Processing additional documentation...")
    other_count = 0
    for md_file in repo_dir.rglob("*.md"):
        if 'solved' in str(md_file) or 'techniques' in str(md_file):
            continue
        if any(part.startswith('.') for part in md_file.parts):
            continue

        print(f"  📄 {md_file.relative_to(repo_dir)}")
        try:
            ex = parse_full_markdown(md_file, "documentation")
            all_examples.append(ex)
            other_count += 1
        except Exception as e:
            print(f"    ⚠️  Error: {e}")

    print(f"\n  ✅ TOTAL: {other_count} additional files")

    print(f"\n📊 TOTAL EXAMPLES: {len(all_examples)}")

    # Split dataset
    print(f"\n4️⃣ Splitting dataset...")
    random.seed(42)
    random.shuffle(all_examples)

    n = len(all_examples)
    train_end = int(args.train_split * n)
    val_end = train_end + int(args.val_split * n)

    train_data = all_examples[:train_end]
    val_data = all_examples[train_end:val_end]
    test_data = all_examples[val_end:]

    print(f"  📚 Train: {len(train_data)} examples ({len(train_data)/n*100:.1f}%)")
    print(f"  📚 Val:   {len(val_data)} examples ({len(val_data)/n*100:.1f}%)")
    print(f"  📚 Test:  {len(test_data)} examples ({len(test_data)/n*100:.1f}%)")

    # Save datasets
    print(f"\n5️⃣ Saving datasets to {output_dir}...")

    with open(output_dir / "train.json", 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"  ✓ Saved train.json")

    with open(output_dir / "validation.json", 'w') as f:
        json.dump(val_data, f, indent=2)
    print(f"  ✓ Saved validation.json")

    with open(output_dir / "test.json", 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"  ✓ Saved test.json")

    # Also save in JSONL format for easier loading
    with open(output_dir / "train.jsonl", 'w') as f:
        for ex in train_data:
            f.write(json.dumps(ex) + '\n')
    print(f"  ✓ Saved train.jsonl")

    with open(output_dir / "validation.jsonl", 'w') as f:
        for ex in val_data:
            f.write(json.dumps(ex) + '\n')
    print(f"  ✓ Saved validation.jsonl")

    with open(output_dir / "test.jsonl", 'w') as f:
        for ex in test_data:
            f.write(json.dumps(ex) + '\n')
    print(f"  ✓ Saved test.jsonl")

    # Save summary stats
    stats = {
        "total_examples": len(all_examples),
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "test_examples": len(test_data),
        "sources": {
            "solved_puzzles": solved_count,
            "technique_guides": technique_count,
            "other_docs": other_count,
        }
    }

    with open(output_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓ Saved stats.json")

    print(f"\n✅ Data preparation complete!")
    print(f"\n📁 Files created in {output_dir}:")
    print(f"   - train.json / train.jsonl ({len(train_data)} examples)")
    print(f"   - validation.json / validation.jsonl ({len(val_data)} examples)")
    print(f"   - test.json / test.jsonl ({len(test_data)} examples)")
    print(f"   - stats.json (summary statistics)")

    print(f"\n📝 Sample training example:")
    print("=" * 80)
    print(train_data[0]['text'][:400] + "...")
    print("=" * 80)

    return stats


if __name__ == "__main__":
    main()
