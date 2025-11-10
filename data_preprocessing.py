"""
Data preprocessing pipeline for cryptopuzzle datasets.
Supports multiple data formats: JSON, CSV, and raw text files.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from datasets import Dataset
import re


class PuzzleDataPreprocessor:
    """Preprocessor for cryptopuzzle training data."""

    def __init__(self, data_path: str, output_path: str = "processed_data"):
        """
        Initialize the preprocessor.

        Args:
            data_path: Path to the raw puzzle data
            output_path: Path to save processed data
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True, parents=True)

    def load_json_data(self, file_path: Path) -> List[Dict]:
        """Load puzzle data from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]

    def load_csv_data(self, file_path: Path) -> List[Dict]:
        """Load puzzle data from CSV file."""
        df = pd.read_csv(file_path)
        return df.to_dict('records')

    def load_text_data(self, file_path: Path) -> List[Dict]:
        """Load puzzle data from text file (one puzzle per line or separated by blank lines)."""
        with open(file_path, 'r') as f:
            content = f.read()

        # Split by double newlines (assuming puzzles are separated by blank lines)
        puzzles = [p.strip() for p in content.split('\n\n') if p.strip()]

        # Convert to dict format
        return [{"puzzle": puzzle, "id": i} for i, puzzle in enumerate(puzzles)]

    def load_data(self) -> List[Dict]:
        """Load data from various file formats."""
        all_data = []

        for file_path in self.data_path.rglob('*'):
            if not file_path.is_file():
                continue

            try:
                if file_path.suffix == '.json':
                    data = self.load_json_data(file_path)
                    all_data.extend(data)
                elif file_path.suffix == '.csv':
                    data = self.load_csv_data(file_path)
                    all_data.extend(data)
                elif file_path.suffix == '.txt':
                    data = self.load_text_data(file_path)
                    all_data.extend(data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        return all_data

    def format_for_training(self, data: List[Dict]) -> List[Dict]:
        """
        Format puzzle data for LLM training.
        Creates instruction-response pairs suitable for fine-tuning.
        """
        formatted_data = []

        for item in data:
            # Try different field names that might contain puzzle content
            puzzle = item.get('puzzle') or item.get('question') or item.get('problem') or item.get('text', '')
            solution = item.get('solution') or item.get('answer') or item.get('response', '')
            hint = item.get('hint', '')
            difficulty = item.get('difficulty', 'unknown')
            category = item.get('category', 'general')

            # Create instruction-response format
            instruction = f"Solve this cryptographic puzzle"
            if category != 'general':
                instruction += f" ({category})"
            if difficulty != 'unknown':
                instruction += f" [Difficulty: {difficulty}]"
            instruction += f":\n\n{puzzle}"

            if hint:
                instruction += f"\n\nHint: {hint}"

            # Format for instruction tuning
            formatted_item = {
                "instruction": instruction,
                "input": "",
                "output": solution if solution else "No solution provided",
                "text": f"### Instruction:\n{instruction}\n\n### Response:\n{solution if solution else 'No solution provided'}"
            }

            formatted_data.append(formatted_item)

        return formatted_data

    def split_data(self, data: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1):
        """Split data into train, validation, and test sets."""
        total = len(data)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)

        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]

        return train_data, val_data, test_data

    def save_datasets(self, train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]):
        """Save processed datasets to disk."""
        # Save as JSON
        with open(self.output_path / "train.json", 'w') as f:
            json.dump(train_data, f, indent=2)

        with open(self.output_path / "validation.json", 'w') as f:
            json.dump(val_data, f, indent=2)

        with open(self.output_path / "test.json", 'w') as f:
            json.dump(test_data, f, indent=2)

        # Also save as HuggingFace datasets
        Dataset.from_list(train_data).save_to_disk(str(self.output_path / "train_dataset"))
        Dataset.from_list(val_data).save_to_disk(str(self.output_path / "val_dataset"))
        Dataset.from_list(test_data).save_to_disk(str(self.output_path / "test_dataset"))

        print(f"Saved {len(train_data)} training examples")
        print(f"Saved {len(val_data)} validation examples")
        print(f"Saved {len(test_data)} test examples")

    def process(self):
        """Run the full preprocessing pipeline."""
        print("Loading data...")
        raw_data = self.load_data()
        print(f"Loaded {len(raw_data)} raw examples")

        print("Formatting data for training...")
        formatted_data = self.format_for_training(raw_data)

        print("Splitting into train/val/test...")
        train_data, val_data, test_data = self.split_data(formatted_data)

        print("Saving datasets...")
        self.save_datasets(train_data, val_data, test_data)

        print("Preprocessing complete!")
        return train_data, val_data, test_data


def create_sample_dataset(output_path: str = "sample_data"):
    """Create a sample puzzle dataset for testing."""
    sample_puzzles = [
        {
            "puzzle": "Caesar cipher: KHOOR ZRUOG",
            "solution": "HELLO WORLD (shift of 3)",
            "difficulty": "easy",
            "category": "classical cipher",
            "hint": "Try different shift values"
        },
        {
            "puzzle": "What is the SHA-256 hash of 'password'?",
            "solution": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",
            "difficulty": "medium",
            "category": "hash functions"
        },
        {
            "puzzle": "Decode base64: SGVsbG8gV29ybGQh",
            "solution": "Hello World!",
            "difficulty": "easy",
            "category": "encoding"
        },
        {
            "puzzle": "ROT13: Uryyb Jbeyq",
            "solution": "Hello World",
            "difficulty": "easy",
            "category": "classical cipher",
            "hint": "ROT13 is a special case of Caesar cipher"
        },
        {
            "puzzle": "What is the result of XOR between 0b1010 and 0b1100?",
            "solution": "0b0110 (6 in decimal)",
            "difficulty": "medium",
            "category": "binary operations"
        }
    ]

    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    with open(output_dir / "puzzles.json", 'w') as f:
        json.dump(sample_puzzles, f, indent=2)

    print(f"Created sample dataset at {output_path}/puzzles.json")


if __name__ == "__main__":
    # Create sample data if no data exists
    import sys

    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        print("Creating sample dataset for testing...")
        create_sample_dataset()
        data_path = "sample_data"

    # Process the data
    preprocessor = PuzzleDataPreprocessor(data_path)
    preprocessor.process()
