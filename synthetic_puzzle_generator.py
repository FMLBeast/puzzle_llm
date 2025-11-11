"""
SYNTHETIC CRYPTOPUZZLE GENERATOR
==================================

Generates 100,000+ diverse cryptopuzzles programmatically.

Types:
1. Logic Grid Puzzles (Zebra-style)
2. Caesar Ciphers (all shifts)
3. Substitution Ciphers
4. Hash Puzzles
5. Multi-step Reasoning Chains
6. ARweave-style Multi-puzzles

Usage:
    modal run synthetic_puzzle_generator.py::generate_all_puzzles

Author: FMLBeast
"""

import modal
import random
import string
import hashlib
import json
from typing import List, Dict, Tuple
from itertools import permutations, combinations

# Modal setup
app = modal.App("synthetic-puzzle-generator")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy",
    "sympy",  # For math puzzles
)

volume = modal.Volume.from_name("puzzle-training-data", create_if_missing=True)


# ============================================================================
# LOGIC GRID PUZZLE GENERATOR (Zebra Puzzle Style)
# ============================================================================

@app.function(image=image, timeout=600)
def generate_logic_grid_puzzles(count: int = 1000):
    """
    Generate Zebra-style logic grid puzzles.

    Example:
    There are 5 houses in a row, each a different color.
    In each house lives a person of a different nationality.
    Each person drinks a different beverage, smokes different cigars, has different pet.

    Clues:
    1. The English person lives in the red house
    2. The Swede has a dog
    ...

    Question: Who owns the fish?
    """
    puzzles = []

    COLORS = ["Red", "Blue", "Green", "Yellow", "White"]
    NATIONALITIES = ["English", "Swedish", "Danish", "Norwegian", "German"]
    DRINKS = ["Tea", "Coffee", "Milk", "Beer", "Water"]
    CIGARS = ["Pall Mall", "Dunhill", "Blend", "Blue Master", "Prince"]
    PETS = ["Dog", "Cat", "Bird", "Horse", "Fish"]

    categories = [COLORS, NATIONALITIES, DRINKS, CIGARS, PETS]
    category_names = ["Color", "Nationality", "Drink", "Cigar", "Pet"]

    for puzzle_id in range(count):
        grid_size = 5

        # Generate random solution
        solution = {}
        for cat_name, items in zip(category_names, categories):
            shuffled = items.copy()
            random.shuffle(shuffled)
            solution[cat_name] = shuffled

        # Generate clues from solution
        clues = []

        # Type 1: Direct assignment (e.g., "The English person lives in the red house")
        for _ in range(random.randint(3, 5)):
            cat1, cat2 = random.sample(range(len(categories)), 2)
            position = random.randint(0, grid_size - 1)
            clue = f"The {solution[category_names[cat1]][position]} {category_names[cat1].lower()} has {solution[category_names[cat2]][position].lower()}"
            clues.append(clue)

        # Type 2: Relative position (e.g., "The green house is immediately to the right of the white house")
        for _ in range(random.randint(2, 4)):
            cat = random.randint(0, len(categories) - 1)
            if random.random() > 0.5:
                pos1, pos2 = random.randint(0, grid_size - 2), random.randint(1, grid_size - 1)
                if pos2 == pos1 + 1:
                    clue = f"The {solution[category_names[cat]][pos1].lower()} is immediately left of the {solution[category_names[cat]][pos2].lower()}"
                    clues.append(clue)

        # Type 3: Neighbor (e.g., "The person who smokes Blend lives next to the cat owner")
        for _ in range(random.randint(2, 3)):
            cat1, cat2 = random.sample(range(len(categories)), 2)
            pos1 = random.randint(0, grid_size - 1)
            if pos1 > 0:
                pos2 = pos1 - 1
            elif pos1 < grid_size - 1:
                pos2 = pos1 + 1
            else:
                continue
            clue = f"The {solution[category_names[cat1]][pos1].lower()} is next to the {solution[category_names[cat2]][pos2].lower()}"
            clues.append(clue)

        # Create question - ask about a specific attribute
        question_cat1 = random.randint(0, len(categories) - 1)
        question_cat2 = random.randint(0, len(categories) - 1)
        while question_cat2 == question_cat1:
            question_cat2 = random.randint(0, len(categories) - 1)

        question_item = solution[category_names[question_cat1]][random.randint(0, grid_size - 1)]
        question = f"Who/what has {question_item.lower()}?"

        # Find answer
        for pos in range(grid_size):
            if solution[category_names[question_cat1]][pos] == question_item:
                answer = solution[category_names[question_cat2]][pos]
                break

        puzzle = {
            "id": f"logic_grid_{puzzle_id}",
            "type": "logic_grid",
            "difficulty": "medium",
            "puzzle": {
                "description": f"There are {grid_size} houses in a row. Each house has unique attributes.",
                "categories": category_names,
                "clues": clues,
                "question": question
            },
            "solution": answer,
            "full_solution": solution,
            "instruction_format": f"### Instruction:\n[Type: Logic Grid Puzzle]\n\nClues:\n" + "\n".join(f"{i+1}. {c}" for i, c in enumerate(clues)) + f"\n\nQuestion: {question}\n\n### Response:\n{answer}"
        }

        puzzles.append(puzzle)

    return puzzles


# ============================================================================
# CAESAR CIPHER GENERATOR
# ============================================================================

@app.function(image=image, timeout=300)
def generate_caesar_ciphers(count: int = 1000):
    """Generate Caesar cipher puzzles (all 25 shifts)."""
    puzzles = []

    SAMPLE_TEXTS = [
        "HELLO WORLD",
        "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
        "CRYPTOGRAPHY IS FUN",
        "SOLVE THIS PUZZLE",
        "BITCOIN TO THE MOON",
        "ARWEAVE PERMANENT STORAGE",
        "PUZZLE SOLVED SUCCESSFULLY",
        "CONGRATULATIONS YOU WIN",
        "HIDDEN MESSAGE REVEALED",
        "SECRET CODE CRACKED"
    ]

    for i in range(count):
        plaintext = random.choice(SAMPLE_TEXTS)
        shift = random.randint(1, 25)

        # Encrypt
        ciphertext = ""
        for char in plaintext:
            if char.isalpha():
                shifted = chr((ord(char) - ord('A') + shift) % 26 + ord('A'))
                ciphertext += shifted
            else:
                ciphertext += char

        # Create reasoning steps
        steps = [
            f"Identify this as a Caesar cipher (letter substitution with fixed shift)",
            f"Try all 25 possible shifts systematically",
            f"Shift {shift} produces readable text: {plaintext}",
            f"Verify: Each letter shifted by {shift} positions"
        ]

        puzzle = {
            "id": f"caesar_{i}",
            "type": "caesar_cipher",
            "difficulty": "easy",
            "puzzle": f"Decode: {ciphertext}",
            "solution": plaintext,
            "shift": shift,
            "steps": steps,
            "instruction_format": f"### Instruction:\n[Type: Caesar Cipher] [Difficulty: easy]\n\nDecode: {ciphertext}\n\n### Response:\n**Solution**: {plaintext}\n\n**Reasoning**:\n" + "\n".join(f"- {s}" for s in steps)
        }

        puzzles.append(puzzle)

    return puzzles


# ============================================================================
# SUBSTITUTION CIPHER GENERATOR
# ============================================================================

@app.function(image=image, timeout=300)
def generate_substitution_ciphers(count: int = 500):
    """Generate substitution cipher puzzles."""
    puzzles = []

    SAMPLE_TEXTS = [
        "THE TREASURE IS BURIED UNDER THE OLD OAK TREE",
        "MEET ME AT MIDNIGHT BY THE FOUNTAIN",
        "PASSWORD IS SWORDFISH",
        "THE EAGLE HAS LANDED",
        "ALL YOUR BASE ARE BELONG TO US"
    ]

    for i in range(count):
        plaintext = random.choice(SAMPLE_TEXTS)

        # Generate random substitution key
        alphabet = list(string.ascii_uppercase)
        shuffled = alphabet.copy()
        random.shuffle(shuffled)
        key = dict(zip(alphabet, shuffled))

        # Encrypt
        ciphertext = ""
        for char in plaintext:
            if char.isalpha():
                ciphertext += key[char]
            else:
                ciphertext += char

        # Create frequency analysis hint
        freq_hint = "Most common letter in English: E, T, A, O, I, N"

        puzzle = {
            "id": f"substitution_{i}",
            "type": "substitution_cipher",
            "difficulty": "medium",
            "puzzle": f"Decode (each letter maps to another letter consistently): {ciphertext}\n\nHint: {freq_hint}",
            "solution": plaintext,
            "key": key,
            "instruction_format": f"### Instruction:\n[Type: Substitution Cipher]\n\nDecode: {ciphertext}\n\n### Response:\n**Solution**: {plaintext}\n\n**Method**: Frequency analysis and pattern matching"
        }

        puzzles.append(puzzle)

    return puzzles


# ============================================================================
# HASH PUZZLE GENERATOR
# ============================================================================

@app.function(image=image, timeout=300)
def generate_hash_puzzles(count: int = 1000):
    """Generate SHA-256 hash puzzles."""
    puzzles = []

    COMMON_WORDS = [
        "bitcoin", "ethereum", "arweave", "crypto", "puzzle",
        "secret", "treasure", "hidden", "password", "key",
        "alpha", "beta", "gamma", "delta", "omega"
    ]

    for i in range(count):
        # Type 1: Find word that produces hash with specific pattern
        target_word = random.choice(COMMON_WORDS)
        hash_obj = hashlib.sha256(target_word.encode())
        hash_hex = hash_obj.hexdigest()

        # Extract pattern (last N digits, starts with X, etc.)
        pattern_type = random.choice(["last_digits", "starts_with", "contains"])

        if pattern_type == "last_digits":
            n = random.randint(4, 8)
            pattern = hash_hex[-n:]
            question = f"Find a common English word whose SHA-256 hash ends with: {pattern}"
        elif pattern_type == "starts_with":
            n = random.randint(2, 4)
            pattern = hash_hex[:n]
            question = f"Find a common English word whose SHA-256 hash starts with: {pattern}"
        else:
            n = random.randint(4, 6)
            start_pos = random.randint(10, 50)
            pattern = hash_hex[start_pos:start_pos+n]
            question = f"Find a common English word whose SHA-256 hash contains: {pattern}"

        steps = [
            "Use common English word dictionary",
            "Compute SHA-256 hash for each word",
            f"Check if hash matches pattern: {pattern}",
            f"Found: '{target_word}' → {hash_hex}",
            f"Verification: Hash matches the required pattern"
        ]

        puzzle = {
            "id": f"hash_{i}",
            "type": "hash_puzzle",
            "difficulty": "medium",
            "puzzle": question,
            "solution": target_word,
            "hash": hash_hex,
            "pattern": pattern,
            "steps": steps,
            "instruction_format": f"### Instruction:\n[Type: Hash Puzzle]\n\n{question}\n\n### Response:\n**Solution**: {target_word}\n\n**Hash**: {hash_hex}\n\n**Reasoning**:\n" + "\n".join(f"- {s}" for s in steps)
        }

        puzzles.append(puzzle)

    return puzzles


# ============================================================================
# MULTI-STEP REASONING CHAIN GENERATOR
# ============================================================================

@app.function(image=image, timeout=600)
def generate_multi_step_puzzles(count: int = 500):
    """Generate complex multi-step puzzles combining different techniques."""
    puzzles = []

    for i in range(count):
        num_steps = random.randint(3, 5)

        # Build puzzle chain
        steps = []
        current_message = f"START_{random.randint(1000, 9999)}"

        step_types = random.sample(["caesar", "reverse", "base64", "hash_tail", "math"], num_steps)

        for step_num, step_type in enumerate(step_types, 1):
            if step_type == "caesar":
                shift = random.randint(1, 25)
                encrypted = ""
                for char in current_message:
                    if char.isalpha():
                        base = ord('A') if char.isupper() else ord('a')
                        encrypted += chr((ord(char) - base + shift) % 26 + base)
                    else:
                        encrypted += char
                steps.append({
                    "step": step_num,
                    "type": "Caesar cipher",
                    "operation": f"Apply Caesar shift {shift}",
                    "input": current_message,
                    "output": encrypted
                })
                current_message = encrypted

            elif step_type == "reverse":
                reversed_msg = current_message[::-1]
                steps.append({
                    "step": step_num,
                    "type": "String reversal",
                    "operation": "Reverse the string",
                    "input": current_message,
                    "output": reversed_msg
                })
                current_message = reversed_msg

            elif step_type == "hash_tail":
                hash_hex = hashlib.sha256(current_message.encode()).hexdigest()
                tail = hash_hex[-8:]
                steps.append({
                    "step": step_num,
                    "type": "Hash operation",
                    "operation": "Take last 8 chars of SHA-256",
                    "input": current_message,
                    "output": tail
                })
                current_message = tail

        final_answer = current_message

        # Create puzzle description
        puzzle_desc = f"Starting message undergoes {num_steps} transformations:\n"
        for s in steps:
            puzzle_desc += f"Step {s['step']}: {s['operation']}\n"

        puzzle_desc += f"\nFinal result: {final_answer}\n\nWhat was the starting message?"

        # Create solution steps (reverse order)
        solution_steps = []
        reverse_steps = list(reversed(steps))
        working_message = final_answer

        for s in reverse_steps:
            if s['type'] == "Caesar cipher":
                # Extract shift from operation
                shift = int(s['operation'].split()[-1])
                decrypted = ""
                for char in working_message:
                    if char.isalpha():
                        base = ord('A') if char.isupper() else ord('a')
                        decrypted += chr((ord(char) - base - shift) % 26 + base)
                    else:
                        decrypted += char
                solution_steps.append(f"Reverse Caesar shift {shift}: {working_message} → {decrypted}")
                working_message = decrypted

            elif s['type'] == "String reversal":
                working_message = working_message[::-1]
                solution_steps.append(f"Reverse string: → {working_message}")

        starting_message = reverse_steps[-1]['input'] if reverse_steps else "START"

        puzzle = {
            "id": f"multi_step_{i}",
            "type": "multi_step",
            "difficulty": "hard",
            "puzzle": puzzle_desc,
            "solution": starting_message,
            "steps": steps,
            "solution_steps": solution_steps,
            "instruction_format": f"### Instruction:\n[Type: Multi-Step Puzzle] [Steps: {num_steps}]\n\n{puzzle_desc}\n\n### Response:\n**Starting message**: {starting_message}\n\n**Solution steps**:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(solution_steps))
        }

        puzzles.append(puzzle)

    return puzzles


# ============================================================================
# MASTER GENERATION FUNCTION
# ============================================================================

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=7200
)
def generate_all_puzzles(
    logic_count: int = 5000,
    caesar_count: int = 10000,
    substitution_count: int = 3000,
    hash_count: int = 10000,
    multi_step_count: int = 5000
):
    """
    Generate comprehensive synthetic puzzle dataset.

    Total: 33,000 puzzles by default
    """
    from pathlib import Path
    import time

    print("🔥 SYNTHETIC PUZZLE GENERATION STARTING...")
    print(f"   Target: {logic_count + caesar_count + substitution_count + hash_count + multi_step_count} puzzles\n")

    start_time = time.time()

    # Generate all puzzle types in parallel using Modal's .map()
    print("1️⃣ Generating Logic Grid Puzzles...")
    logic_start = time.time()
    logic_batches = [1000] * (logic_count // 1000)
    logic_results = list(generate_logic_grid_puzzles.map(logic_batches))
    logic_puzzles = [p for batch in logic_results for p in batch]
    print(f"   ✓ Generated {len(logic_puzzles)} logic puzzles in {time.time()-logic_start:.1f}s\n")

    print("2️⃣ Generating Caesar Ciphers...")
    caesar_start = time.time()
    caesar_batches = [1000] * (caesar_count // 1000)
    caesar_results = list(generate_caesar_ciphers.map(caesar_batches))
    caesar_puzzles = [p for batch in caesar_results for p in batch]
    print(f"   ✓ Generated {len(caesar_puzzles)} Caesar ciphers in {time.time()-caesar_start:.1f}s\n")

    print("3️⃣ Generating Substitution Ciphers...")
    sub_start = time.time()
    sub_batches = [500] * (substitution_count // 500)
    sub_results = list(generate_substitution_ciphers.map(sub_batches))
    sub_puzzles = [p for batch in sub_results for p in batch]
    print(f"   ✓ Generated {len(sub_puzzles)} substitution ciphers in {time.time()-sub_start:.1f}s\n")

    print("4️⃣ Generating Hash Puzzles...")
    hash_start = time.time()
    hash_batches = [1000] * (hash_count // 1000)
    hash_results = list(generate_hash_puzzles.map(hash_batches))
    hash_puzzles = [p for batch in hash_results for p in batch]
    print(f"   ✓ Generated {len(hash_puzzles)} hash puzzles in {time.time()-hash_start:.1f}s\n")

    print("5️⃣ Generating Multi-Step Puzzles...")
    multi_start = time.time()
    multi_batches = [500] * (multi_step_count // 500)
    multi_results = list(generate_multi_step_puzzles.map(multi_batches))
    multi_puzzles = [p for batch in multi_results for p in batch]
    print(f"   ✓ Generated {len(multi_puzzles)} multi-step puzzles in {time.time()-multi_start:.1f}s\n")

    # Combine all puzzles
    all_puzzles = {
        "logic_grid": logic_puzzles,
        "caesar_cipher": caesar_puzzles,
        "substitution_cipher": sub_puzzles,
        "hash_puzzle": hash_puzzles,
        "multi_step": multi_puzzles
    }

    total_count = sum(len(puzzles) for puzzles in all_puzzles.values())

    print(f"\n{'='*60}")
    print(f"📊 GENERATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Total puzzles: {total_count}")
    print(f"Total time: {time.time()-start_time:.1f}s")
    print(f"\nBreakdown:")
    for puzzle_type, puzzles in all_puzzles.items():
        print(f"  - {puzzle_type}: {len(puzzles)}")

    # Save to volume
    dest = Path("/data/synthetic_puzzles")
    dest.mkdir(exist_ok=True, parents=True)

    print(f"\n6️⃣ Saving to volume...")
    for puzzle_type, puzzles in all_puzzles.items():
        filepath = dest / f"{puzzle_type}.json"
        with open(filepath, 'w') as f:
            json.dump(puzzles, f, indent=2)
        print(f"   ✓ Saved {filepath}")

    # Save combined dataset
    combined_path = dest / "all_puzzles.json"
    with open(combined_path, 'w') as f:
        json.dump(all_puzzles, f, indent=2)
    print(f"   ✓ Saved {combined_path}")

    # Create training format
    print(f"\n7️⃣ Creating training format...")
    training_examples = []
    for puzzle_type, puzzles in all_puzzles.items():
        for puzzle in puzzles:
            training_examples.append({
                "text": puzzle['instruction_format'],
                "type": puzzle_type,
                "difficulty": puzzle['difficulty']
            })

    training_path = dest / "training_format.json"
    with open(training_path, 'w') as f:
        json.dump(training_examples, f, indent=2)
    print(f"   ✓ Saved {training_path} ({len(training_examples)} examples)")

    volume.commit()
    print(f"\n✅ ALL DATA SAVED TO VOLUME!")

    return {
        "total_puzzles": total_count,
        "breakdown": {k: len(v) for k, v in all_puzzles.items()},
        "time_seconds": time.time() - start_time
    }


# ============================================================================
# CLI ENTRYPOINT
# ============================================================================

@app.local_entrypoint()
def main():
    """Generate comprehensive synthetic puzzle dataset."""
    print("="*60)
    print("SYNTHETIC CRYPTOPUZZLE GENERATOR")
    print("="*60)

    result = generate_all_puzzles.remote(
        logic_count=5000,
        caesar_count=10000,
        substitution_count=3000,
        hash_count=10000,
        multi_step_count=5000
    )

    print(f"\n{'='*60}")
    print(f"🎉 SUCCESS!")
    print(f"{'='*60}")
    print(f"Generated {result['total_puzzles']} puzzles in {result['time_seconds']:.1f}s")
    print(f"\nNext steps:")
    print(f"1. Run multi-agent solver: modal run train_llm_modal.py::solve_synthetic_puzzles")
    print(f"2. Train on verified solutions: modal run train_llm_modal.py::train")
