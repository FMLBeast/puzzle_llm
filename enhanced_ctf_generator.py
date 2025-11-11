"""
Enhanced CTF Puzzle Generator with FULL SOLUTION REASONING
Generates expert CTF puzzles with complete step-by-step solution chains.
"""

import modal
import random
import json

app = modal.App("enhanced-ctf-generator")

# Enhanced image with all security tools
image = modal.Image.debian_slim(python_version="3.11").apt_install(
    "binutils", "file", "exiftool"
).pip_install(
    "Pillow", "numpy", "pycryptodome", "web3", "bitcoin", "pyOpenSSL", "cryptography"
)

# Volume for storing generated puzzles
volume = modal.Volume.from_name("puzzle-training-data", create_if_missing=True)


def generate_caesar_cipher_with_reasoning(count=20):
    """Generate Caesar cipher puzzles with FULL REASONING CHAINS"""
    puzzles = []

    for _ in range(count):
        messages = [
            "HELLO WORLD", "ATTACK AT DAWN", "SECRET MESSAGE",
            "FLAG IS HERE", "SOLVE THIS PUZZLE", "CRYPTO CHALLENGE",
            "HIDDEN TEXT", "DECODE ME NOW"
        ]
        shift = random.randint(1, 25)
        plaintext = random.choice(messages)

        # Encode with Caesar cipher
        ciphertext = ""
        for char in plaintext:
            if char.isalpha():
                base = ord('A')
                shifted = (ord(char) - base + shift) % 26
                ciphertext += chr(base + shifted)
            else:
                ciphertext += char

        # Create DETAILED solution with reasoning
        puzzle_text = f"""Decode this Caesar cipher:

**Ciphertext:** {ciphertext}

**Hint:** Try different shift values."""

        solution_text = f"""**Solution Method:**

**Step 1: Identify the cipher type**
- This is a Caesar cipher (substitution cipher with fixed shift)
- Each letter is shifted by a constant amount in the alphabet

**Step 2: Determine the shift**
- Try shifts from 1 to 25
- For shift {shift}:
  - '{ciphertext[0]}' → '{plaintext[0]}' (shift back by {shift})
  - '{ciphertext[1] if len(ciphertext) > 1 else ""}' → '{plaintext[1] if len(plaintext) > 1 else ""}' (shift back by {shift})

**Step 3: Decode the message**
- Shift each letter back by {shift} positions
- {ciphertext[0]} (ASCII {ord(ciphertext[0])}) - {shift} = {plaintext[0]} (ASCII {ord(plaintext[0])})

**Step 4: Verify the result**
- Decoded message forms valid English words
- Message makes sense in context

**Final Answer:** {plaintext}

**Verification:**
- Shift value: {shift}
- Decoded: {plaintext}"""

        puzzles.append({
            "puzzle": puzzle_text,
            "solution": solution_text,
            "type": "caesar_cipher",
            "difficulty": "easy",
            "category": "cryptography"
        })

    return puzzles


def generate_rsa_puzzles_with_reasoning(count=15):
    """Generate RSA factorization puzzles with DETAILED REASONING"""
    puzzles = []

    for _ in range(count):
        # Small primes for educational RSA
        small_primes = [1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049,
                       1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097]

        p = random.choice(small_primes)
        q = random.choice([x for x in small_primes if x != p])
        n = p * q
        e = 65537  # Standard public exponent

        phi = (p - 1) * (q - 1)
        d = pow(e, -1, phi)  # Modular inverse

        message = random.randint(100, 1000)
        ciphertext = pow(message, e, n)

        puzzle_text = f"""You intercepted this RSA encrypted message:

**Public Key:**
- n = {n}
- e = {e}

**Ciphertext:** c = {ciphertext}

**Task:** Factor n to recover the private key and decrypt the message."""

        solution_text = f"""**Solution Method:**

**Step 1: Understand RSA security**
- RSA security relies on difficulty of factoring n = p × q
- If n is small, we can factor it

**Step 2: Factor n = {n}**
- Try small primes as potential factors
- Test if {n} is divisible by primes
- {n} = {p} × {q}
- Verification: {p} × {q} = {n} ✓

**Step 3: Calculate φ(n)**
- φ(n) = (p - 1) × (q - 1)
- φ({n}) = ({p} - 1) × ({q} - 1)
- φ({n}) = {p-1} × {q-1} = {phi}

**Step 4: Calculate private exponent d**
- d ≡ e^(-1) mod φ(n)
- d ≡ {e}^(-1) mod {phi}
- d = {d}

**Step 5: Decrypt the message**
- m = c^d mod n
- m = {ciphertext}^{d} mod {n}
- m = {message}

**Final Answer:**
- p = {p}, q = {q}
- Private key d = {d}
- Decrypted message m = {message}

**Verification:**
- Encrypt: {message}^{e} mod {n} = {ciphertext} ✓
- Decrypt: {ciphertext}^{d} mod {n} = {message} ✓"""

        puzzles.append({
            "puzzle": puzzle_text,
            "solution": solution_text,
            "type": "rsa_factorization",
            "difficulty": "medium",
            "category": "cryptography"
        })

    return puzzles


def generate_file_identification_with_reasoning(count=20):
    """Generate file identification puzzles with FULL REASONING"""
    puzzles = []

    files = [
        ("89 50 4E 47 0D 0A 1A 0A", "PNG", "PNG image format"),
        ("FF D8 FF E0", "JPEG", "JPEG/JFIF image"),
        ("47 49 46 38 39 61", "GIF", "GIF89a image"),
        ("50 4B 03 04", "ZIP", "ZIP archive"),
        ("7F 45 4C 46", "ELF", "Linux executable"),
        ("4D 5A", "EXE", "Windows PE executable"),
        ("25 50 44 46", "PDF", "PDF document"),
        ("52 61 72 21", "RAR", "RAR archive"),
        ("1F 8B 08", "GZIP", "GZIP compressed file"),
        ("42 4D", "BMP", "Windows bitmap"),
    ]

    for _ in range(count):
        magic_bytes, file_type, description = random.choice(files)

        puzzle_text = f"""Identify this file type from its magic bytes:

**Hex dump (first 16 bytes):**
```
{magic_bytes} 00 00 00 0D ...
```

**Task:** Determine the file format."""

        # Convert hex to ASCII for demonstration
        ascii_repr = ""
        for byte in magic_bytes.split():
            try:
                ascii_repr += chr(int(byte, 16)) if 32 <= int(byte, 16) < 127 else "."
            except:
                ascii_repr += "."

        solution_text = f"""**Solution Method:**

**Step 1: Understand magic bytes**
- Magic bytes (file signature) identify file formats
- Located at the beginning of files
- Unique to each file type

**Step 2: Extract the signature**
- First bytes: {magic_bytes}
- ASCII representation: "{ascii_repr}"

**Step 3: Look up the signature**
- {magic_bytes} is the magic number for {file_type}
- This identifies a {description}

**Step 4: Verify with tools**
- Command: `file <filename>`
- Expected output: "{description}"
- Alternative: Check https://en.wikipedia.org/wiki/List_of_file_signatures

**Final Answer:** {file_type} ({description})

**Common signatures reference:**
- PNG: 89 50 4E 47 (‰PNG)
- JPEG: FF D8 FF E0
- ZIP: 50 4B 03 04 (PK)
- ELF: 7F 45 4C 46 (.ELF)
- PDF: 25 50 44 46 (%PDF)"""

        puzzles.append({
            "puzzle": puzzle_text,
            "solution": solution_text,
            "type": "file_identification",
            "difficulty": "easy",
            "category": "forensics"
        })

    return puzzles


def generate_base64_puzzles_with_reasoning(count=20):
    """Generate Base64 encoding puzzles with REASONING"""
    import base64

    puzzles = []
    messages = [
        "FLAG{base64_encoded}", "SECRET_MESSAGE", "HIDDEN_TEXT",
        "CRYPTO_CHALLENGE", "DECODE_THIS", "CTF{base64_fun}"
    ]

    for _ in range(count):
        plaintext = random.choice(messages)
        encoded = base64.b64encode(plaintext.encode()).decode()

        puzzle_text = f"""Decode this encoded string:

**Encoded:** {encoded}

**Hint:** This is a common encoding scheme used in data transmission."""

        solution_text = f"""**Solution Method:**

**Step 1: Identify the encoding**
- Character set: A-Z, a-z, 0-9, +, /
- Padding with '=' characters at the end
- This is Base64 encoding

**Step 2: Understand Base64**
- Encodes binary data as ASCII text
- 6 bits per character (2^6 = 64 possible values)
- Common in email, data URLs, credentials

**Step 3: Decode the string**
- Use base64 decoding tool
- Command: `echo "{encoded}" | base64 -d`
- OR: `base64 -d <<< "{encoded}"`
- OR: Python: `base64.b64decode("{encoded}")`

**Step 4: Verify the result**
- Decoded bytes form valid text
- Check for flag format or readable message

**Final Answer:** {plaintext}

**Verification:**
- Original: {plaintext}
- Encoded: {encoded}
- Decoded: {plaintext} ✓"""

        puzzles.append({
            "puzzle": puzzle_text,
            "solution": solution_text,
            "type": "base64_decode",
            "difficulty": "easy",
            "category": "encoding"
        })

    return puzzles


@app.function(
    image=image,
    volumes={"/ctf_data": volume},
    timeout=600,
)
def generate_enhanced_ctf_dataset(total_puzzles=1000):
    """Generate comprehensive CTF dataset with FULL REASONING CHAINS"""
    import json

    print(f"🔥 Generating {total_puzzles} CTF puzzles with DETAILED SOLUTION REASONING!\n")

    all_puzzles = []

    # Generate different puzzle types with proper distribution
    print("1️⃣ Generating Caesar cipher puzzles with reasoning...")
    all_puzzles.extend(generate_caesar_cipher_with_reasoning(200))
    print(f"   ✓ Generated {len(all_puzzles)} puzzles")

    print("\n2️⃣ Generating RSA factorization puzzles with reasoning...")
    all_puzzles.extend(generate_rsa_puzzles_with_reasoning(150))
    print(f"   ✓ Total: {len(all_puzzles)} puzzles")

    print("\n3️⃣ Generating file identification puzzles with reasoning...")
    all_puzzles.extend(generate_file_identification_with_reasoning(200))
    print(f"   ✓ Total: {len(all_puzzles)} puzzles")

    print("\n4️⃣ Generating Base64 decoding puzzles with reasoning...")
    all_puzzles.extend(generate_base64_puzzles_with_reasoning(150))
    print(f"   ✓ Total: {len(all_puzzles)} puzzles")

    # Generate more from the original advanced generator
    print("\n5️⃣ Loading additional puzzles from advanced generator...")
    # We'll generate the rest up to total_puzzles count
    remaining = total_puzzles - len(all_puzzles)
    if remaining > 0:
        print(f"   (Need {remaining} more puzzles - will use existing advanced generator)")

    # Convert to training format with proper instruction structure
    print(f"\n6️⃣ Converting to training format...")
    training_data = []

    for puzzle in all_puzzles:
        instruction = f"### Instruction:\nSolve this CTF challenge ({puzzle['category']} - {puzzle['difficulty']}):\n\n{puzzle['puzzle']}\n\n### Response:\n"
        response = puzzle['solution']

        training_data.append({
            "text": instruction + response,
            "type": puzzle['type'],
            "difficulty": puzzle['difficulty'],
            "category": puzzle['category']
        })

    # Save to volume
    print(f"\n7️⃣ Saving {len(training_data)} puzzles to volume...")

    from pathlib import Path
    output_dir = Path("/ctf_data/enhanced_ctf_puzzles")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    with open(output_dir / "training_format.json", 'w') as f:
        json.dump(training_data, f, indent=2)

    # Save metadata
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump({
            "total_puzzles": len(training_data),
            "categories": {cat: sum(1 for p in all_puzzles if p['category'] == cat)
                          for cat in set(p['category'] for p in all_puzzles)},
            "difficulties": {diff: sum(1 for p in all_puzzles if p['difficulty'] == diff)
                           for diff in set(p['difficulty'] for p in all_puzzles)},
            "types": list(set(p['type'] for p in all_puzzles))
        }, f, indent=2)

    volume.commit()

    print(f"\n✅ SUCCESS!")
    print(f"   Total puzzles: {len(training_data)}")
    print(f"   All puzzles include FULL STEP-BY-STEP REASONING")
    print(f"   Saved to: /ctf_data/enhanced_ctf_puzzles/")

    # Show sample
    print(f"\n📝 Sample puzzle with reasoning:")
    print("="*60)
    print(training_data[0]['text'][:800])
    print("...")
    print("="*60)

    return {
        "total": len(training_data),
        "status": "success"
    }


@app.local_entrypoint()
def main():
    """Generate the enhanced CTF dataset"""
    print("🚀 Starting ENHANCED CTF puzzle generation with FULL REASONING...")
    result = generate_enhanced_ctf_dataset.remote(total_puzzles=700)
    print(f"\n✅ Complete: {result}")
