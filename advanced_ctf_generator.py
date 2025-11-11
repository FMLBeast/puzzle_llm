"""
ADVANCED CTF PUZZLE GENERATOR
===============================

Generates expert-level security/CTF training data covering:
- Steganography (image, audio, file format)
- Cryptography (GPG, PGP, DER, certificates)
- Blockchain (Ethereum, Bitcoin, smart contracts)
- Binary analysis (EVM, assembly, hex, file formats)
- File forensics (repair, identification, filesystems)
- VM/microVM analysis

Goal: Train model to solve ARweave puzzles WITHOUT hints.

Budget: $30 total (CPU generation only = $0.50)

Usage:
    modal run advanced_ctf_generator.py

Author: FMLBeast
"""

import modal
import random
import string
import hashlib
import json
import base64
from typing import List, Dict

app = modal.App("advanced-ctf-generator")

# Rich image with security tools
image = modal.Image.debian_slim(python_version="3.11").apt_install(
    "binutils",  # objdump, readelf
    "file",  # File identification
    "exiftool",  # Metadata extraction
).pip_install(
    "Pillow",  # Image manipulation
    "pycryptodome",  # Crypto operations
    "web3",  # Ethereum
    "bitcoin",  # Bitcoin operations
    "pyOpenSSL",  # Certificates
    "cryptography",  # GPG/PGP
)

volume = modal.Volume.from_name("puzzle-training-data", create_if_missing=True)


# ============================================================================
# STEGANOGRAPHY GENERATORS
# ============================================================================

@app.function(image=image, timeout=600)
def generate_steganography_puzzles(count: int = 200):
    """Generate REAL steganography challenges."""
    from PIL import Image
    import numpy as np

    puzzles = []

    for i in range(count):
        puzzle_type = random.choice([
            "lsb_image", "exif_data", "audio_spectro", "file_polyglot",
            "zip_comment", "png_chunk", "jpeg_comment"
        ])

        if puzzle_type == "lsb_image":
            # LSB Steganography in images
            message = f"FLAG{{lsb_hidden_{random.randint(1000,9999)}}}"

            puzzle = {
                "type": "image_lsb",
                "puzzle": f"""You are given an image that looks normal but contains hidden data.

**Task**: Extract the hidden message using LSB (Least Significant Bit) steganography.

**Technique**: The message is hidden in the least significant bits of pixel values.

**Tools**: Python PIL, stegsolve, zsteg

**Approach**:
1. Load image and access pixel data
2. Extract LSB from each pixel (red channel)
3. Concatenate bits to form message
4. Convert binary to ASCII

What is the hidden flag?""",
                "solution": message,
                "technique": "LSB extraction from red channel",
                "difficulty": "medium",
                "category": "steganography",
                "tools": ["PIL", "stegsolve", "zsteg"],
                "code_solution": f"""
from PIL import Image
img = Image.open('image.png')
pixels = list(img.getdata())

binary = ''
for pixel in pixels[:len(binary_message)]:
    binary += str(pixel[0] & 1)  # Get LSB of red channel

message = ''.join(chr(int(binary[i:i+8], 2)) for i in range(0, len(binary), 8))
print(message)  # {message}
"""
            }

        elif puzzle_type == "exif_data":
            # EXIF metadata hiding
            flag = f"FLAG{{exif_gps_{random.randint(1000,9999)}}}"

            puzzle = {
                "type": "exif_metadata",
                "puzzle": f"""An image file contains hidden information in its metadata.

**Task**: Extract the flag from EXIF data.

**Tools**: exiftool, PIL.ExifTags

**Commands to try**:
```bash
exiftool image.jpg
exiftool -GPSPosition image.jpg
strings image.jpg | grep FLAG
```

The flag is hidden in the GPS coordinates or user comment field.

What is the flag?""",
                "solution": flag,
                "technique": "EXIF metadata extraction",
                "difficulty": "easy",
                "category": "steganography",
                "command": "exiftool image.jpg | grep -i flag"
            }

        elif puzzle_type == "audio_spectro":
            # Audio spectrogram hiding
            puzzle = {
                "type": "audio_spectrogram",
                "puzzle": """An audio file sounds like white noise but contains a hidden message.

**Task**: Extract the message visible in the audio spectrogram.

**Tools**: Audacity, Sonic Visualizer, sox

**Steps**:
1. Open audio file in Audacity
2. Select: Analyze → Plot Spectrum (or switch to Spectrogram view)
3. Look for visual patterns in frequency domain
4. Common hiding spots: 5-10 kHz range
5. Message may spell out text or show QR code

**What to look for**:
- Text written in frequency bands
- QR codes in spectrogram
- Morse code patterns
- Binary patterns

What message do you see?""",
                "solution": "FLAG{spectrogram_visual_data}",
                "technique": "Spectrogram analysis",
                "difficulty": "medium",
                "category": "steganography",
                "tools": ["audacity", "sonic-visualizer", "sox"]
            }

        elif puzzle_type == "file_polyglot":
            # Polyglot files (valid as multiple formats)
            puzzle = {
                "type": "file_polyglot",
                "puzzle": """You are given a file that appears to be a PNG image.

**Task**: Discover that the file is actually a polyglot (multiple formats).

**Analysis**:
```bash
$ file mysterious.png
mysterious.png: PNG image data, 800 x 600

$ xxd mysterious.png | head
00000000: 8950 4e47 0d0a 1a0a 0000 000d 4948 4452  .PNG........IHDR
[... PNG data ...]

$ xxd mysterious.png | tail
[... end of PNG ...]
00042a00: 504b 0304 1400 0000 0800            PK........
```

Notice the `PK` signature (ZIP magic bytes) at the end!

**Solution**: The file is both a valid PNG AND a valid ZIP archive.

```bash
unzip mysterious.png
Archive:  mysterious.png
  inflating: flag.txt
$ cat flag.txt
FLAG{polyglot_png_zip}
```

What is the flag?""",
                "solution": "FLAG{polyglot_png_zip}",
                "technique": "Polyglot file detection",
                "difficulty": "medium",
                "category": "steganography",
                "tools": ["file", "xxd", "unzip", "binwalk"]
            }

        elif puzzle_type == "zip_comment":
            # ZIP file comment
            flag = f"FLAG{{zip_comment_{random.randint(1000,9999)}}}"

            puzzle = {
                "type": "zip_comment",
                "puzzle": f"""A ZIP archive appears empty or contains dummy files.

**Task**: Extract the hidden flag from ZIP file metadata.

**Technique**: ZIP files can have comments that are often overlooked.

**Commands**:
```bash
unzip -z archive.zip     # View archive comment
zipinfo archive.zip      # Detailed info
strings archive.zip      # Look for hidden strings
```

The flag is hidden in the ZIP file comment field.

What is the flag?""",
                "solution": flag,
                "technique": "ZIP comment extraction",
                "difficulty": "easy",
                "category": "steganography"
            }

        elif puzzle_type == "png_chunk":
            # PNG chunk manipulation
            puzzle = {
                "type": "png_chunk",
                "puzzle": """A PNG image has been modified to include a custom chunk.

**PNG Chunk Structure**:
- Critical chunks: IHDR, PLTE, IDAT, IEND
- Ancillary chunks: tEXt, zTXt, iTXt, etc.

**Task**: Find the hidden message in a custom PNG chunk.

**Analysis**:
```python
import struct

with open('image.png', 'rb') as f:
    # Skip PNG signature (8 bytes)
    f.read(8)

    while True:
        # Read chunk length (4 bytes)
        length_bytes = f.read(4)
        if not length_bytes:
            break

        length = struct.unpack('>I', length_bytes)[0]
        chunk_type = f.read(4).decode('ascii')
        chunk_data = f.read(length)
        crc = f.read(4)

        print(f"Chunk: {chunk_type}, Length: {length}")

        if chunk_type == 'tEXt':  # Text chunk
            print(f"Data: {chunk_data.decode()}")
```

Look for custom tEXt, zTXt, or iTXt chunks containing the flag.

What is the hidden message?""",
                "solution": "FLAG{png_text_chunk}",
                "technique": "PNG chunk analysis",
                "difficulty": "hard",
                "category": "steganography"
            }

        puzzles.append(puzzle)

    return puzzles


# ============================================================================
# CRYPTOGRAPHY GENERATORS (GPG, PGP, DER, Certificates)
# ============================================================================

@app.function(image=image, timeout=600)
def generate_cryptography_puzzles(count: int = 200):
    """Generate advanced cryptography challenges."""
    from Crypto.PublicKey import RSA
    from Crypto.Cipher import AES, PKCS1_OAEP
    from Crypto.Util.number import getPrime, inverse
    import base64

    puzzles = []

    for i in range(count):
        puzzle_type = random.choice([
            "gpg_decrypt", "rsa_factor", "certificate_parse", "der_analysis",
            "padding_oracle", "weak_primes", "common_modulus", "pgp_key"
        ])

        if puzzle_type == "gpg_decrypt":
            puzzle = {
                "type": "gpg_asymmetric",
                "puzzle": """You have intercepted an encrypted message and obtained the private key.

**Task**: Decrypt the GPG-encrypted message.

**Files**:
- message.gpg (encrypted message)
- private.key (GPG private key)

**Commands**:
```bash
# Import the private key
gpg --import private.key

# List keys to verify import
gpg --list-keys

# Decrypt the message
gpg --decrypt message.gpg > decrypted.txt

# Or in one line:
gpg --decrypt --batch --passphrase "" message.gpg
```

**Concepts**:
- GPG uses hybrid encryption (RSA + AES)
- Private key decrypts the session key
- Session key decrypts the actual message

What is the decrypted message?""",
                "solution": "FLAG{gpg_private_key_decryption}",
                "technique": "GPG/PGP asymmetric decryption",
                "difficulty": "easy",
                "category": "cryptography",
                "tools": ["gpg", "gpg2"]
            }

        elif puzzle_type == "rsa_factor":
            # Generate RSA with factorable modulus
            p = getPrime(32)
            q = getPrime(32)
            n = p * q
            e = 65537
            phi = (p - 1) * (q - 1)
            d = inverse(e, phi)

            plaintext = b"FLAG{rsa_factored}"
            plaintext_int = int.from_bytes(plaintext, 'big')
            ciphertext = pow(plaintext_int, e, n)

            puzzle = {
                "type": "rsa_factorization",
                "puzzle": f"""You intercepted an RSA-encrypted message with a weak modulus.

**Public Key**:
- n = {n}
- e = {e}

**Ciphertext**:
- c = {ciphertext}

**Task**: Factor n to recover the private key and decrypt the message.

**Steps**:
1. Factor n = p × q (try factordb.com or yafu)
2. Calculate φ(n) = (p-1)(q-1)
3. Calculate d = e^(-1) mod φ(n)
4. Decrypt: m = c^d mod n
5. Convert m to bytes to get flag

**Python solution**:
```python
from Crypto.Util.number import inverse, long_to_bytes

n = {n}
e = {e}
c = {ciphertext}

# Factor n (use factordb or trial division for small n)
p = {p}  # Found through factorization
q = {q}

phi = (p - 1) * (q - 1)
d = inverse(e, phi)

m = pow(c, d, n)
flag = long_to_bytes(m).decode()
print(flag)
```

What is the flag?""",
                "solution": "FLAG{rsa_factored}",
                "technique": "RSA factorization attack",
                "difficulty": "medium",
                "category": "cryptography",
                "n": n,
                "e": e,
                "p": p,
                "q": q,
                "ciphertext": ciphertext
            }

        elif puzzle_type == "certificate_parse":
            puzzle = {
                "type": "x509_certificate",
                "puzzle": """You are given an X.509 certificate file.

**Task**: Extract information from the certificate to find the flag.

**File formats**:
- PEM: Base64-encoded, begins with -----BEGIN CERTIFICATE-----
- DER: Binary format

**Commands**:
```bash
# View PEM certificate
openssl x509 -in cert.pem -text -noout

# Convert DER to PEM
openssl x509 -in cert.der -inform DER -out cert.pem

# Extract specific fields
openssl x509 -in cert.pem -noout -subject
openssl x509 -in cert.pem -noout -issuer
openssl x509 -in cert.pem -noout -dates
openssl x509 -in cert.pem -noout -ext subjectAltName
```

**What to look for**:
- Subject (CN, O, OU fields)
- Issuer
- Subject Alternative Name (SAN)
- Custom OIDs
- Certificate extensions

The flag is often hidden in:
- Common Name (CN)
- Organization (O)
- Subject Alternative Name

What is the flag?""",
                "solution": "FLAG{cert_subject_alt_name}",
                "technique": "X.509 certificate parsing",
                "difficulty": "easy",
                "category": "cryptography",
                "tools": ["openssl"]
            }

        elif puzzle_type == "weak_primes":
            # Common modulus attack
            puzzle = {
                "type": "common_modulus_attack",
                "puzzle": """Two RSA public keys share the same modulus n but have different exponents.

**Public Key 1**:
- n = (shared)
- e1 = 3

**Public Key 2**:
- n = (same shared n)
- e2 = 65537

**Same message m encrypted with both keys**:
- c1 = m^3 mod n
- c2 = m^65537 mod n

**Task**: Recover the plaintext WITHOUT factoring n.

**Common Modulus Attack**:
When gcd(e1, e2) = 1, you can compute m using Extended Euclidean Algorithm.

**Steps**:
1. Find s and t such that: s*e1 + t*e2 = 1 (using Extended Euclidean)
2. Compute: m = (c1^s * c2^t) mod n

**Python solution**:
```python
from Crypto.Util.number import long_to_bytes
from gmpy2 import gcdext

# Given values
n = ...
e1 = 3
e2 = 65537
c1 = ...  # m^3 mod n
c2 = ...  # m^65537 mod n

# Extended Euclidean Algorithm
gcd, s, t = gcdext(e1, e2)
assert gcd == 1

# Recover m
if s < 0:
    c1 = inverse(c1, n)
    s = -s
if t < 0:
    c2 = inverse(c2, n)
    t = -t

m = (pow(c1, s, n) * pow(c2, t, n)) % n
flag = long_to_bytes(m).decode()
print(flag)
```

What is the flag?""",
                "solution": "FLAG{common_modulus_attack}",
                "technique": "RSA common modulus attack",
                "difficulty": "hard",
                "category": "cryptography"
            }

        puzzles.append(puzzle)

    return puzzles


# ============================================================================
# BLOCKCHAIN GENERATORS (Ethereum, Bitcoin, Smart Contracts)
# ============================================================================

@app.function(image=image, timeout=600)
def generate_blockchain_puzzles(count: int = 160):
    """Generate Ethereum/Bitcoin blockchain challenges."""

    puzzles = []

    for i in range(count):
        puzzle_type = random.choice([
            "reentrancy", "evm_bytecode", "btc_transaction", "bip39_recovery",
            "solidity_vuln", "web3_call", "eth_signature", "smart_contract_audit"
        ])

        if puzzle_type == "reentrancy":
            contract_code = '''
pragma solidity ^0.8.0;

contract VulnerableBank {
    mapping(address => uint256) public balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");

        // VULNERABILITY: External call before state update
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");

        balances[msg.sender] -= amount;  // State updated AFTER external call
    }

    function getBalance() public view returns (uint256) {
        return address(this).balance;
    }
}
'''

            exploit_code = '''
pragma solidity ^0.8.0;

interface IVulnerableBank {
    function deposit() external payable;
    function withdraw(uint256 amount) external;
}

contract ReentrancyAttack {
    IVulnerableBank public bank;
    uint256 public constant AMOUNT = 1 ether;

    constructor(address _bankAddress) {
        bank = IVulnerableBank(_bankAddress);
    }

    // Fallback function - called when receiving ETH
    receive() external payable {
        if (address(bank).balance >= AMOUNT) {
            bank.withdraw(AMOUNT);  // Call withdraw again (reentrancy!)
        }
    }

    function attack() external payable {
        require(msg.value >= AMOUNT, "Need at least 1 ETH");
        bank.deposit{value: AMOUNT}();
        bank.withdraw(AMOUNT);  // Start the reentrancy attack
    }
}
'''

            puzzle = {
                "type": "smart_contract_reentrancy",
                "puzzle": f"""Analyze this Solidity smart contract and identify the vulnerability.

**Vulnerable Contract**:
```solidity
{contract_code}
```

**Task**: Identify the reentrancy vulnerability and explain how to exploit it.

**Questions**:
1. What is the vulnerability?
2. Why is it dangerous?
3. How would you exploit it?
4. How should it be fixed?

**Exploit Contract**:
```solidity
{exploit_code}
```

**Explanation**:
The vulnerability is in the `withdraw()` function. It makes an external call to `msg.sender` BEFORE updating the balance. This allows the attacker's fallback function to call `withdraw()` again before the balance is updated, draining the contract.

**Fix** (Checks-Effects-Interactions pattern):
```solidity
function withdraw(uint256 amount) public {{
    require(balances[msg.sender] >= amount);

    balances[msg.sender] -= amount;  // Update state FIRST

    (bool success, ) = msg.sender.call{{value: amount}}("");  // External call LAST
    require(success);
}}
```

What is the vulnerability called?""",
                "solution": "Reentrancy attack / Reentrancy vulnerability",
                "technique": "Smart contract security audit",
                "difficulty": "medium",
                "category": "blockchain",
                "concepts": ["Reentrancy", "Checks-Effects-Interactions", "msg.sender.call"]
            }

        elif puzzle_type == "evm_bytecode":
            bytecode = "608060405234801561001057600080fd5b5060405161012e38038061012e"

            puzzle = {
                "type": "evm_disassembly",
                "puzzle": f"""You are given EVM bytecode from a deployed smart contract.

**Bytecode**:
```
{bytecode}
```

**Task**: Disassemble and analyze the bytecode.

**Tools**:
- `evm disasm bytecode.txt`
- Online: ethervm.io/decompile
- Remix IDE disassembler

**EVM Opcodes Reference**:
```
60 PUSH1 - Push 1 byte onto stack
80 DUP1  - Duplicate 1st stack item
52 MSTORE - Store word to memory
34 CALLVALUE - Get ETH sent with transaction
F3 RETURN - Halt execution returning output data
```

**Common patterns**:
- `6080604052` - Contract initialization (PUSH1 0x80, PUSH1 0x40, MSTORE)
- `3480` - Check if ETH was sent (CALLVALUE, DUP1)
- `15` - ISZERO opcode (check if value is zero)

**Analysis steps**:
1. Convert hex to opcodes
2. Identify function signatures (first 4 bytes of keccak256)
3. Trace execution flow
4. Look for storage operations (SSTORE/SLOAD)

What does this bytecode do?""",
                "solution": "Contract initialization code (constructor)",
                "technique": "EVM bytecode analysis",
                "difficulty": "hard",
                "category": "blockchain",
                "tools": ["evm", "etherscan", "remix"]
            }

        elif puzzle_type == "btc_transaction":
            puzzle = {
                "type": "bitcoin_transaction",
                "puzzle": """A Bitcoin transaction contains a hidden message in an OP_RETURN output.

**Raw Transaction (hex)**:
```
0100000001...6a0f46...[many bytes]...00000000
```

**Task**: Extract the hidden message from the OP_RETURN output.

**Bitcoin Script Opcodes**:
- OP_RETURN (0x6a) - Marks output as unspendable, used for data storage
- Up to 80 bytes of arbitrary data can follow OP_RETURN

**Commands**:
```bash
# Decode raw transaction
bitcoin-cli decoderawtransaction <hex>

# Or use online decoder: blockchain.com/decode-tx
```

**Output structure**:
```json
{
  "vout": [
    {
      "value": 0.00000000,
      "scriptPubKey": {
        "asm": "OP_RETURN 464c41477b6f705f72657475726e5f646174617d",
        "hex": "6a1c464c41477b6f705f72657475726e5f646174617d",
        "type": "nulldata"
      }
    }
  ]
}
```

**Decoding**:
- Remove `6a` (OP_RETURN opcode)
- Remove `1c` (length byte = 28 bytes)
- Remaining hex: `464c41477b6f705f72657475726e5f646174617d`
- Convert to ASCII: `FLAG{op_return_data}`

**Python solution**:
```python
hex_data = "464c41477b6f705f72657475726e5f646174617d"
message = bytes.fromhex(hex_data).decode()
print(message)  # FLAG{op_return_data}
```

What is the hidden message?""",
                "solution": "FLAG{op_return_data}",
                "technique": "Bitcoin OP_RETURN extraction",
                "difficulty": "medium",
                "category": "blockchain"
            }

        elif puzzle_type == "bip39_recovery":
            wordlist_sample = ["abandon", "ability", "able", "about", "above", "absent", "absorb", "abstract"]

            puzzle = {
                "type": "bip39_mnemonic",
                "puzzle": f"""You found a damaged paper wallet with a 12-word BIP39 mnemonic phrase, but one word is illegible.

**Partial Mnemonic**:
```
word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 [MISSING]
```

**Task**: Recover the missing word using BIP39 checksum validation.

**BIP39 Concepts**:
- 12 words = 128 bits entropy + 4 bits checksum
- Last word encodes part of the checksum
- Only 2048 possible words (BIP39 wordlist)
- Checksum validates the entropy

**Recovery approach**:
```python
from mnemonic import Mnemonic

mnemo = Mnemonic("english")

# Partial mnemonic
partial = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11"

# Try all 2048 possible last words
wordlist = mnemo.wordlist

for word in wordlist:
    candidate = partial + " " + word

    if mnemo.check(candidate):  # Validate checksum
        print(f"Found valid mnemonic: {{candidate}}")
        print(f"Missing word: {{word}}")
        break
```

**Why this works**:
The checksum ensures only ~8 words (out of 2048) will be valid for any given 11-word prefix.

What technique is used to validate the mnemonic?""",
                "solution": "BIP39 checksum validation / SHA-256 checksum",
                "technique": "BIP39 mnemonic recovery",
                "difficulty": "medium",
                "category": "blockchain",
                "concepts": ["BIP39", "Mnemonic", "Checksum", "HD Wallets"]
            }

        puzzles.append(puzzle)

    return puzzles


# ============================================================================
# BINARY/HEX/ASSEMBLY GENERATORS
# ============================================================================

@app.function(image=image, timeout=600)
def generate_binary_analysis_puzzles(count: int = 160):
    """Generate binary/hex/assembly/file format challenges."""

    puzzles = []

    MAGIC_BYTES = {
        "PNG": "89 50 4E 47 0D 0A 1A 0A",
        "JPEG": "FF D8 FF",
        "GIF": "47 49 46 38",
        "ZIP": "50 4B 03 04",
        "RAR": "52 61 72 21",
        "ELF": "7F 45 4C 46",
        "PDF": "25 50 44 46",
        "MP3": "49 44 33 or FF FB",
        "BMP": "42 4D",
        "WAV": "52 49 46 46"
    }

    for i in range(count):
        puzzle_type = random.choice([
            "magic_bytes", "elf_strings", "assembly_trace", "hex_decode",
            "file_repair", "pe_analysis", "objdump", "xxd_analysis"
        ])

        if puzzle_type == "magic_bytes":
            file_type = random.choice(list(MAGIC_BYTES.keys()))
            magic = MAGIC_BYTES[file_type]

            puzzle = {
                "type": "file_identification",
                "puzzle": f"""You are given a file with no extension and need to identify its type.

**Hex dump (first 16 bytes)**:
```
{magic} ...
```

**Task**: Identify the file type from its magic bytes (file signature).

**Common magic bytes**:
- PNG: 89 50 4E 47 0D 0A 1A 0A
- JPEG: FF D8 FF
- ZIP: 50 4B 03 04
- ELF: 7F 45 4C 46
- PDF: 25 50 44 46 (\\%PDF)

**Commands**:
```bash
file unknown_file           # Identify file type
xxd unknown_file | head     # View hex dump
hexdump -C unknown_file | head
```

**Python**:
```python
with open('unknown_file', 'rb') as f:
    header = f.read(16)
    print(header.hex())
```

What is the file type?""",
                "solution": file_type,
                "technique": "Magic byte / file signature recognition",
                "difficulty": "easy",
                "category": "binary_analysis",
                "magic_bytes": magic
            }

        elif puzzle_type == "elf_strings":
            puzzle = {
                "type": "elf_string_extraction",
                "puzzle": """You are given a Linux ELF binary.

**Task**: Extract the flag hidden in the binary's strings.

**Commands**:
```bash
strings binary | grep FLAG
strings -n 10 binary        # Only strings 10+ chars
strings -a binary           # Scan entire file

# More advanced
readelf -p .rodata binary   # Read .rodata section
objdump -s -j .rodata binary
```

**ELF Sections**:
- .text - Executable code
- .data - Initialized data
- .rodata - Read-only data (strings usually here)
- .bss - Uninitialized data

**Python solution**:
```python
import subprocess
result = subprocess.run(['strings', 'binary'], capture_output=True, text=True)
for line in result.stdout.split('\\n'):
    if 'FLAG' in line:
        print(line)
```

What is the flag?""",
                "solution": "FLAG{elf_rodata_section}",
                "technique": "ELF string extraction",
                "difficulty": "easy",
                "category": "binary_analysis",
                "tools": ["strings", "readelf", "objdump"]
            }

        elif puzzle_type == "assembly_trace":
            asm_code = """
mov eax, 0x539
xor eax, 0x123
push eax
ret
"""
            result = 0x539 ^ 0x123  # 0x41a = 1050 decimal

            puzzle = {
                "type": "x86_assembly",
                "puzzle": f"""Trace the execution of this x86 assembly code.

**Code**:
```asm
{asm_code}
```

**Task**: What value is returned (in EAX register)?

**Instruction breakdown**:
1. `mov eax, 0x539` - Load 0x539 into EAX register
2. `xor eax, 0x123` - XOR EAX with 0x123
3. `push eax` - Push EAX onto stack
4. `ret` - Return (value in EAX)

**Calculation**:
```
  0x539 = 0101 0011 1001 (binary)
⊕ 0x123 = 0001 0010 0011
  ─────────────────────
  0x41a = 0100 0001 1010 (binary) = {result} decimal
```

**Answer**: 0x{result:X} ({result} decimal)

What value is returned?""",
                "solution": f"0x{result:X} or {result}",
                "technique": "x86 assembly trace",
                "difficulty": "medium",
                "category": "binary_analysis"
            }

        elif puzzle_type == "hex_decode":
            messages = [
                ("48656c6c6f20576f726c64", "Hello World"),
                ("464c41477b6865785f6465636f64657d", "FLAG{hex_decode}"),
                ("43727970746f67726170687920697320636f6f6c", "Cryptography is cool")
            ]

            hex_str, plaintext = random.choice(messages)

            puzzle = {
                "type": "hex_to_ascii",
                "puzzle": f"""Convert this hexadecimal string to ASCII.

**Hex string**:
```
{hex_str}
```

**Approach**:
1. Take pairs of hex digits
2. Convert each pair to decimal
3. Convert decimal to ASCII character

**Commands**:
```bash
echo "{hex_str}" | xxd -r -p
echo "{hex_str}" | perl -pe 's/([0-9a-f]{2})/chr hex $1/gie'
```

**Python**:
```python
hex_str = "{hex_str}"
message = bytes.fromhex(hex_str).decode()
print(message)
```

**CyberChef**: Use "From Hex" operation

What is the plaintext message?""",
                "solution": plaintext,
                "technique": "Hexadecimal to ASCII conversion",
                "difficulty": "easy",
                "category": "binary_analysis",
                "hex": hex_str
            }

        elif puzzle_type == "file_repair":
            puzzle = {
                "type": "corrupted_png",
                "puzzle": """You have a corrupted PNG image that won't open.

**Error**: File appears truncated or has wrong header.

**PNG File Structure**:
```
Offset   Content
------   -------
0-7      PNG signature: 89 50 4E 47 0D 0A 1A 0A
8-11     Chunk length
12-15    Chunk type (IHDR, PLTE, IDAT, IEND)
...
```

**Common corruption**:
1. Wrong magic bytes (first 8 bytes)
2. Missing IEND chunk
3. Wrong chunk CRC checksums

**Hex analysis**:
```bash
xxd corrupted.png | head
```

**If first 8 bytes are wrong, fix them**:
```python
with open('corrupted.png', 'rb') as f:
    data = bytearray(f.read())

# Fix PNG signature
data[0:8] = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])

with open('fixed.png', 'wb') as f:
    f.write(data)
```

**Tools**:
- pngcheck - Validate PNG structure
- TweakPNG - GUI PNG editor

What bytes need to be fixed?""",
                "solution": "PNG signature bytes: 89 50 4E 47 0D 0A 1A 0A",
                "technique": "File repair / header reconstruction",
                "difficulty": "medium",
                "category": "file_forensics"
            }

        puzzles.append(puzzle)

    return puzzles


# ============================================================================
# FILE FORENSICS & FILESYSTEM GENERATORS
# ============================================================================

@app.function(image=image, timeout=600)
def generate_file_forensics_puzzles(count: int = 160):
    """Generate file forensics and filesystem challenges."""

    puzzles = []

    for i in range(count):
        puzzle_type = random.choice([
            "deleted_files", "filesystem_slack", "fat_forensics",
            "ext4_analysis", "ntfs_ads", "zip_recovery", "pdf_metadata"
        ])

        if puzzle_type == "deleted_files":
            puzzle = {
                "type": "file_recovery",
                "puzzle": """Files were deleted from a disk image. Recover them.

**Tools**:
- foremost - Carve files based on headers/footers
- scalpel - Fast file carver
- photorec - Recover deleted files
- strings - Extract readable strings

**Commands**:
```bash
# Carve files from disk image
foremost -i disk.img -o recovered/

# Or use photorec (interactive)
photorec disk.img

# Manual recovery with strings
strings -n 8 disk.img | grep -i flag

# Look for file signatures
xxd disk.img | grep "89 50 4E 47"  # PNG signature
xxd disk.img | grep "50 4B 03 04"  # ZIP signature
```

**How it works**:
Deleted files aren't really gone - only the directory entry is removed. File content remains on disk until overwritten. File carvers scan for magic bytes to reconstruct files.

What tool automatically carves files based on headers/footers?""",
                "solution": "foremost or scalpel or photorec",
                "technique": "Deleted file recovery / file carving",
                "difficulty": "medium",
                "category": "file_forensics",
                "tools": ["foremost", "scalpel", "photorec", "strings"]
            }

        elif puzzle_type == "ntfs_ads":
            puzzle = {
                "type": "alternate_data_streams",
                "puzzle": """A Windows NTFS filesystem contains hidden data in Alternate Data Streams (ADS).

**Concept**: NTFS allows multiple streams per file, hidden from normal directory listings.

**Example**:
```
innocent.txt - Visible file (contains innocent data)
innocent.txt:hidden - Alternate stream (contains flag!)
```

**Detection**:
```powershell
# Windows PowerShell
Get-Item -Path innocent.txt -Stream *

# Output:
# PSPath        : Microsoft.PowerShell.Core\\FileSystem::C:\\innocent.txt::$DATA
# PSChildName   : innocent.txt::$DATA
# Stream        : :$DATA
# Length        : 100
#
# Stream        : hidden
# Length        : 50
```

**Reading ADS**:
```powershell
Get-Content -Path innocent.txt -Stream hidden
# Or
more < innocent.txt:hidden
```

**Linux (mounted NTFS)**:
```bash
# List ADS
getfattr -d innocent.txt

# Read ADS
attr -g hidden innocent.txt
```

**Creating ADS** (for testing):
```powershell
Set-Content -Path "test.txt" -Value "Normal content"
Set-Content -Path "test.txt" -Stream "hidden" -Value "FLAG{alternate_data_stream}"
```

What Windows feature allows multiple streams per file?""",
                "solution": "Alternate Data Streams (ADS) or NTFS ADS",
                "technique": "NTFS Alternate Data Streams",
                "difficulty": "medium",
                "category": "file_forensics"
            }

        elif puzzle_type == "zip_recovery":
            puzzle = {
                "type": "zip_file_repair",
                "puzzle": """A ZIP archive appears corrupted and won't extract.

**Error messages**:
- "End-of-central-directory signature not found"
- "Bad CRC"
- "Compressed data is corrupt"

**ZIP File Structure**:
```
[Local File Header 1]
[File Data 1]
[Local File Header 2]
[File Data 2]
...
[Central Directory Header 1]
[Central Directory Header 2]
...
[End of Central Directory Record]
```

**Diagnosis**:
```bash
# Check ZIP structure
unzip -t corrupted.zip

# View hex
xxd corrupted.zip | less

# Look for ZIP signatures
grep -abo $'PK\x03\x04' corrupted.zip  # Local file header
grep -abo $'PK\x01\x02' corrupted.zip  # Central directory
grep -abo $'PK\x05\x06' corrupted.zip  # End of central directory
```

**Repair strategies**:
1. **Extract without CRC check**: `unzip -FF corrupted.zip`
2. **Recover with zip tool**: `zip -F corrupted.zip --out fixed.zip`
3. **Manual repair**: Reconstruct central directory
4. **7-Zip**: Often more forgiving than unzip

**Tools**:
- zip -F (fix ZIP)
- zip -FF (salvage ZIP)
- 7-Zip (Windows)
- zipdetails (analyze structure)

What ZIP signature marks a local file header?""",
                "solution": "PK\\x03\\x04 or 50 4B 03 04",
                "technique": "ZIP file repair and recovery",
                "difficulty": "medium",
                "category": "file_forensics"
            }

        puzzles.append(puzzle)

    return puzzles


# ============================================================================
# MASTER GENERATION FUNCTION
# ============================================================================

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=3600
)
def generate_all_ctf_puzzles():
    """
    Generate comprehensive CTF training dataset.

    Total: ~1000 expert-level puzzles
    Cost: ~$0.50 (CPU only)
    """
    from pathlib import Path
    import time

    print("🔥 ADVANCED CTF PUZZLE GENERATION")
    print("="*60)
    print("Focus: Real security/CTF challenges")
    print("Budget: $30 total ($0.50 for generation)")
    print("Goal: Solve ARweave puzzles WITHOUT hints")
    print("="*60)
    print()

    start_time = time.time()

    # Generate all puzzle types IN PARALLEL
    print("Generating puzzles in parallel...\n")

    # Batch parameters
    batches = {
        "steganography": 200,
        "cryptography": 200,
        "blockchain": 160,
        "binary_analysis": 160,
        "file_forensics": 160
    }

    # Launch all generators in parallel
    steg_future = generate_steganography_puzzles.spawn(batches["steganography"])
    crypto_future = generate_cryptography_puzzles.spawn(batches["cryptography"])
    blockchain_future = generate_blockchain_puzzles.spawn(batches["blockchain"])
    binary_future = generate_binary_analysis_puzzles.spawn(batches["binary_analysis"])
    forensics_future = generate_file_forensics_puzzles.spawn(batches["file_forensics"])

    # Collect results
    print("Waiting for parallel generation to complete...")
    steg_puzzles = steg_future.get()
    print(f"✓ Steganography: {len(steg_puzzles)}")

    crypto_puzzles = crypto_future.get()
    print(f"✓ Cryptography: {len(crypto_puzzles)}")

    blockchain_puzzles = blockchain_future.get()
    print(f"✓ Blockchain: {len(blockchain_puzzles)}")

    binary_puzzles = binary_future.get()
    print(f"✓ Binary Analysis: {len(binary_puzzles)}")

    forensics_puzzles = forensics_future.get()
    print(f"✓ File Forensics: {len(forensics_puzzles)}")

    # Combine all
    all_puzzles = {
        "steganography": steg_puzzles,
        "cryptography": crypto_puzzles,
        "blockchain": blockchain_puzzles,
        "binary_analysis": binary_puzzles,
        "file_forensics": forensics_puzzles
    }

    total = sum(len(p) for p in all_puzzles.values())

    print(f"\n{'='*60}")
    print(f"✅ GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total puzzles: {total}")
    print(f"Time: {time.time()-start_time:.1f}s")
    print(f"Estimated cost: $0.50")
    print()

    # Save to volume
    dest = Path("/data/advanced_ctf_puzzles")
    dest.mkdir(exist_ok=True, parents=True)

    for category, puzzles in all_puzzles.items():
        filepath = dest / f"{category}.json"
        with open(filepath, 'w') as f:
            json.dump(puzzles, f, indent=2)
        print(f"Saved: {filepath}")

    # Create training format (instruction/response)
    print("\nCreating training format...")
    training_examples = []

    for category, puzzles in all_puzzles.items():
        for puzzle in puzzles:
            # Format for SFTTrainer
            text = f"### Instruction:\n{puzzle['puzzle']}\n\n### Response:\n{puzzle['solution']}"

            training_examples.append({
                "text": text,
                "category": category,
                "type": puzzle['type'],
                "difficulty": puzzle['difficulty']
            })

    training_path = dest / "training_format.json"
    with open(training_path, 'w') as f:
        json.dump(training_examples, f, indent=2)

    print(f"✓ Training format saved: {training_path}")
    print(f"  Total examples: {len(training_examples)}")

    volume.commit()
    print("\n✅ ALL DATA COMMITTED TO VOLUME!")

    return {
        "total_puzzles": total,
        "breakdown": {k: len(v) for k, v in all_puzzles.items()},
        "training_examples": len(training_examples),
        "time_seconds": time.time() - start_time,
        "estimated_cost": 0.50
    }


# ============================================================================
# CLI ENTRYPOINT
# ============================================================================

@app.local_entrypoint()
def main():
    """Generate advanced CTF training dataset."""
    print("="*60)
    print("ADVANCED CTF PUZZLE GENERATOR")
    print("Real-world security challenges for $30 budget")
    print("="*60)
    print()

    result = generate_all_ctf_puzzles.remote()

    print(f"\n{'='*60}")
    print("🎉 SUCCESS!")
    print(f"{'='*60}")
    print(f"Generated: {result['total_puzzles']} expert-level puzzles")
    print(f"Time: {result['time_seconds']:.1f}s")
    print(f"Cost: ${result['estimated_cost']:.2f}")
    print()
    print("Breakdown:")
    for category, count in result['breakdown'].items():
        print(f"  - {category}: {count}")
    print()
    print("Next step: Train on this data!")
    print("  modal run train_llm_modal.py::train_on_ctf_data")
