"""Test the improved markdown parser locally"""
import re
from pathlib import Path

def extract_subpuzzles_from_markdown(file_path: Path, puzzle_name: str) -> list:
    """Extract each sub-puzzle as a PROPER training example with Question → Solution+Reasoning."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    examples = []

    # Strategy 1: Extract structured Question/Solution/Method blocks (puzzle_01.md format)
    # Pattern: ### Title\n**Question**: ...\n**Solution**: ...\n**Method**: ...
    question_pattern = r'###\s+(.+?)\n\*\*Question\*\*:(.+?)(?:\n\*\*Solution\*\*:|$)(.+?)(?:\n\*\*Method\*\*:|$)(.+?)(?=\n###|\n---|\Z)'
    matches = re.finditer(question_pattern, content, re.DOTALL | re.IGNORECASE)

    for match in matches:
        title = match.group(1).strip()
        question = match.group(2).strip()

        # Try to extract solution and method
        full_match = match.group(0)
        solution_match = re.search(r'\*\*Solution\*\*:\s*(.+?)(?=\n\*\*Method\*\*:|\n###|\n---|\Z)', full_match, re.DOTALL)
        method_match = re.search(r'\*\*Method\*\*:\s*(.+?)(?=\n###|\n---|\Z)', full_match, re.DOTALL)

        if question and (solution_match or method_match):
            solution = solution_match.group(1).strip() if solution_match else ""
            method = method_match.group(1).strip() if method_match else ""

            # Create proper training format
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

    # Strategy 2: Extract Medium writeup format (arweave_puzzle_medium_writeups.md)
    # More flexible pattern to capture all sub-puzzle variations
    subpuzzle_pattern = r'####\s+Sub-Puzzle\s+\d+:\s+(.+?)(?=\n####|\n###|\Z)'
    matches = re.finditer(subpuzzle_pattern, content, re.DOTALL)

    for match in matches:
        full_section = match.group(0)
        title = match.group(1).split('\n')[0].strip()

        # Extract challenge (can be "Challenge:", "Challenge Text:", etc.)
        challenge_match = re.search(r'\*\*Challenge[^*]*:\*\*\s*(?:```\n)?(.*?)(?:\n```)?(?=\n\*\*|\Z)', full_section, re.DOTALL)
        challenge = challenge_match.group(1).strip() if challenge_match else ""

        # Extract solution/analysis (can be "Solution Process:", "Analysis:", "Pattern Recognition:", "Research Process:", etc.)
        solution_fields = ['Solution Process', 'Analysis', 'Pattern Recognition', 'Research Process', 'Chess Analysis', 'Pattern', 'Method']
        solution_text = ""

        for field in solution_fields:
            solution_match = re.search(rf'\*\*{field}[^*]*:\*\*\s*(?:```\n)?(.*?)(?:\n```)?(?=\n\*\*|\n####|\Z)', full_section, re.DOTALL)
            if solution_match:
                solution_text += solution_match.group(1).strip() + "\n\n"

        # Also capture **Solution:** field if present
        final_solution_match = re.search(r'\*\*Solution:\*\*\s*`?([^`\n]+)`?', full_section)
        if final_solution_match:
            solution_text += f"**Final Answer:** {final_solution_match.group(1).strip()}"

        if challenge and solution_text:
            instruction = f"### Instruction:\nSolve this ARweave puzzle challenge:\n\n**{title}**\n\n{challenge}\n\n### Response:\n"
            response = f"**Step-by-step solution:**\n\n{solution_text.strip()}"

            examples.append({
                "text": instruction + response,
                "source": f"arweave_writeup_{file_path.stem}",
                "difficulty": "hard"
            })

    # Strategy 3: Extract technique examples from comprehensive_guide.md
    # Pattern: **Example** (Puzzle X):\n```\n...\n```
    technique_pattern = r'\*\*Example\*\*.*?:\n```\n(.+?)\n```\n\n(?:\*\*.*?:\*\*\n)?(.*?)(?=\n---|\n###|\Z)'
    matches = re.finditer(technique_pattern, content, re.DOTALL)

    for match in matches:
        example = match.group(1).strip()
        explanation = match.group(2).strip() if match.group(2) else ""

        if len(example) > 50:  # Only substantial examples
            instruction = f"### Instruction:\nSolve this cryptopuzzle problem:\n\n{example}\n\n### Response:\n"
            response = f"**Solution:**\n\n{explanation}" if explanation else example

            examples.append({
                "text": instruction + response,
                "source": f"technique_example_{file_path.stem}",
                "difficulty": "medium"
            })

    # Strategy 4: Generic section extraction (fallback) - but extract ALL sections
    if len(examples) == 0:  # Only if no structured data found
        sections = re.split(r'\n###\s+', content)

        for i, section in enumerate(sections[1:], 1):  # Skip first (header)
            if len(section.strip()) < 150:  # Skip tiny sections
                continue

            lines = section.split('\n')
            sub_title = lines[0].strip()
            sub_content = '\n'.join(lines[1:]).strip()

            # Create training example
            instruction = f"### Instruction:\n[Puzzle: {puzzle_name}] Explain:\n\n**{sub_title}**\n\n### Response:\n"
            response = sub_content

            examples.append({
                "text": instruction + response,
                "source": f"general_{file_path.stem}",
                "difficulty": "medium"
            })

    return examples


# Test on ALL markdown files
repo_dir = Path("/tmp/cryptopuzzles")
total_examples = 0
file_counts = {}

print("Testing parser on ALL markdown files in cryptopuzzles repo:")
print("="*60)

# Process ALL markdown files recursively
for md_file in sorted(repo_dir.rglob("*.md")):
    # Skip hidden directories
    if any(part.startswith('.') for part in md_file.parts):
        continue

    examples = extract_subpuzzles_from_markdown(md_file, md_file.stem)

    if examples:
        rel_path = md_file.relative_to(repo_dir)
        print(f"\n{rel_path}: {len(examples)} examples")
        total_examples += len(examples)
        file_counts[str(rel_path)] = len(examples)

        # Show first example from this file
        if examples:
            print(f"  Sample: {examples[0]['text'][:150]}...")

print("\n" + "="*60)
print(f"TOTAL: {total_examples} examples from {len(file_counts)} files")
print("\nTop contributors:")
for path, count in sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {path}: {count} examples")
