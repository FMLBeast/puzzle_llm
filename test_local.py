#!/usr/bin/env python3
"""
Local testing script for trained puzzle-solving models.
Run inference on your trained models locally.

Usage:
    # Test primary model
    python test_local.py --model-dir ./models/primary/final --prompt "Solve: KHOOR ZRUOG"

    # Test benchmark model
    python test_local.py --model-dir ./models/benchmark/final --prompt "Solve: KHOOR ZRUOG"

    # Interactive mode
    python test_local.py --model-dir ./models/primary/final --interactive
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os


DEFAULT_PROMPTS = [
    "### Instruction:\nSolve this Caesar cipher:\n\nKHOOR ZRUOG\n\nHint: Try shift values 1-25\n\n### Response:\n",
    "### Instruction:\nDecode this Base64:\n\nSGVsbG8gV29ybGQ=\n\n### Response:\n",
    "### Instruction:\nSolve this ROT13 cipher:\n\nUryyb Jbeyq\n\n### Response:\n",
]


def load_model(model_dir: Path, base_model_name: str = None, hf_token: str = None):
    """Load a trained model from directory."""
    print(f"🔍 Loading model from {model_dir}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), token=hf_token)

    # Detect base model name from config if not provided
    if base_model_name is None:
        # Try to read from adapter_config.json
        import json
        config_file = model_dir / "adapter_config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
                base_model_name = config.get('base_model_name_or_path')

    if base_model_name is None:
        print("⚠️  Could not detect base model. Please specify with --base-model")
        return None, None

    print(f"  Base model: {base_model_name}")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        token=hf_token,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, str(model_dir))
    model.eval()

    print("  ✅ Model loaded successfully!")

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate a response from the model."""

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")

    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = inputs.to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response


def interactive_mode(model, tokenizer):
    """Run interactive testing session."""
    print("\n" + "=" * 80)
    print("🤖 INTERACTIVE MODE")
    print("=" * 80)
    print("Enter your puzzle prompts. Type 'quit' or 'exit' to stop.")
    print("Type 'example' to see sample prompts.\n")

    while True:
        try:
            user_input = input("\n👤 You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if user_input.lower() == 'example':
                print("\n📝 Sample prompts:")
                for i, prompt in enumerate(DEFAULT_PROMPTS, 1):
                    print(f"\n{i}. {prompt[:100]}...")
                continue

            if not user_input:
                continue

            # Format as instruction if not already formatted
            if "### Instruction:" not in user_input:
                prompt = f"### Instruction:\nSolve this puzzle:\n\n{user_input}\n\n### Response:\n"
            else:
                prompt = user_input

            print("\n🤖 Model:", end=" ", flush=True)
            response = generate_response(model, tokenizer, prompt)

            # Extract just the response part
            if "### Response:" in response:
                response = response.split("### Response:")[-1].strip()

            print(response)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def batch_test(model, tokenizer, prompts: list):
    """Test model on a batch of prompts."""
    print("\n" + "=" * 80)
    print("📊 BATCH TESTING")
    print("=" * 80)

    results = []

    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}/{len(prompts)}")
        print(f"{'='*80}")
        print(f"\n📝 Prompt:\n{prompt[:200]}...")

        response = generate_response(model, tokenizer, prompt)

        # Extract response
        if "### Response:" in response:
            response_text = response.split("### Response:")[-1].strip()
        else:
            response_text = response

        print(f"\n🤖 Model Response:\n{response_text}")

        results.append({
            "prompt": prompt,
            "response": response_text,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Test trained puzzle-solving model locally")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing trained model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name (auto-detected if not specified)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to test"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch tests with default prompts"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var)"
    )

    args = parser.parse_args()

    # Get HF token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # Check GPU
    if torch.cuda.is_available():
        print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  No GPU detected. Inference will be slower on CPU.")

    # Load model
    model, tokenizer = load_model(
        Path(args.model_dir),
        base_model_name=args.base_model,
        hf_token=hf_token
    )

    if model is None:
        return

    # Run appropriate mode
    if args.interactive:
        interactive_mode(model, tokenizer)

    elif args.batch:
        results = batch_test(model, tokenizer, DEFAULT_PROMPTS)

        # Save results
        import json
        output_file = Path(args.model_dir).parent / "test_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")

    elif args.prompt:
        print("\n" + "=" * 80)
        print("🧪 SINGLE PROMPT TEST")
        print("=" * 80)
        print(f"\n📝 Prompt:\n{args.prompt}")

        response = generate_response(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()

        print(f"\n🤖 Model Response:\n{response}")

    else:
        print("\n⚠️  Please specify --prompt, --interactive, or --batch mode")
        print("   Examples:")
        print(f"     python {Path(__file__).name} --model-dir ./models/primary/final --interactive")
        print(f"     python {Path(__file__).name} --model-dir ./models/primary/final --batch")


if __name__ == "__main__":
    main()
