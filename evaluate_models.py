#!/usr/bin/env python3
"""
Model comparison and evaluation script.
Compare primary vs benchmark models on a test set.

Usage:
    python evaluate_models.py \
        --primary-model ./models/primary/final \
        --benchmark-model ./models/benchmark/final \
        --test-data ./training_data/test.json
"""

import argparse
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import time
from datetime import datetime


def load_model(model_dir: Path, model_name: str, hf_token: str = None):
    """Load a trained model."""
    print(f"  Loading {model_name}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), token=hf_token)

    # Get base model name
    import json as json_lib
    config_file = model_dir / "adapter_config.json"
    with open(config_file, 'r') as f:
        config = json_lib.load(f)
        base_model_name = config.get('base_model_name_or_path')

    # Load model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        token=hf_token,
    )

    model = PeftModel.from_pretrained(base_model, str(model_dir))
    model.eval()

    print(f"    ✅ {model_name} ready")

    return model, tokenizer, base_model_name


def generate_response(model, tokenizer, prompt: str, max_tokens: int = 256) -> tuple:
    """Generate response and measure inference time."""
    inputs = tokenizer(prompt, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = inputs.to(model.device)

    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    inference_time = time.time() - start_time

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    return response, inference_time


def evaluate_on_test_set(
    primary_model,
    primary_tokenizer,
    benchmark_model,
    benchmark_tokenizer,
    test_data: list,
    max_samples: int = None,
):
    """Evaluate both models on test set."""

    if max_samples:
        test_data = test_data[:max_samples]

    results = []

    print(f"\n📊 Evaluating on {len(test_data)} test examples...")

    for i, example in enumerate(tqdm(test_data, desc="Testing")):
        # Extract prompt (everything before ### Response:)
        text = example.get('text', '')
        if '### Response:' in text:
            prompt = text.split('### Response:')[0] + '### Response:\n'
            expected = text.split('### Response:')[-1].strip()
        else:
            prompt = text
            expected = ""

        # Get predictions from both models
        primary_response, primary_time = generate_response(
            primary_model, primary_tokenizer, prompt
        )

        benchmark_response, benchmark_time = generate_response(
            benchmark_model, benchmark_tokenizer, prompt
        )

        result = {
            "index": i,
            "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "expected": expected[:200] + "..." if len(expected) > 200 else expected,
            "primary_response": primary_response,
            "primary_time": primary_time,
            "benchmark_response": benchmark_response,
            "benchmark_time": benchmark_time,
            "source": example.get('source', 'unknown'),
            "difficulty": example.get('difficulty', 'unknown'),
        }

        results.append(result)

    return results


def calculate_metrics(results: list) -> dict:
    """Calculate comparison metrics."""

    primary_times = [r['primary_time'] for r in results]
    benchmark_times = [r['benchmark_time'] for r in results]

    primary_avg_len = sum(len(r['primary_response']) for r in results) / len(results)
    benchmark_avg_len = sum(len(r['benchmark_response']) for r in results) / len(results)

    metrics = {
        "total_tests": len(results),
        "primary": {
            "avg_inference_time": sum(primary_times) / len(primary_times),
            "min_inference_time": min(primary_times),
            "max_inference_time": max(primary_times),
            "avg_response_length": primary_avg_len,
        },
        "benchmark": {
            "avg_inference_time": sum(benchmark_times) / len(benchmark_times),
            "min_inference_time": min(benchmark_times),
            "max_inference_time": max(benchmark_times),
            "avg_response_length": benchmark_avg_len,
        }
    }

    # Speed comparison
    metrics["speed_comparison"] = {
        "faster_model": "primary" if metrics["primary"]["avg_inference_time"] < metrics["benchmark"]["avg_inference_time"] else "benchmark",
        "speedup": max(metrics["primary"]["avg_inference_time"], metrics["benchmark"]["avg_inference_time"]) /
                  min(metrics["primary"]["avg_inference_time"], metrics["benchmark"]["avg_inference_time"])
    }

    return metrics


def print_results(results: list, metrics: dict, primary_name: str, benchmark_name: str):
    """Print evaluation results."""

    print("\n" + "=" * 100)
    print("📊 EVALUATION RESULTS")
    print("=" * 100)

    print(f"\n🔵 Primary Model: {primary_name}")
    print(f"🟢 Benchmark Model: {benchmark_name}")
    print(f"\n📈 Metrics Summary:")
    print("-" * 100)

    # Primary metrics
    print(f"\n🔵 PRIMARY MODEL:")
    print(f"   Avg inference time: {metrics['primary']['avg_inference_time']:.3f}s")
    print(f"   Min/Max time: {metrics['primary']['min_inference_time']:.3f}s / {metrics['primary']['max_inference_time']:.3f}s")
    print(f"   Avg response length: {metrics['primary']['avg_response_length']:.0f} chars")

    # Benchmark metrics
    print(f"\n🟢 BENCHMARK MODEL:")
    print(f"   Avg inference time: {metrics['benchmark']['avg_inference_time']:.3f}s")
    print(f"   Min/Max time: {metrics['benchmark']['min_inference_time']:.3f}s / {metrics['benchmark']['max_inference_time']:.3f}s")
    print(f"   Avg response length: {metrics['benchmark']['avg_response_length']:.0f} chars")

    # Speed comparison
    print(f"\n⚡ SPEED COMPARISON:")
    faster = metrics['speed_comparison']['faster_model']
    speedup = metrics['speed_comparison']['speedup']
    print(f"   {faster.upper()} is {speedup:.2f}x faster")

    # Sample comparisons
    print("\n" + "=" * 100)
    print("📝 SAMPLE COMPARISONS (First 3 examples)")
    print("=" * 100)

    for i, result in enumerate(results[:3]):
        print(f"\n{'='*100}")
        print(f"Example {i+1} [{result['source']} / {result['difficulty']}]")
        print(f"{'='*100}")
        print(f"\n📝 Prompt:\n{result['prompt']}")
        print(f"\n🎯 Expected:\n{result['expected'][:300]}...")
        print(f"\n🔵 Primary ({result['primary_time']:.2f}s):\n{result['primary_response'][:300]}...")
        print(f"\n🟢 Benchmark ({result['benchmark_time']:.2f}s):\n{result['benchmark_response'][:300]}...")


def main():
    parser = argparse.ArgumentParser(description="Compare primary and benchmark models")
    parser.add_argument(
        "--primary-model",
        type=str,
        required=True,
        help="Path to primary model directory"
    )
    parser.add_argument(
        "--benchmark-model",
        type=str,
        required=True,
        help="Path to benchmark model directory"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test data (JSON or JSONL)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of test samples (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token"
    )

    args = parser.parse_args()

    print("=" * 100)
    print("🚀 PUZZLE LLM MODEL COMPARISON")
    print("=" * 100)

    # Check GPU
    if torch.cuda.is_available():
        print(f"\n📊 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("\n⚠️  No GPU detected. Evaluation will be slower on CPU.")

    # Load test data
    print(f"\n📚 Loading test data from {args.test_data}...")
    test_file = Path(args.test_data)

    if test_file.suffix == '.jsonl':
        with open(test_file, 'r') as f:
            test_data = [json.loads(line) for line in f]
    else:
        with open(test_file, 'r') as f:
            test_data = json.load(f)

    print(f"   ✅ Loaded {len(test_data)} test examples")

    # Load models
    print(f"\n🤖 Loading models...")

    hf_token = args.hf_token or __import__('os').environ.get("HF_TOKEN")

    primary_model, primary_tokenizer, primary_base = load_model(
        Path(args.primary_model), "Primary", hf_token
    )

    benchmark_model, benchmark_tokenizer, benchmark_base = load_model(
        Path(args.benchmark_model), "Benchmark", hf_token
    )

    # Run evaluation
    results = evaluate_on_test_set(
        primary_model,
        primary_tokenizer,
        benchmark_model,
        benchmark_tokenizer,
        test_data,
        max_samples=args.max_samples,
    )

    # Calculate metrics
    metrics = calculate_metrics(results)

    # Print results
    print_results(results, metrics, primary_base, benchmark_base)

    # Save results
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "primary_model": str(args.primary_model),
            "primary_base": primary_base,
            "benchmark_model": str(args.benchmark_model),
            "benchmark_base": benchmark_base,
            "test_data": str(args.test_data),
            "num_samples": len(results),
        },
        "metrics": metrics,
        "results": results,
    }

    output_file = Path(args.output)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n💾 Full results saved to: {output_file}")

    print("\n" + "=" * 100)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 100)


if __name__ == "__main__":
    main()
