#!/usr/bin/env python3
"""
Quick comparison of primary vs benchmark training configurations.
Run this to see the differences between the two models.
"""

# Primary model config (from train_llm_modal.py)
PRIMARY_CONFIG = {
    "name": "Primary Model",
    "model_name": "mistralai/Mistral-7B-v0.1",
    "max_seq_length": 512,
    "num_train_epochs": 5,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 100,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "output_dir": "/models/puzzle-llm",
}

# Benchmark model config (from train_llm_modal_benchmark.py)
BENCHMARK_CONFIG = {
    "name": "Benchmark Model",
    "model_name": "meta-llama/Llama-3.2-3B-Instruct",
    "max_seq_length": 512,
    "num_train_epochs": 8,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "learning_rate": 3e-4,
    "warmup_steps": 150,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.1,
    "output_dir": "/models/puzzle-llm-benchmark",
}


def main():
    print("=" * 80)
    print("PUZZLE LLM TRAINING CONFIGURATIONS COMPARISON")
    print("=" * 80)
    print()

    # Calculate effective batch size
    primary_eff_batch = PRIMARY_CONFIG["per_device_train_batch_size"] * PRIMARY_CONFIG["gradient_accumulation_steps"]
    benchmark_eff_batch = BENCHMARK_CONFIG["per_device_train_batch_size"] * BENCHMARK_CONFIG["gradient_accumulation_steps"]

    configs = [PRIMARY_CONFIG, BENCHMARK_CONFIG]

    # Print table header
    print(f"{'Parameter':<35} {'Primary':<30} {'Benchmark':<30}")
    print("-" * 95)

    # Compare each parameter
    params = [
        "model_name",
        "num_train_epochs",
        "learning_rate",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "warmup_steps",
    ]

    for param in params:
        primary_val = PRIMARY_CONFIG[param]
        benchmark_val = BENCHMARK_CONFIG[param]

        # Format values
        if isinstance(primary_val, float):
            primary_str = f"{primary_val:.1e}"
            benchmark_str = f"{benchmark_val:.1e}"
        else:
            primary_str = str(primary_val)
            benchmark_str = str(benchmark_val)

        # Highlight differences
        if primary_val != benchmark_val:
            marker = " ⚡"
        else:
            marker = ""

        print(f"{param:<35} {primary_str:<30} {benchmark_str:<30}{marker}")

    # Additional computed metrics
    print("-" * 95)
    print(f"{'Effective Batch Size':<35} {primary_eff_batch:<30} {benchmark_eff_batch:<30}")
    print()

    # Key differences summary
    print("=" * 80)
    print("KEY DIFFERENCES:")
    print("=" * 80)
    print()
    print("✅ BENCHMARK model has:")
    print(f"   - Different architecture: Llama 3.2 (3B) vs Mistral (7B)")
    print(f"   - More training epochs: {BENCHMARK_CONFIG['num_train_epochs']} vs {PRIMARY_CONFIG['num_train_epochs']}")
    print(f"   - Higher learning rate: {BENCHMARK_CONFIG['learning_rate']:.1e} vs {PRIMARY_CONFIG['learning_rate']:.1e}")
    print(f"   - Stronger LoRA adaptation: r={BENCHMARK_CONFIG['lora_r']} vs r={PRIMARY_CONFIG['lora_r']}")
    print(f"   - Higher dropout: {BENCHMARK_CONFIG['lora_dropout']} vs {PRIMARY_CONFIG['lora_dropout']}")
    print(f"   - Same effective batch size: {benchmark_eff_batch}")
    print()

    # Training cost estimate
    print("=" * 80)
    print("ESTIMATED TRAINING COSTS (A100 @ $3.50/hr):")
    print("=" * 80)
    print()
    primary_hours = 2.5
    benchmark_hours = 3.5
    print(f"Primary Model:   ~{primary_hours} hours × $3.50 = ~${primary_hours * 3.50:.2f}")
    print(f"Benchmark Model: ~{benchmark_hours} hours × $3.50 = ~${benchmark_hours * 3.50:.2f}")
    print(f"Total:           ~${(primary_hours + benchmark_hours) * 3.50:.2f}")
    print()

    print("=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print()
    print("1. Process training data:")
    print("   modal run train_llm_modal.py::main --command process")
    print()
    print("2. Train primary model:")
    print("   modal run train_llm_modal.py::main --command train")
    print()
    print("3. Train benchmark model:")
    print("   modal run train_llm_modal_benchmark.py::main --command train")
    print()
    print("4. Compare results on test puzzles!")
    print()


if __name__ == "__main__":
    main()
