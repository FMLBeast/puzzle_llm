#!/usr/bin/env python3
"""
Script to fetch and setup puzzle data from the cryptopuzzles repository.
"""

import os
import sys
import subprocess
import json
from pathlib import Path


def run_command(cmd, description=None):
    """Run a shell command and print output."""
    if description:
        print(f"📝 {description}...")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e.stderr}")
        return False


def clone_cryptopuzzles_repo(target_dir="cryptopuzzles_data"):
    """Clone the cryptopuzzles repository."""
    print("=" * 60)
    print("📦 FETCHING CRYPTOPUZZLES DATA")
    print("=" * 60)

    target_path = Path(target_dir)

    # Check if already cloned
    if target_path.exists():
        print(f"✅ Directory {target_dir} already exists")
        response = input("Do you want to pull latest changes? (y/n): ")
        if response.lower() == 'y':
            os.chdir(target_dir)
            run_command("git pull", "Pulling latest changes")
            os.chdir("..")
        return True

    # Clone the repository
    repo_url = input("Enter the cryptopuzzles repository URL (or path): ").strip()

    if not repo_url:
        print("Using default: https://github.com/FMLBeast/cryptopuzzles.git")
        repo_url = "https://github.com/FMLBeast/cryptopuzzles.git"

    success = run_command(
        f"git clone {repo_url} {target_dir}",
        f"Cloning {repo_url}"
    )

    if success:
        print(f"✅ Successfully cloned to {target_dir}")
        return True
    else:
        print("❌ Failed to clone repository")
        return False


def analyze_data_structure(data_dir="cryptopuzzles_data"):
    """Analyze the structure of the cloned data."""
    print("\n" + "=" * 60)
    print("🔍 ANALYZING DATA STRUCTURE")
    print("=" * 60)

    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"❌ Directory {data_dir} not found")
        return

    # Find all data files
    json_files = list(data_path.rglob("*.json"))
    csv_files = list(data_path.rglob("*.csv"))
    txt_files = list(data_path.rglob("*.txt"))

    print(f"\n📊 Data Summary:")
    print(f"  - JSON files: {len(json_files)}")
    print(f"  - CSV files: {len(csv_files)}")
    print(f"  - TXT files: {len(txt_files)}")

    # Show sample structure
    if json_files:
        print(f"\n📄 Sample JSON file: {json_files[0]}")
        try:
            with open(json_files[0], 'r') as f:
                sample = json.load(f)
                if isinstance(sample, list) and len(sample) > 0:
                    sample = sample[0]
                print(f"  Structure: {json.dumps(sample, indent=2)[:500]}")
        except Exception as e:
            print(f"  Could not read: {e}")

    return {
        'json_files': json_files,
        'csv_files': csv_files,
        'txt_files': txt_files,
    }


def process_data(source_dir="cryptopuzzles_data"):
    """Process the raw data into training format."""
    print("\n" + "=" * 60)
    print("⚙️  PROCESSING DATA")
    print("=" * 60)

    success = run_command(
        f"python data_preprocessing.py {source_dir}",
        "Running data preprocessing"
    )

    if success:
        print("✅ Data processing complete")
        print("📁 Processed data saved to: processed_data/")
        return True
    else:
        print("❌ Data processing failed")
        return False


def verify_setup():
    """Verify Modal setup."""
    print("\n" + "=" * 60)
    print("🔧 VERIFYING MODAL SETUP")
    print("=" * 60)

    # Check if Modal is installed
    try:
        import modal
        print(f"✅ Modal installed (version: {modal.__version__})")
    except ImportError:
        print("❌ Modal not installed. Run: pip install modal")
        return False

    # Check Modal authentication
    result = subprocess.run(
        "modal token get",
        shell=True,
        capture_output=True,
        text=True
    )

    if "No token found" in result.stderr or result.returncode != 0:
        print("⚠️  Modal token not configured")
        print("Run one of:")
        print("  1. python3 -m modal setup")
        print("  2. modal token set --token-id <id> --token-secret <secret>")
        return False
    else:
        print("✅ Modal token configured")

    return True


def show_next_steps():
    """Show next steps to the user."""
    print("\n" + "=" * 60)
    print("🚀 NEXT STEPS")
    print("=" * 60)

    print("""
1. Upload data to Modal:
   modal run train_llm_modal.py --command upload --data-path processed_data

2. Start training:
   modal run train_llm_modal.py --command train

3. Test the trained model:
   modal run train_llm_modal.py --command test

4. Optional - Customize configuration:
   Edit config.yaml to adjust model, GPU, or training parameters

For more details, see README.md
""")


def main():
    """Main setup flow."""
    print("\n" + "🎯 " + "=" * 56)
    print("🎯 PUZZLE LLM TRAINING - DATA SETUP")
    print("🎯 " + "=" * 56 + "\n")

    # Step 1: Verify Modal setup
    if not verify_setup():
        print("\n❌ Please set up Modal first, then run this script again")
        sys.exit(1)

    # Step 2: Ask for data source
    print("\nWhere is your puzzle data?")
    print("1. Clone from git repository")
    print("2. Use local directory")
    print("3. Create sample data for testing")

    choice = input("\nEnter choice (1/2/3): ").strip()

    data_dir = None

    if choice == "1":
        if clone_cryptopuzzles_repo():
            data_dir = "cryptopuzzles_data"
    elif choice == "2":
        data_dir = input("Enter path to data directory: ").strip()
        if not Path(data_dir).exists():
            print(f"❌ Directory {data_dir} not found")
            sys.exit(1)
    elif choice == "3":
        print("Creating sample dataset...")
        run_command("python data_preprocessing.py", "Creating sample data")
        print("✅ Sample data created at: sample_data/")
        data_dir = "sample_data"
    else:
        print("Invalid choice")
        sys.exit(1)

    # Step 3: Analyze data structure
    if data_dir and Path(data_dir).exists():
        analyze_data_structure(data_dir)

        # Step 4: Process data
        process = input("\nProcess this data for training? (y/n): ").strip().lower()
        if process == 'y':
            if process_data(data_dir):
                show_next_steps()
            else:
                print("\n❌ Setup incomplete")
                sys.exit(1)
        else:
            print("\n⚠️  Data not processed. Run manually:")
            print(f"   python data_preprocessing.py {data_dir}")
    else:
        print("\n❌ No data directory found")
        sys.exit(1)


if __name__ == "__main__":
    main()
