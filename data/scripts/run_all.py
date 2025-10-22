"""
Main runner script to execute all three data preparation steps sequentially.

Usage:
    python run_all.py
    
Or with custom config overrides:
    python run_all.py data.min_th_icd=10 data.bin_type=quantile
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print(f"\n{'=' * 80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('=' * 80)
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with return code {result.returncode}")
        sys.exit(result.returncode)
    
    print(f"\n✓ {description} completed successfully")


def main():
    """Run all three data preparation steps."""
    script_dir = Path(__file__).parent
    
    # Get any command line arguments to pass to hydra
    args = sys.argv[1:]
    args_str = ' '.join(args) if args else ''
    
    print("=" * 80)
    print("MIMIC-III Data Preparation Pipeline")
    print("=" * 80)
    if args:
        print(f"\nConfig overrides: {args_str}")
    
    # Step 1: Read data from MIMIC-III benchmarks
    run_command(
        [sys.executable, "step1_read_data.py"] + args,
        "STEP 1: Reading data from MIMIC-III benchmarks"
    )
    
    # Step 2: Create token dictionary
    run_command(
        [sys.executable, "step2_create_tokens.py"] + args,
        "STEP 2: Creating token dictionary"
    )
    
    # Step 3: Discretize data
    run_command(
        [sys.executable, "step3_discretize.py"] + args,
        "STEP 3: Discretizing data"
    )
    
    print("\n" + "=" * 80)
    print("✓ ALL STEPS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - {split}Dataset.pkl (raw data)")
    print("  - metadata2.pkl or metadata.pkl (token dictionary)")
    print("  - {split}DiscDataset.pkl (discretized data)")
    print("\nwhere {split} = train, val, test")


if __name__ == "__main__":
    main()
