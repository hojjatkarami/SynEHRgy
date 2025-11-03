"""
Quick test script to verify the MIMIC-III HuggingFace dataset creation.

This script runs the data preparation on a small subset of patients
to verify everything works correctly.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datasets import load_from_disk
import pandas as pd


def test_dataset_creation():
    """Test creating a small dataset."""
    print("=" * 80)
    print("MIMIC-III HuggingFace Dataset Creation Test")
    print("=" * 80)
    
    # Import the processor
    from prepare_mimic_hf import MIMICDataProcessor
    from omegaconf import OmegaConf
    
    # Create test config
    cfg = OmegaConf.create({
        'mimic_path': '/home/hokarami/data/homes/hokarami/data/mimic3/mimic-iii-clinical-database-1.4/',
        'output_path': '/home/hokarami/code/SynEHRgy/data/processed/mimic3_hf_test',
        'output_name': 'mimic3_test',
        'max_patients': 10,  # Small test
        'inpatient_only': True,
        'chunksize': 100000,
    })
    
    print(f"\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Initialize and run
    processor = MIMICDataProcessor(cfg)
    
    print("\n1. Loading tables...")
    processor.load_tables()
    
    print("\n2. Preprocessing...")
    processor.preprocess_tables()
    
    print("\n3. Creating dataset...")
    dataset = processor.create_dataset()
    
    print("\n4. Saving dataset...")
    processor.save_dataset(dataset)
    
    return dataset


def inspect_dataset(dataset_path):
    """Inspect a created dataset."""
    print("\n" + "=" * 80)
    print("Dataset Inspection")
    print("=" * 80)
    
    dataset = load_from_disk(dataset_path)
    
    print(f"\nDataset info:")
    print(f"  Number of patients: {len(dataset)}")
    print(f"  Features: {dataset.features}")
    
    if len(dataset) > 0:
        print(f"\nFirst patient example:")
        example = dataset[0]
        print(f"  Subject ID: {example['subject_id']}")
        print(f"  Hospital admissions: {example['hadm_ids']}")
        print(f"  Number of records: {len(example['table'])}")
        
        # Count by table type
        table_counts = {}
        table_names = ['covariates', 'labs', 'problems', 'medications']
        for table_idx in example['table']:
            table_name = table_names[table_idx]
            table_counts[table_name] = table_counts.get(table_name, 0) + 1
        
        print(f"  Records by type:")
        for table_name, count in table_counts.items():
            print(f"    {table_name}: {count}")
        
        # Show first few records
        print(f"\n  First 5 records:")
        for i in range(min(5, len(example['table']))):
            table_name = table_names[example['table'][i]]
            reced_dt = example['reced_dt'][i]
            concept_uid = example['concept_uid'][i]
            value_float = example['value_float'][i]
            print(f"    [{i}] {table_name}: concept={concept_uid}, value={value_float}, time={reced_dt}")
    
    return dataset


def main():
    """Run the test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MIMIC-III HF dataset creation")
    parser.add_argument('--create', action='store_true', help='Create test dataset')
    parser.add_argument('--inspect', type=str, help='Inspect existing dataset at path')
    args = parser.parse_args()
    
    if args.create:
        dataset = test_dataset_creation()
        print("\n✓ Dataset created successfully!")
        print("\nTo inspect the dataset, run:")
        print("  python test_prepare_mimic_hf.py --inspect /home/hokarami/code/SynEHRgy/data/processed/mimic3_hf_test/mimic3_test")
    
    elif args.inspect:
        dataset = inspect_dataset(args.inspect)
        print("\n✓ Dataset inspection complete!")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
