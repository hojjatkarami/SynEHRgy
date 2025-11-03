"""
Script to create vocabulary from MIMIC-III HuggingFace dataset.

This script:
1. Reads the HuggingFace dataset created by prepare_mimic_hf.py
2. Creates a vocabulary dataframe with concept_uid, concept_name, and counts
3. Creates a quantile dictionary for continuous variables (those with value_float)
4. Saves both outputs in pickle format
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VocabCreator:
    """Creates vocabulary and quantile information from HuggingFace dataset."""
    
    # Covariate UIDs from prepare_mimic_hf.py
    COVARIATE_UIDS = {
        'age': 1000001,
        'gender': 1000002,
    }
    
    # Table type mapping
    TABLE_TYPES = {
        0: 'covariates',
        1: 'labs',
        2: 'problems',
    }
    
    def __init__(self, cfg: DictConfig):
        """
        Initialize the vocabulary creator.
        
        Args:
            cfg: Hydra configuration object
        """
        self.cfg = cfg
        self.mimic_path = Path(cfg.mimic_path)
        self.dataset_path = Path(cfg.output_path) / cfg.output_name
        self.output_path = Path(cfg.output_path)
        
        # Check if dataset exists
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset not found at {self.dataset_path}. "
                           "Please run prepare_mimic_hf.py first.")
        
        # Data containers
        self.dataset = None
        self.d_icd_diagnoses = None
        self.d_items = None
        
        # Vocabulary data
        self.concept_counts = Counter()
        self.concept_values = defaultdict(list)  # For storing values for quantile calculation
        self.concept_types = {}  # Map concept_uid to table type
        
    def load_dataset(self):
        """Load the HuggingFace dataset."""
        logger.info(f"Loading dataset from {self.dataset_path}...")
        self.dataset = load_from_disk(str(self.dataset_path))
        logger.info(f"Loaded dataset with {len(self.dataset)} patients")
        
    def load_definition_tables(self):
        """Load MIMIC-III definition tables for concept names."""
        logger.info("Loading MIMIC-III definition tables...")
        
        # Load D_ICD_DIAGNOSES
        logger.info("Loading D_ICD_DIAGNOSES.csv...")
        d_icd_file = self.mimic_path / "D_ICD_DIAGNOSES.csv"
        if not d_icd_file.exists():
            d_icd_file = self.mimic_path / "D_ICD_DIAGNOSES.csv.gz"
        self.d_icd_diagnoses = pd.read_csv(d_icd_file)
        
        # Create lookup dictionary: ICD9_CODE -> (SHORT_TITLE, LONG_TITLE)
        # Convert ICD9_CODE to integer (removing decimal)
        self.icd_lookup = {}
        for _, row in self.d_icd_diagnoses.iterrows():
            try:
                icd_code = row['ICD9_CODE']
                # Try to convert to integer (same as in prepare_mimic_hf.py)
                try:
                    icd_uid = int(str(icd_code).replace('.', ''))
                except (ValueError, AttributeError):
                    icd_uid = hash(str(icd_code)) % (10 ** 8)
                
                self.icd_lookup[icd_uid] = {
                    'short_title': row['SHORT_TITLE'],
                    'long_title': row['LONG_TITLE']
                }
            except Exception as e:
                logger.warning(f"Error processing ICD code {row.get('ICD9_CODE')}: {e}")
        
        logger.info(f"Loaded {len(self.icd_lookup)} ICD diagnoses")
        
        # Load D_LABITEMS
        logger.info("Loading D_LABITEMS.csv...")
        d_labitems_file = self.mimic_path / "D_LABITEMS.csv"
        if not d_labitems_file.exists():
            d_labitems_file = self.mimic_path / "D_LABITEMS.csv.gz"
        self.d_labitems = pd.read_csv(d_labitems_file)

        # Create lookup dictionary: ITEMID -> LABEL (concept_name)
        self.labitems_lookup = {}
        for _, row in self.d_labitems.iterrows():
            self.labitems_lookup[int(row['ITEMID'])] = {
                'label': row['LABEL'],
                'fluid': row.get('FLUID', ''),
                'category': row.get('CATEGORY', ''),
                'loinc_code': row.get('LOINC_CODE', '')
            }

        logger.info(f"Loaded {len(self.labitems_lookup)} lab items from D_LABITEMS")
        
    def process_dataset(self):
        """Process the dataset to extract concept counts and values."""
        logger.info("Processing dataset to extract vocabulary...")
        
        for patient in tqdm(self.dataset, desc="Processing patients"):
            concept_uids = patient['concepts']
            value_floats = patient['values']
            table_types = patient['types']
            
            # Process each concept in the patient's record
            for concept_uid, value_float, table_type in zip(concept_uids, value_floats, table_types):
                # Count the concept
                self.concept_counts[concept_uid] += 1
                
                # Store the table type (keep first occurrence)
                if concept_uid not in self.concept_types:
                    self.concept_types[concept_uid] = table_type
                
                # Store value if it's not None (for quantile calculation)
                if value_float is not None and not np.isnan(value_float):
                    self.concept_values[concept_uid].append(value_float)
        
        logger.info(f"Found {len(self.concept_counts)} unique concepts")
        logger.info(f"Found {len(self.concept_values)} concepts with numeric values")
        
    def get_concept_name(self, concept_uid: int, table_type: int = None) -> str:
        """
        Get the human-readable name for a concept UID.
        
        Args:
            concept_uid: The concept UID
            table_type: Optional table type (0=covariates, 1=labs, 2=problems)
            
        Returns:
            Human-readable concept name
        """
        # Check if it's a covariate
        if concept_uid == self.COVARIATE_UIDS['age']:
            return "Age"
        elif concept_uid == self.COVARIATE_UIDS['gender']:
            return "Gender"

        # Check if it's a lab item (from D_LABITEMS)
        if concept_uid in self.labitems_lookup:
            labitem_info = self.labitems_lookup[concept_uid]
            return labitem_info['label']
        
        # Check if it's an ICD diagnosis
        if concept_uid in self.icd_lookup:
            icd_info = self.icd_lookup[concept_uid]
            return icd_info['short_title']
        
        # If not found, return a generic name
        return f"Unknown_Concept_{concept_uid}"
    
    def create_vocab_dataframe(self) -> pd.DataFrame:
        """
        Create vocabulary dataframe with concept_uid, concept_name, and counts.
        
        Returns:
            DataFrame with vocabulary information
        """
        logger.info("Creating vocabulary dataframe...")
        
        vocab_data = []
        for concept_uid, count in tqdm(self.concept_counts.items(), desc="Creating vocab"):
            concept_name = self.get_concept_name(concept_uid)
            table_type = self.concept_types.get(concept_uid, -1)
            table_type_name = self.TABLE_TYPES.get(table_type, 'unknown')
            
            vocab_data.append({
                'concept_uid': concept_uid,
                'concept_name': concept_name,
                'type': table_type_name,
                'counts': count,
                'has_values': concept_uid in self.concept_values,
                'n_values': len(self.concept_values.get(concept_uid, []))
            })
        
        vocab_df = pd.DataFrame(vocab_data)
        
        # Sort by counts (descending)
        vocab_df = vocab_df.sort_values('counts', ascending=False).reset_index(drop=True)
        
        logger.info(f"Created vocabulary with {len(vocab_df)} concepts")
        logger.info(f"Concepts with numeric values: {vocab_df['has_values'].sum()}")
        
        return vocab_df
    
    def create_quantile_dict(self, n_bins: int = 10) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Create quantile dictionary for concepts with numeric values.
        
        Args:
            n_bins: Number of bins for quantile calculation
            
        Returns:
            Dictionary mapping concept_uid to {'uniform': ..., 'quantile': ...}
        """
        logger.info(f"Creating quantile dictionary with {n_bins} bins...")
        
        quantile_dict = {}
        
        for concept_uid, values in tqdm(self.concept_values.items(), desc="Computing quantiles"):
            if len(values) < n_bins:
                logger.warning(f"Concept {concept_uid} has only {len(values)} values, "
                             f"skipping quantile calculation")
                continue
            
            values_array = np.array(values)
            
            # Remove NaN values
            values_array = values_array[~np.isnan(values_array)]
            
            if len(values_array) < n_bins:
                continue
            
            try:
                # Uniform bins
                uniform_bins = np.linspace(
                    values_array.min(),
                    values_array.max(),
                    n_bins + 1
                )
                
                # Quantile bins
                quantile_percentiles = np.linspace(0, 100, n_bins + 1)
                quantile_bins = np.percentile(values_array, quantile_percentiles)
                
                # Ensure unique bin edges
                uniform_bins = np.unique(uniform_bins)
                quantile_bins = np.unique(quantile_bins)
                
                quantile_dict[concept_uid] = {
                    'uniform': uniform_bins,
                    'quantile': quantile_bins,
                    'n_values': len(values_array),
                    'min': float(values_array.min()),
                    'max': float(values_array.max()),
                    'mean': float(values_array.mean()),
                    'std': float(values_array.std()),
                    'median': float(np.median(values_array))
                }
            except Exception as e:
                logger.warning(f"Error computing quantiles for concept {concept_uid}: {e}")
        
        logger.info(f"Created quantile dictionary for {len(quantile_dict)} concepts")
        
        return quantile_dict
    
    def save_outputs(self, vocab_df: pd.DataFrame, quantile_dict: Dict):
        """
        Save vocabulary dataframe and quantile dictionary.
        
        Args:
            vocab_df: Vocabulary dataframe
            quantile_dict: Quantile dictionary
        """
        logger.info("Saving outputs...")
        
        # Save vocabulary dataframe as CSV
        vocab_csv_path = self.output_path / f"{self.cfg.output_name}_vocab.csv"
        vocab_df.to_csv(vocab_csv_path, index=False)
        logger.info(f"Saved vocabulary CSV to {vocab_csv_path}")
        
        # Save vocabulary dataframe as pickle
        vocab_pkl_path = self.output_path / f"{self.cfg.output_name}_vocab.pkl"
        with open(vocab_pkl_path, 'wb') as f:
            pickle.dump(vocab_df, f)
        logger.info(f"Saved vocabulary pickle to {vocab_pkl_path}")
        
        # Save quantile dictionary as pickle
        quantile_pkl_path = self.output_path / f"{self.cfg.output_name}_quantiles.pkl"
        with open(quantile_pkl_path, 'wb') as f:
            pickle.dump(quantile_dict, f)
        logger.info(f"Saved quantile dictionary to {quantile_pkl_path}")
        
        # Print summary statistics
        logger.info("\n" + "=" * 80)
        logger.info("VOCABULARY SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total unique concepts: {len(vocab_df)}")
        logger.info(f"Concepts with numeric values: {vocab_df['has_values'].sum()}")
        logger.info(f"Concepts in quantile dictionary: {len(quantile_dict)}")
        logger.info(f"\nConcepts by type:")
        logger.info(vocab_df['type'].value_counts().to_string())
        logger.info(f"\nTop 10 most frequent concepts:")
        logger.info("\n" + vocab_df.head(10)[['concept_uid', 'concept_name', 'type', 'counts']].to_string(index=False))
        logger.info("=" * 80)


@hydra.main(version_base=None, config_path="../../configs/data", config_name="prepare_mimic_hf")
def main(cfg: DictConfig):
    """Main function to create vocabulary from HuggingFace dataset."""
    logger.info("=" * 80)
    logger.info("CREATING VOCABULARY FROM HUGGINGFACE DATASET")
    logger.info("=" * 80)
    logger.info(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Initialize vocabulary creator
    creator = VocabCreator(cfg)
    
    # Load dataset
    creator.load_dataset()
    
    # Load definition tables
    creator.load_definition_tables()
    
    # Process dataset
    creator.process_dataset()
    
    # Create vocabulary dataframe
    vocab_df = creator.create_vocab_dataframe()
    
    # Create quantile dictionary
    n_bins = cfg.get('n_bins_vocab', 10)
    quantile_dict = creator.create_quantile_dict(n_bins=n_bins)
    
    # Save outputs
    creator.save_outputs(vocab_df, quantile_dict)
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ VOCABULARY CREATION COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
