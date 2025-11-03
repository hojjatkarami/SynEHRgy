"""
Script to prepare MIMIC-III data as a Hugging Face dataset.

This script reads MIMIC-III tables (patients, admissions, diagnoses_icd, labevents)
and converts them into a Hugging Face dataset with the specified features.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import Dataset, Features, Value, Sequence, ClassLabel, load_from_disk, concatenate_datasets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MIMICDataProcessor:
    """Processes MIMIC-III data into HuggingFace dataset format."""
    
    # Define concept UIDs for covariates
    COVARIATE_UIDS = {
        'age': 1000001,
        'gender': 1000002,
    }
    
    # Table type mapping
    TABLE_TYPES = {
        'covariates': 0,
        'labs': 1,
        'problems': 2,
    }
    
    def __init__(self, cfg: DictConfig):
        """
        Initialize the processor.
        
        Args:
            cfg: Hydra configuration object
        """
        self.cfg = cfg
        self.mimic_path = Path(cfg.mimic_path)
        self.output_path = Path(cfg.output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.patients_df = None
        self.admissions_df = None
        self.diagnoses_df = None
        self.labevents_df = None
        
    def load_tables(self):
        """Load required MIMIC-III tables."""
        logger.info("Loading MIMIC-III tables...")
        
        # Load patients table
        logger.info("Loading PATIENTS.csv...")
        patients_file = self.mimic_path / "PATIENTS.csv"
        if not patients_file.exists():
            patients_file = self.mimic_path / "PATIENTS.csv.gz"
        self.patients_df = pd.read_csv(patients_file)
        logger.info(f"Loaded {len(self.patients_df)} patients")
        
        # Load admissions table
        logger.info("Loading ADMISSIONS.csv...")
        admissions_file = self.mimic_path / "ADMISSIONS.csv"
        if not admissions_file.exists():
            admissions_file = self.mimic_path / "ADMISSIONS.csv.gz"
        self.admissions_df = pd.read_csv(admissions_file)
        logger.info(f"Loaded {len(self.admissions_df)} admissions")
        
        # Load diagnoses_icd table
        logger.info("Loading DIAGNOSES_ICD.csv...")
        diagnoses_file = self.mimic_path / "DIAGNOSES_ICD.csv"
        if not diagnoses_file.exists():
            diagnoses_file = self.mimic_path / "DIAGNOSES_ICD.csv.gz"
        self.diagnoses_df = pd.read_csv(diagnoses_file)
        logger.info(f"Loaded {len(self.diagnoses_df)} diagnoses")
        
        # Load labevents table
        logger.info("Loading LABEVENTS.csv...")
        labevents_file = self.mimic_path / "LABEVENTS.csv"
        if not labevents_file.exists():
            labevents_file = self.mimic_path / "LABEVENTS.csv.gz"
        
        # Load in chunks due to large size
        chunks = []
        chunksize = self.cfg.get('chunksize', 1000000)
        for chunk in tqdm(pd.read_csv(labevents_file, chunksize=chunksize), desc="Loading LABEVENTS"):
            chunks.append(chunk)
        self.labevents_df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Loaded {len(self.labevents_df)} lab events")
        
    def preprocess_tables(self):
        """Preprocess the loaded tables."""
        logger.info("Preprocessing tables...")
        
        # Convert timestamps
        self.admissions_df['ADMITTIME'] = pd.to_datetime(self.admissions_df['ADMITTIME'])
        self.admissions_df['DISCHTIME'] = pd.to_datetime(self.admissions_df['DISCHTIME'])
        self.patients_df['DOB'] = pd.to_datetime(self.patients_df['DOB'])
        self.labevents_df['CHARTTIME'] = pd.to_datetime(self.labevents_df['CHARTTIME'])
        
        # Filter patients if required
        if self.cfg.get('max_patients'):
            logger.info(f"Limiting to {self.cfg.max_patients} patients")
            unique_subjects = self.patients_df['SUBJECT_ID'].unique()[:self.cfg.max_patients]
            self.patients_df = self.patients_df[self.patients_df['SUBJECT_ID'].isin(unique_subjects)]
            self.admissions_df = self.admissions_df[self.admissions_df['SUBJECT_ID'].isin(unique_subjects)]
            self.diagnoses_df = self.diagnoses_df[self.diagnoses_df['SUBJECT_ID'].isin(unique_subjects)]
            self.labevents_df = self.labevents_df[self.labevents_df['SUBJECT_ID'].isin(unique_subjects)]
        
        # Filter lab events to only those with HADM_ID (in-hospital)
        if self.cfg.get('inpatient_only', True):
            logger.info("Filtering to in-hospital lab events only...")
            self.labevents_df = self.labevents_df[self.labevents_df['HADM_ID'].notna()]
        
        # Remove lab events without valid values
        self.labevents_df = self.labevents_df[self.labevents_df['VALUENUM'].notna()]
        
        logger.info("Preprocessing complete")
        
    def calculate_age_at_admission(self, row):
        """Calculate patient age at admission."""
        patient = self.patients_df[self.patients_df['SUBJECT_ID'] == row['SUBJECT_ID']].iloc[0]
        dob = patient['DOB']
        admit_time = row['ADMITTIME']
        
        # Convert to Python datetime to avoid pandas overflow issues
        try:
            if pd.notna(dob) and pd.notna(admit_time):
                dob_dt = pd.Timestamp(dob).to_pydatetime()
                admit_dt = pd.Timestamp(admit_time).to_pydatetime()
                age = (admit_dt - dob_dt).days / 365.25
                
                # Handle shifted ages for patients > 89
                if age > 300:  # Shifted DOB indicator
                    age = 91.4  # Use median age for shifted patients
            else:
                age = 65.0  # Default age if missing data
        except (ValueError, OverflowError):
            # If calculation fails, assume shifted age
            age = 91.4
        
        return age
    
    def process_patient_data(self, subject_id: int) -> Optional[Dict]:
        """
        Process data for a single patient.
        
        Args:
            subject_id: Patient's SUBJECT_ID
            
        Returns:
            Dictionary with patient data in HF dataset format
        """
        # Get patient info
        patient = self.patients_df[self.patients_df['SUBJECT_ID'] == subject_id]
        if len(patient) == 0:
            return None
        patient = patient.iloc[0]
        
        # Get admissions
        admissions = self.admissions_df[self.admissions_df['SUBJECT_ID'] == subject_id].copy()
        if len(admissions) == 0:
            return None
        
        # Sort by admission time
        admissions = admissions.sort_values('ADMITTIME')
        hadm_ids = admissions['HADM_ID'].tolist()
        
        # Initialize lists for sequences
        tables = []
        reced_dts = []
        concept_uids = []
        value_floats = []
        
        # Add covariates (age and gender)
        # Gender
        gender_value = 1.0 if patient['GENDER'] == 'M' else 0.0
        tables.append(self.TABLE_TYPES['covariates'])
        reced_dts.append(None)
        concept_uids.append(self.COVARIATE_UIDS['gender'])
        value_floats.append(gender_value)
        
        # Age (calculated at first admission)
        first_admission = admissions.iloc[0]
        age = self.calculate_age_at_admission(first_admission)
        tables.append(self.TABLE_TYPES['covariates'])
        reced_dts.append(None)
        concept_uids.append(self.COVARIATE_UIDS['age'])
        value_floats.append(float(age))
        
        # Process each admission
        for _, admission in admissions.iterrows():
            hadm_id = admission['HADM_ID']
            admit_time = admission['ADMITTIME']
            
            # Add diagnoses (problems)
            diagnoses = self.diagnoses_df[self.diagnoses_df['HADM_ID'] == hadm_id]
            for _, diag in diagnoses.iterrows():
                if pd.notna(diag['ICD9_CODE']):
                    # Convert ICD9 code to integer (hash if necessary)
                    try:
                        icd_uid = int(diag['ICD9_CODE'].replace('.', ''))
                    except (ValueError, AttributeError):
                        icd_uid = hash(str(diag['ICD9_CODE'])) % (10 ** 8)
                    
                    tables.append(self.TABLE_TYPES['problems'])
                    # Normalize to admission time (diagnoses recorded at admission time = 0)
                    reced_dts.append(pd.Timestamp(0))
                    concept_uids.append(icd_uid)
                    value_floats.append(None)  # No numeric value for diagnoses
            
            # Add lab events
            labs = self.labevents_df[self.labevents_df['HADM_ID'] == hadm_id]
            for _, lab in labs.iterrows():
                if pd.notna(lab['CHARTTIME']) and pd.notna(lab['VALUENUM']):
                    tables.append(self.TABLE_TYPES['labs'])
                    # Normalize chart time relative to admission time
                    time_delta = lab['CHARTTIME'] - admit_time
                    reced_dts.append(pd.Timestamp(0) + time_delta)
                    concept_uids.append(int(lab['ITEMID']))
                    value_floats.append(float(lab['VALUENUM']))
        
        # Create patient record
        patient_record = {
            'subject_id': int(subject_id),
            'hadm_ids': [int(h) for h in hadm_ids],
            'table': tables,
            'reced_dt': reced_dts,
            'concept_uid': concept_uids,
            'value_float': value_floats,
        }
        
        return patient_record
    
    def get_features_schema(self):
        """Get the HuggingFace dataset features schema."""
        return Features({
            'subject_id': Value('int64'),
            'hadm_ids': Sequence(Value('int64')),
            'table': Sequence(
                ClassLabel(
                    num_classes=3,
                    names=['covariates', 'labs', 'problems'],
                )
            ),
            'reced_dt': Sequence(Value('timestamp[us]')),
            'concept_uid': Sequence(Value('int64')),
            'value_float': Sequence(Value('float32')),
        })
    
    def process_chunk(self, chunk_id: int, subject_ids: List[int], chunk_dir: Path, features: Features):
        """
        Process a single chunk of patients.
        
        Args:
            chunk_id: The chunk identifier
            subject_ids: List of all subject IDs
            chunk_dir: Directory to save chunks
            features: HuggingFace dataset features schema
        """
        logger.info(f"Processing chunk {chunk_id + 1}/{self.cfg.n_chunks}...")
        
        # Get subject IDs for this chunk
        chunk_subject_ids = [
            sid for sid in subject_ids 
            if int(sid) % self.cfg.n_chunks == chunk_id
        ]
        
        if len(chunk_subject_ids) == 0:
            logger.warning(f"Chunk {chunk_id} has no patients, skipping...")
            return
        
        logger.info(f"Chunk {chunk_id} has {len(chunk_subject_ids)} patients")
        
        # Process each patient in this chunk
        patient_records = []
        for subject_id in tqdm(chunk_subject_ids, desc=f"Processing chunk {chunk_id}", position=chunk_id % 10):
            record = self.process_patient_data(subject_id)
            if record is not None:
                patient_records.append(record)
        
        if len(patient_records) == 0:
            logger.warning(f"Chunk {chunk_id} has no valid records, skipping...")
            return
        
        logger.info(f"Chunk {chunk_id}: Successfully processed {len(patient_records)} patients")
        
        # Create dataset for this chunk
        chunk_dataset = Dataset.from_list(patient_records, features=features)
        
        # Save chunk
        chunk_path = chunk_dir / f"chunk_{chunk_id:04d}"
        chunk_dataset.save_to_disk(str(chunk_path))
        logger.info(f"Saved chunk {chunk_id} to {chunk_path}")
        
        # Clear memory
        del patient_records
        del chunk_dataset
    
    def create_chunk_datasets(self):
        """Create HuggingFace datasets in chunks and save them."""
        logger.info("Creating HuggingFace datasets in chunks...")
        
        # Get unique subject IDs
        subject_ids = self.patients_df['SUBJECT_ID'].unique()
        logger.info(f"Processing {len(subject_ids)} patients in {self.cfg.n_chunks} chunks...")
        
        # Create chunk directory
        chunk_dir = self.output_path / "hf_chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        # Get features schema
        features = self.get_features_schema()
        
        # Determine number of workers
        n_workers = self.cfg.get('n_workers', 1)
        if n_workers < 0:
            n_workers = cpu_count()
        
        logger.info(f"Using {n_workers} workers for parallel processing")
        
        # Determine number of chunks to process (limited if specified)
        limit = self.cfg.get('limit', None)
        n_chunks_to_process = min(limit, self.cfg.n_chunks) if limit is not None else self.cfg.n_chunks
        
        if limit is not None:
            logger.info(f"Limiting processing to {n_chunks_to_process} chunks (out of {self.cfg.n_chunks})")
        
        if n_workers == 1:
            # Sequential processing
            for chunk_id in range(n_chunks_to_process):
                self.process_chunk(chunk_id, subject_ids, chunk_dir, features)
        else:
            # Parallel processing
            chunk_ids = list(range(n_chunks_to_process))
            
            # Create partial function with fixed arguments
            process_func = partial(
                self.process_chunk,
                subject_ids=subject_ids,
                chunk_dir=chunk_dir,
                features=features
            )
            
            # Process chunks in parallel
            with Pool(processes=n_workers) as pool:
                pool.map(process_func, chunk_ids)
        
        logger.info(f"All chunks saved to {chunk_dir}")
    
    def merge_chunks(self) -> Dataset:
        """Load and merge all chunk datasets."""
        logger.info("Merging all saved chunks...")
        
        chunk_dir = self.output_path / "hf_chunks"
        
        if not chunk_dir.exists():
            raise ValueError(f"Chunk directory {chunk_dir} does not exist!")
        
        # Get all chunk directories
        chunk_dirs = sorted([
            d for d in os.listdir(chunk_dir)
            if (chunk_dir / d).is_dir() and d.startswith("chunk_")
        ])
        
        if len(chunk_dirs) == 0:
            raise ValueError(f"No chunk directories found in {chunk_dir}")
        
        logger.info(f"Found {len(chunk_dirs)} chunks to merge")
        
        # Load all chunks
        hf_datasets = []
        for chunk_name in tqdm(chunk_dirs, desc="Loading chunks"):
            chunk_path = chunk_dir / chunk_name
            dataset = load_from_disk(str(chunk_path))
            hf_datasets.append(dataset)
            logger.info(f"Loaded {chunk_name} with {len(dataset)} patients")
        
        # Concatenate all datasets
        logger.info("Concatenating datasets...")
        final_dataset = concatenate_datasets(hf_datasets)

        col_names_dict = {
            "subject_id": "id",
            "reced_dt": "dates",
            "value_float": "values",
            "concept_uid": "concepts",
            "table": "types",
        }
        final_dataset = final_dataset.rename_columns(col_names_dict)
        
        logger.info(f"Merged dataset contains {len(final_dataset)} patients")
        
        return final_dataset
    
    def save_dataset(self, dataset: Dataset):
        """Save the dataset to disk."""
        logger.info(f"Saving dataset to {self.output_path}...")
        
        # Save as parquet
        output_file = self.output_path / f"{self.cfg.output_name}.parquet"
        dataset.to_parquet(output_file)
        logger.info(f"Saved to {output_file}")
        
        # Also save as HF dataset directory
        output_dir = self.output_path / self.cfg.output_name
        dataset.save_to_disk(output_dir)
        logger.info(f"Saved to {output_dir}")
        
        # Save statistics
        stats = {
            'num_patients': len(dataset),
            'total_records': sum(len(d['types']) for d in dataset),
            'features': str(dataset.features),
            'created_at': datetime.now().isoformat(),
        }
        
        stats_file = self.output_path / f"{self.cfg.output_name}_stats.txt"
        with open(stats_file, 'w') as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        
        logger.info(f"Statistics saved to {stats_file}")
        logger.info(f"Dataset created with {stats['num_patients']} patients and {stats['total_records']} total records")


@hydra.main(version_base=None, config_path="../../configs/data", config_name="prepare_mimic_hf")
def main(cfg: DictConfig):
    """Main function to process MIMIC-III data."""
    logger.info("Starting MIMIC-III to HuggingFace dataset conversion")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Initialize processor
    processor = MIMICDataProcessor(cfg)
    
    # Load and preprocess tables
    processor.load_tables()
    processor.preprocess_tables()
    
    # Create chunk datasets
    processor.create_chunk_datasets()
    
    # Merge all chunks
    dataset = processor.merge_chunks()
    
    # Save final dataset
    processor.save_dataset(dataset)
    
    logger.info("Processing complete!")


if __name__ == "__main__":
    main()
