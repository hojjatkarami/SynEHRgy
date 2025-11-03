# MIMIC-III to HuggingFace Dataset Converter

This script converts MIMIC-III data into a HuggingFace dataset format with the following features:
- Patient demographics (age, gender)
- ICD-9 diagnoses
- Laboratory events with timestamps and values

## Features

The output dataset contains the following features for each patient:

- `subject_id`: Patient ID (int64)
- `hadm_ids`: List of hospital admission IDs (Sequence[int64])
- `table`: Type of record - covariates (0), labs (1), problems (2), medications (3) (Sequence[ClassLabel])
- `reced_dt`: Timestamp of the record (None for covariates) (Sequence[timestamp[us]])
- `concept_uid`: Identifier for the concept (ICD code, lab ITEMID, or covariate ID) (Sequence[int64])
- `value_float`: Numeric value (for labs and covariates, None for diagnoses) (Sequence[float32])

## Requirements

Make sure you have the required packages installed:

```bash
pip install hydra-core datasets pandas numpy tqdm
```

## Configuration

The configuration file is located at `configs/data/prepare_mimic_hf.yaml`. Key parameters:

- `mimic_path`: Path to MIMIC-III data directory
- `output_path`: Where to save the HuggingFace dataset
- `output_name`: Name of the output dataset
- `max_patients`: Limit number of patients (for testing), set to `null` for all patients
- `inpatient_only`: Whether to include only in-hospital lab events (default: `true`)
- `chunksize`: Chunk size for reading the large LABEVENTS table
- `n_workers`: Number of parallel workers for processing patients (default: 8, set to 1 for single-threaded)

## Usage

### Basic Usage

Run the script with default configuration:

```bash
cd /home/hokarami/code/SynEHRgy/data/scripts2
python prepare_mimic_hf.py
```

### Override Configuration

You can override any configuration parameter from the command line:

```bash
# Process only 100 patients for testing
python prepare_mimic_hf.py max_patients=100

# Change output path
python prepare_mimic_hf.py output_path=/path/to/output

# Change output name
python prepare_mimic_hf.py output_name=mimic3_test

# Include outpatient lab events
python prepare_mimic_hf.py inpatient_only=false

# Use more workers for faster processing
python prepare_mimic_hf.py n_workers=16

# Use single-threaded processing (useful for debugging)
python prepare_mimic_hf.py n_workers=1
```

### Multiple Overrides

```bash
python prepare_mimic_hf.py max_patients=1000 output_name=mimic3_small inpatient_only=true n_workers=8
```

## Output

The script produces:

1. **Parquet file**: `{output_path}/{output_name}.parquet` - Single file format
2. **HF Dataset directory**: `{output_path}/{output_name}/` - HuggingFace dataset format
3. **Statistics file**: `{output_path}/{output_name}_stats.txt` - Summary statistics

## Loading the Dataset

Once created, you can load the dataset using:

```python
from datasets import load_from_disk

# Load the dataset
dataset = load_from_disk("/path/to/output/mimic3_dataset")

# Or load from parquet
from datasets import Dataset
dataset = Dataset.from_parquet("/path/to/output/mimic3_dataset.parquet")

# Explore the data
print(dataset)
print(dataset[0])  # First patient
```

## Data Schema

### Covariate UIDs
- Age: 1000001
- Gender: 1000002 (Male=1.0, Female=0.0)

### Table Types
- 0: Covariates (age, gender)
- 1: Labs (laboratory events)
- 2: Problems (ICD-9 diagnoses)
- 3: Medications (not implemented in current version)

## Example Record

```python
{
    'subject_id': 12345,
    'hadm_ids': [100001, 100002],
    'table': [0, 0, 2, 2, 1, 1, ...],  # 0=covariate, 1=lab, 2=problem
    'reced_dt': [None, None, '2020-01-15 10:30:00', ...],
    'concept_uid': [1000002, 1000001, 41401, ...],  # gender, age, ICD codes, lab ITEMIDs
    'value_float': [1.0, 65.5, None, None, 98.6, ...],  # values for covariates and labs
}
```

## Notes

- Patients with age > 89 have their DOB shifted. The script uses a median age of 91.4 for these patients.
- ICD-9 codes are converted to integers by removing dots. If conversion fails, a hash is used.
- Only lab events with valid numeric values (VALUENUM) are included.
- Diagnoses (problems) do not have numeric values (value_float is None).
- The script can handle both compressed (.csv.gz) and uncompressed (.csv) MIMIC files.

## Troubleshooting

### Memory Issues

If you encounter memory issues with the large LABEVENTS table:

1. Increase the chunk size: `chunksize=2000000`
2. Test with fewer patients first: `max_patients=100`
3. Filter to inpatient only: `inpatient_only=true`

### File Not Found

Make sure the `mimic_path` in the config points to the correct MIMIC-III directory containing:
- PATIENTS.csv (or .csv.gz)
- ADMISSIONS.csv (or .csv.gz)
- DIAGNOSES_ICD.csv (or .csv.gz)
- LABEVENTS.csv (or .csv.gz)

## Future Enhancements

Potential improvements:
- Add medication data from PRESCRIPTIONS table
- Filter by minimum occurrence counts for ICD codes and lab items
- Add more sophisticated age calculation for multiple admissions
- Add data quality checks and validation
- Support for incremental processing
- Add train/val/test splitting
