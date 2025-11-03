# MIMIC-III Data Preparation Scripts

This directory contains scripts to prepare MIMIC-III data for the SynEHRgy project. The notebook `prepare_mimic.ipynb` has been broken down into three modular Python scripts with Hydra configuration management.

## Table of Contents

- [Quick Start](#quick-start)
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Pipeline Details](#pipeline-details)
- [Output Files](#output-files)
- [Data Structure](#data-structure)
- [Token Dictionary](#token-dictionary)
- [Advanced Options](#advanced-options)
- [Validation and Testing](#validation-and-testing)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Quick Start

Get started in 5 steps:

### 1. Installation

```bash
cd data/scripts
pip install -r requirements.txt
```

### 2. Configuration

Edit `configs/data/prepare_mimic.yaml` and update the paths:

```yaml
# Update these paths to match your MIMIC-III installation
mimic_path: "~/data/mimic3/mimic-iii-clinical-database-1.4/"
path_root: "~/data/mimic3-benchmarks/data/root4"
path_ihm: "~/data/mimic3-benchmarks/data/in-hospital-mortality2"
path_phe: "~/data/mimic3-benchmarks/data/phenotyping2"
path_data: "data/processed/mimic3-v2"
```

### 3. Validate Setup

```bash
python validate_setup.py
```

### 4. Run Pipeline

```bash
# Run all steps
python run_all.py

# Or run individually
python step1_read_data.py
python step2_create_tokens.py
python step3_discretize.py
```

### 5. Inspect Results

```bash
python example_usage.py
```

## Overview

The data preparation pipeline consists of three modular steps that transform raw MIMIC-III data into tokenized sequences suitable for training deep learning models.

### Pipeline Steps

1. **Step 1: Read Data** (`step1_read_data.py`)
   - Reads ICD codes from MIMIC-III database
   - Filters codes based on minimum frequency threshold
   - Restructures data from mimic3-benchmarks format
   - Combines time series, codes, and covariates
   - **Outputs**: `{split}Dataset.pkl` files

2. **Step 2: Create Token Dictionary** (`step2_create_tokens.py`)
   - Analyzes time series distributions
   - Creates discretization bins (uniform or quantile)
   - Builds comprehensive token dictionary
   - Tokenizes ICD codes and time series variables
   - Adds covariate, label, and special tokens
   - **Outputs**: `metadata2.pkl` (uniform) or `metadata.pkl` (quantile)

3. **Step 3: Discretize Data** (`step3_discretize.py`)
   - Loads processed dataset and metadata
   - Converts all data to token IDs
   - Discretizes time series using token dictionary
   - Calculates prediction horizons
   - **Outputs**: `{split}DiscDataset.pkl` files

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Raw MIMIC-III Data                      │
│  (DIAGNOSES_ICD.csv, PROCEDURES_ICD.csv, benchmarks data)   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  STEP 1: step1_read_data.py                 │
│  • Filter ICD codes by frequency                            │
│  • Load phenotyping and mortality labels                    │
│  • Restructure patient data                                 │
│  • Combine time series, codes, covariates                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
              {split}Dataset.pkl files
         (trainDataset, valDataset, testDataset)
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                STEP 2: step2_create_tokens.py               │
│  • Analyze time series distributions                        │
│  • Create discretization bins                               │
│  • Build token dictionary                                   │
│  • Generate soft labels (experimental)                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
              metadata2.pkl or metadata.pkl
            (token dictionary & discretization)
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                STEP 3: step3_discretize.py                  │
│  • Load raw data and metadata                               │
│  • Convert all values to token IDs                          │
│  • Calculate prediction horizons                            │
│  • Save discretized data                                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
           {split}DiscDataset.pkl files
      (trainDiscDataset, valDiscDataset, testDiscDataset)
```

## Prerequisites

Before running the pipeline, ensure you have:

- **MIMIC-III clinical database** (version 1.4) downloaded and extracted
- **MIMIC-III benchmarks data** processed for:
  - Phenotyping task
  - In-hospital mortality task
- **Python 3.8+** installed
- **Git repository** cloned

## Installation

Install all required dependencies:

```bash
cd data/scripts
pip install -r requirements.txt
```

Dependencies include:
- `hydra-core`: Configuration management
- `omegaconf`: Configuration utilities
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `scipy`: Statistical functions
- `plotly`: Interactive visualizations
- `tqdm`: Progress bars

**Verify installation:**

```bash
python validate_setup.py
```

## Configuration

All configuration is managed through Hydra configs in `configs/data/prepare_mimic.yaml`.

### Key Configuration Sections

1. **Paths**: Location of MIMIC-III data

```yaml
mimic_path: "/path/to/mimic-iii-clinical-database-1.4/"
path_root: "/path/to/mimic3-benchmarks/data/root4"
path_ihm: "/path/to/mimic3-benchmarks/data/in-hospital-mortality2"
path_phe: "/path/to/mimic3-benchmarks/data/phenotyping2"
path_data: "data/processed/mimic3-v2"
```

2. **ICD Filtering**: Minimum frequency threshold

```yaml
min_th_icd: 5  # Keep codes appearing ≥5 times
```

3. **Variables**: Time series and covariate definitions

```yaml
variables_ts: ["Capillary refill rate", "Diastolic blood pressure", ...]
variables_covar: ["Age", "Gender"]
```

4. **Discretization**: Binning parameters

```yaml
bin_type: "uniform"  # or "quantile"
n_bins_default: 10
n_bins: {...}  # Per-variable bin counts
```

5. **Horizons**: Prediction horizons (experimental)

```yaml
horizons: [1, 2, 5, 10, 20, 50]
```

### Configuration Tips

- Use **absolute paths** to avoid navigation issues
- Set `save_plots: false` to skip visualization generation
- Adjust `min_th_icd` to control vocabulary size
- Choose `bin_type: "quantile"` for non-uniform distributions

## Usage

### Option 1: Run All Steps (Recommended)

```bash
cd data/scripts
python run_all.py
```

This runs all three steps sequentially with consistent configuration.

### Option 2: Run Individual Steps

```bash
# Step 1: Read and filter raw data
python step1_read_data.py n_workers=8

# Step 2: Create token dictionary
python step2_create_tokens.py bin_type=uniform
# or
python step2_create_tokens.py bin_type=quantile
# or 
python step2_create_tokens.py bin_type=quantile_ueq

# Step 3: Convert to token IDs
python step3_discretize.py disc_name=uniform_v1 n_workers=8



# for eicu data
python step2_create_tokens.py bin_type=uniform --config-name=prepare_eicu

python step3_discretize.py disc_name=uniform_v1 n_workers=8 --config-name=prepare_eicu


```

### Configuration Overrides

Override any parameter from the command line:

```bash
# Change ICD code threshold
python run_all.py data.min_th_icd=10

# Change bin type to quantile
python run_all.py data.bin_type=quantile

# Multiple overrides
python run_all.py data.min_th_icd=10 data.bin_type=quantile data.n_bins_default=15

# Change output directory
python run_all.py data.path_data=data/processed/mimic3-v3

# Disable plots
python run_all.py data.save_plots=false
```

### Hydra Working Directory

By default, Hydra creates output directories for each run:

```bash
# Disable output directories
python run_all.py hydra.output_subdir=null hydra.run.dir=.
```

## Project Structure

```
SynEHRgy/
├── configs/
│   └── data/
│       └── prepare_mimic.yaml      # Main configuration file
├── data/
│   ├── scripts/
│   │   ├── step1_read_data.py      # Script 1: Read raw data
│   │   ├── step2_create_tokens.py  # Script 2: Create token dictionary
│   │   ├── step3_discretize.py     # Script 3: Discretize data
│   │   ├── run_all.py              # Run all steps sequentially
│   │   ├── example_usage.py        # Example of using processed data
│   │   ├── validate_setup.py       # Validate configuration
│   │   ├── requirements.txt        # Python dependencies
│   │   ├── README.md               # This file
│   │   ├── CHECKLIST.md            # Setup checklist
│   │   └── CONVERSION_SUMMARY.md   # Conversion details
│   └── processed/
│       └── mimic3-v2/              # Output directory (created by scripts)
│           ├── trainDataset.pkl
│           ├── valDataset.pkl
│           ├── testDataset.pkl
│           ├── metadata2.pkl
│           ├── trainDiscDataset.pkl
│           ├── valDiscDataset.pkl
│           ├── testDiscDataset.pkl
│           ├── plots/
│           └── ts/
└── ...
```

## Pipeline Details

### Script 1: `step1_read_data.py`

**Purpose**: Load and restructure raw MIMIC-III data

**Key Functions**:
- `load_icd_codes()`: Filter ICD codes by frequency threshold
- `process_benchmarks_data()`: Restructure patient data from benchmarks format

**Processing Steps**:
1. Load ICD diagnosis and procedure codes from MIMIC-III
2. Filter codes based on minimum frequency (`min_th_icd`)
3. Load phenotyping and in-hospital mortality data
4. Restructure data by patient ID and admission
5. Combine time series, codes, covariates, and labels

**Output Format**:
```python
{
    'sid': subject_id,
    'hadm_id': [hadm_id1, hadm_id2, ...],
    'covariates': [[age, gender], ...],
    'codes': [[diag_codes, proc_codes], ...],
    'ts': [ts_df1, ts_df2, ...],
    'label_ihm': [0, 1, ...],
    'label_phe': [[0,1,0,...], ...]
}
```

### Script 2: `step2_create_tokens.py`

**Purpose**: Create comprehensive token dictionary

**Key Functions**:
- `create_code_tokens()`: Tokenize ICD codes
- `tokenize_timeseries()`: Discretize and tokenize time series
- `add_covariate_and_label_tokens()`: Add special tokens
- `create_soft_label_matrix()`: Create soft labels (experimental)

**Processing Steps**:
1. Analyze time series distributions from training data
2. Create discretization bins (uniform or quantile)
3. Generate tokens for each variable and bin combination
4. Create tokens for ICD codes, covariates, and labels
5. Add special tokens (`<s>`, `</s>`, `<pad>`, etc.)
6. Build token-to-ID mapping
7. Generate soft labels for continuous variables

**Token Types**:
- Code tokens: `('code', code_id)`
- Time series tokens: `('ts', var_id, bin_id)`
- Covariate tokens: `('covar', var_id, bin_id)`
- Timestamp tokens: `('timestamp', var_id, bin_id)`
- Label tokens: `('label', task, class_id)`
- Special tokens: `'<s>'`, `'</s>'`, `'<pad>'`, etc.

### Script 3: `step3_discretize.py`

**Purpose**: Convert all data to token IDs

**Key Functions**:
- `discretize_covariates()`: Convert covariates to tokens
- `discretize_codes()`: Convert ICD codes to token IDs
- `discretize_timeseries()`: Convert time series to tokens
- `calculate_horizons()`: Calculate prediction horizons

**Processing Steps**:
1. Load raw datasets and metadata
2. Discretize covariates using predefined bins
3. Map ICD codes to token IDs
4. Discretize time series values using bins from metadata
5. Convert timestamps to time gap bins
6. Calculate prediction horizons for each admission
7. Save discretized data as token ID sequences

## Output Files

After running all steps successfully:

```
data/processed/mimic3-v2/
├── trainDataset.pkl          # Raw training data
├── valDataset.pkl            # Raw validation data
├── testDataset.pkl           # Raw test data
├── metadata2.pkl             # Token dictionary (uniform binning)
│   or metadata.pkl           # Token dictionary (quantile binning)
├── trainDiscDataset.pkl      # Discretized training data
├── valDiscDataset.pkl        # Discretized validation data
├── testDiscDataset.pkl       # Discretized test data
├── plots/                    # Distribution plots
│   ├── icd_diagnosis_top10.html
│   └── icd_procedures_top10.html
└── ts/                       # Time series variable plots
    ├── Heart Rate.html
    ├── Temperature.html
    └── ...
```

## Data Structure

### Raw Dataset (`{split}Dataset.pkl`)

Dictionary containing patient records:

```python
{
    'sid': int,                    # Subject ID
    'hadm_id': List[int],          # Admission IDs
    'covariates': List[List],      # [Age, Gender] per admission
    'codes': List[List[List]],     # [diag_codes, proc_codes] per admission
    'ts': List[DataFrame],         # Time series DataFrames per admission
    'label_ihm': List[int],        # In-hospital mortality labels
    'label_phe': List[List[int]]   # Phenotyping labels (25 phenotypes)
}
```

### Discretized Dataset (`{split}DiscDataset.pkl`)

Dictionary containing tokenized patient records:

```python
{
    'covars': List[List[int]],           # Discretized covariate token IDs
    'codes': List[List[int]],            # Code token IDs
    'ts': List[Tuple],                   # (variable_ids, values, time_gaps)
    'labels_phe': List[List[int]],       # Phenotyping labels
    'labels_ihm': List[int],             # Mortality labels
    'horizons': List[List[int]]          # Prediction horizon indices
}
```

### Metadata (`metadata2.pkl` or `metadata.pkl`)

Dictionary containing token mappings and discretization info:

```python
{
    'token2id': Dict,              # Token to ID mapping
    'var2id': Dict,                # Variable name to ID mapping
    'codeToId': Dict,              # Code to ID mapping
    'idToCode': Dict,              # ID to code mapping
    'ts_info': Dict,               # Time series variable info
    'possibleValues': Dict,        # Categorical value mappings
    'isCategorical': Dict,         # Variable type indicators
    'discretization': Dict,        # Binning information (edges, centers)
    'idToLabel': Dict,             # Label ID mappings
    'vocab_size': Dict,            # Vocabulary size breakdown
    'M_soft_labels': array         # Soft label matrix (experimental)
}
```

## Token Dictionary

### Token Format

Tokens are stored as tuples or strings:

- **Code tokens**: `('code', code_id)` - ICD diagnosis/procedure codes
- **Time series tokens**: `('ts', var_id, bin_id)` - Discretized measurements
- **Covariate tokens**: `('covar', var_id, bin_id)` - Age and gender bins
- **Timestamp tokens**: `('timestamp', var_id, bin_id)` - Time gap bins
- **Label tokens**: `('label', task, class_id)` - Phenotype/mortality labels
- **Special tokens**: `'<s>'`, `'</s>'`, `'<pad>'`, `'<unk>'`, etc.

### Vocabulary Size

The vocabulary includes:
- Code tokens: ~1000-5000 (depends on `min_th_icd`)
- Time series tokens: ~100-200 (depends on variables and bins)
- Covariate tokens: ~20-30
- Timestamp tokens: ~10
- Label tokens: ~30
- Special tokens: ~10

Total vocabulary size: ~2000-6000 tokens

## Advanced Options

### Custom Configuration

Create a new config file in `configs/data/`:

```yaml
# configs/data/my_config.yaml
defaults:
  - prepare_mimic

data:
  min_th_icd: 10
  bin_type: quantile
  n_bins_default: 20
```

Run with custom config:

```bash
python run_all.py --config-name=my_config
```

### Multiple Experiments

Run with different parameters:

```bash
# Experiment 1: Low threshold, uniform binning
python run_all.py data.min_th_icd=5 data.bin_type=uniform data.path_data=data/processed/exp1

# Experiment 2: High threshold, quantile binning
python run_all.py data.min_th_icd=20 data.bin_type=quantile data.path_data=data/processed/exp2
```

### Performance Optimization

For faster processing:

```bash
# Disable plot generation
python run_all.py data.save_plots=false

# Disable Hydra output directories
python run_all.py hydra.output_subdir=null hydra.run.dir=.
```

## Validation and Testing

### Validate Setup

Before running the pipeline:

```bash
python validate_setup.py
```

This checks:
- Configuration file exists and is valid
- Required paths are accessible
- Dependencies are installed
- MIMIC-III data files exist

### Inspect Processed Data

After running the pipeline:

```bash
python example_usage.py
```

This displays:
- Dataset statistics (patient counts, admission counts)
- Vocabulary size
- Token distribution
- Sample patient records

### Verification Checklist

After completion, verify:

- [ ] Output directory exists: `data/processed/mimic3-v2/`
- [ ] All `.pkl` files created (7 files total)
- [ ] Plots generated in `plots/` and `ts/` (if enabled)
- [ ] No error messages in console output
- [ ] `example_usage.py` runs without errors
- [ ] Vocabulary size is reasonable (2000-6000 tokens)

## Troubleshooting

### Import Errors

**Problem**: Missing dependencies

**Solution**:
```bash
pip install -r requirements.txt
```

Verify Python version:
```bash
python --version  # Should be 3.8+
```

### Path Errors

**Problem**: Cannot find MIMIC-III files

**Solution**:
1. Check paths in `configs/data/prepare_mimic.yaml`
2. Use absolute paths instead of relative paths
3. Verify file permissions:
```bash
ls -la /path/to/mimic-iii-clinical-database-1.4/
```

**Common path issues**:
- MIMIC-III database: Look for `DIAGNOSES_ICD.csv`, `PROCEDURES_ICD.csv`
- Benchmarks data: Look for `listfile.csv` and time series CSVs

### Memory Errors

**Problem**: Out of memory during processing

**Solution**:
1. Close other applications
2. Process on a machine with more RAM (16GB+ recommended)
3. Reduce data size by increasing `min_th_icd` threshold
4. Process splits separately instead of using `run_all.py`

### Data Format Errors

**Problem**: Unexpected data format or missing columns

**Solution**:
1. Verify MIMIC-III benchmarks were processed correctly
2. Check that CSV files have expected columns
3. Use the exact versions specified in references
4. Regenerate benchmarks data if necessary

### Configuration Errors

**Problem**: Hydra configuration errors

**Solution**:
1. Validate YAML syntax in config file
2. Check for typos in parameter names
3. Use `--cfg job` to inspect resolved configuration:
```bash
python run_all.py --cfg job
```

### File Not Found Errors

**Problem**: Cannot find intermediate files

**Solution**:
1. Run steps in order (1 → 2 → 3)
2. Check `path_data` in configuration
3. Verify previous step completed successfully
4. Check for error messages in previous outputs

### Plotting Errors

**Problem**: Plotly errors or missing plots

**Solution**:
1. Update plotly: `pip install --upgrade plotly`
2. Disable plots: `data.save_plots=false`
3. Check write permissions on output directory

## References

- **MIMIC-III Database**: https://mimic.physionet.org/
- **MIMIC-III Benchmarks**: https://github.com/YerevaNN/mimic3-benchmarks
- **Hydra Framework**: https://hydra.cc/
- **Original Notebook**: `data/prepare_mimic.ipynb`
- **Project Repository**: https://github.com/hojjatkarami/SynEHRgy

## Additional Documentation

- **CHECKLIST.md**: Step-by-step setup checklist
- **CONVERSION_SUMMARY.md**: Details on notebook-to-script conversion
- **QUICKSTART.md**: Minimal quick start guide (deprecated, see Quick Start section)
- **SUMMARY.md**: Project structure overview (deprecated, see Overview section)

## Advantages of This Pipeline

1. **Modularity**: Each step is independent and can be run separately
2. **Configuration Management**: Hydra provides flexible, reproducible configuration
3. **Reproducibility**: All parameters tracked in version-controlled config files
4. **Extensibility**: Easy to add new variables or modify discretization
5. **Debugging**: Inspect intermediate outputs at each step
6. **Reusability**: Well-documented functions for common operations

## Next Steps

After successfully running the pipeline:

1. **Train Models**: Use discretized data for training deep learning models
2. **Analyze Distributions**: Review plots to understand data characteristics
3. **Tune Parameters**: Experiment with different `min_th_icd` and `n_bins` values
4. **Extend Pipeline**: Add new variables or preprocessing steps
5. **Backup Data**: Save processed data for future experiments

---

**Questions or Issues?** Refer to the troubleshooting section or open an issue in the project repository.
