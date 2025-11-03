#!/bin/bash
# Script to run MIMIC-III to HuggingFace dataset conversion

# Default: Run with full dataset
echo "=========================================="
echo "MIMIC-III to HuggingFace Dataset Converter"
echo "=========================================="
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Check if we want to run a test
if [ "$1" == "test" ]; then
    echo "Running test with 100 patients..."
    python prepare_mimic_hf.py max_patients=100 output_name=mimic3_test
elif [ "$1" == "small" ]; then
    echo "Running with 1000 patients..."
    python prepare_mimic_hf.py max_patients=1000 output_name=mimic3_small
elif [ "$1" == "help" ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Usage: ./run_prepare_mimic_hf.sh [option]"
    echo ""
    echo "Options:"
    echo "  test       - Process 100 patients (quick test)"
    echo "  small      - Process 1000 patients"
    echo "  (no args)  - Process all patients (full dataset)"
    echo "  help       - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_prepare_mimic_hf.sh test"
    echo "  ./run_prepare_mimic_hf.sh small"
    echo "  ./run_prepare_mimic_hf.sh"
    echo ""
    echo "Custom overrides:"
    echo "  python prepare_mimic_hf.py max_patients=500 output_name=custom"
else
    echo "Running with all patients..."
    echo "This may take a while for the full MIMIC-III dataset."
    echo ""
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python prepare_mimic_hf.py
    else
        echo "Cancelled."
        exit 0
    fi
fi

echo ""
echo "Done!"
