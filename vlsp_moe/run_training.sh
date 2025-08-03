#!/bin/bash


export MY_SCRATCH_DIR="/scratch/users/sutd/1010042/VLSP-2025-MedMT"

export DATA_DIR="${MY_SCRATCH_DIR}/VLSP-2025-MedMT/data"
export CACHE_DIR="${MY_SCRATCH_DIR}/VLSP-2025-MedMT/cache"
export OUTPUT_DIR="${MY_SCRATCH_DIR}/VLSP-2025-MedMT/outputs"
export LOGS_DIR="${MY_SCRATCH_DIR}/VLSP-2025-MedMT/logs"
export TMP_DIR="${MY_SCRATCH_DIR}/VLSP-2025-MedMT/tmp"

# Set the environment variables that Hugging Face and other libraries will use
export HF_HOME=$CACHE_DIR
export TMPDIR=$TMP_DIR

# Create all necessary directories in scratch to prevent errors
mkdir -p $DATA_DIR/processed
mkdir -p $CACHE_DIR
mkdir -p $OUTPUT_DIR
mkdir -p $LOGS_DIR
mkdir -p $TMP_DIR

echo "---------------------------------------------------------"
echo "Pipeline starting. All files will be written to your scratch space:"
echo "Data Directory:   $DATA_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Cache Directory:  $CACHE_DIR"
echo "Temp Directory:   $TMP_DIR"
echo "---------------------------------------------------------"


# ==============================================================================
# --- 2. RUN THE PIPELINE ---
# ==============================================================================

# Always run from the project root, regardless of where the script is called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
cd "$SCRIPT_DIR/.."

# Activate your conda environment (adjust if needed)
# source /path/to/your/conda/etc/profile.d/conda.sh
# conda activate env_vlsp2025


echo "Step 1.1: Training SentencePiece tokenizer..."
# NOTE: The output directory is now pointed to your scratch data directory
python vlsp_moe/scripts/train_tokenizer.py \
    --config /home/users/sutd/1010042/VLSP-2025-MedMT/vlsp_moe/configs/moe_config.yaml \
    --output_dir "$DATA_DIR/tokenizer"


echo "Step 1.2: Converting CSV to parallel TXT..."
# NOTE: We tell the script to save its output to the scratch data directory
python /home/users/sutd/1010042/VLSP-2025-MedMT/vlsp_moe/scripts/csv_to_parallel_txt.py \
    --output_dir "$DATA_DIR"


# NOTE: These scripts must also be modified to accept --input_dir and --output_dir arguments
# similar to the other scripts to work correctly in this pipeline.
echo "Step 1.3: Converting CSV to JSONL..."
python /home/users/sutd/1010042/VLSP-2025-MedMT/vlsp_moe/scripts/convert_csv_to_jsonl.py \
    --output_dir "$DATA_DIR/processed"


echo "Step 2: Preparing data for all experts..."
python /home/users/sutd/1010042/VLSP-2025-MedMT/vlsp_moe/scripts/prepare_data.py \
    --input_dir "$DATA_DIR" \
    --output_dir "$DATA_DIR/processed"


echo "Step 4: Creating validation split..."
python /home/users/sutd/1010042/VLSP-2025-MedMT/vlsp_moe/scripts/create_validation_split.py \
    --data_dir "$DATA_DIR/processed"


echo "!!   train_file: ${DATA_DIR}/processed/train.jsonl"
echo "!!   val_file:   ${DATA_DIR}/processed/val.jsonl"
echo "!!   output_dir: ${OUTPUT_DIR}/moe_model_best"
echo "!!   logging_dir: ${LOGS_DIR}/moe_training"


echo "Step 5: Training MoE model..."
python /home/users/sutd/1010042/VLSP-2025-MedMT/vlsp_moe/scripts/train_moe.py

# The evaluate script will read the model path from the config file's output_dir
echo "Step 6: Evaluating model..."
if [ -f "${DATA_DIR}/processed/val.jsonl" ]; then
    python /home/users/sutd/1010042/VLSP-2025-MedMT/vlsp_moe/scripts/evaluate_moe.py \
        --model_path "${OUTPUT_DIR}/moe_model_best" --interactive
else
    echo "Validation file not found in scratch directory. Skipping evaluation."
fi

echo "MoE Model Training Pipeline completed!"
echo "Model saved in: ${OUTPUT_DIR}/moe_model_best"
echo "Training logs saved in: ${LOGS_DIR}/moe_training"