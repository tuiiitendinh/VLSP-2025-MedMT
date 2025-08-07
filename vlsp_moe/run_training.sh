#!/bin/bash


export MY_SCRATCH_DIR="/scratch/users/sutd/1010047/VLSP-2025-MedMT"

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
# --- CORRECT PIPELINE ORDER ---
# ==============================================================================

# Always run from the project root, regardless of where the script is called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
cd "$SCRIPT_DIR/.."

# Activate your conda environment (adjust if needed)
# source /path/to/your/conda/etc/profile.d/conda.sh
# conda activate env_vlsp2025


echo "Step 1: Converting CSV to JSONL for MoE training..."
python /home/users/sutd/1010047/VLSP-2025-MedMT/vlsp_moe/scripts/convert_csv_to_jsonl.py

echo "Step 2: Converting CSV to parallel TXT..."
python /home/users/sutd/1010047/VLSP-2025-MedMT/vlsp_moe/scripts/csv_to_parallel_txt.py \
    --output_dir "$DATA_DIR"

echo "Step 3: Preparing data for all experts..."
python /home/users/sutd/1010047/VLSP-2025-MedMT/vlsp_moe/scripts/prepare_data.py \
    --input_dir "$DATA_DIR" \
    --output_dir "$DATA_DIR/processed"

echo "Step 4: Creating validation split..."
python /home/users/sutd/1010047/VLSP-2025-MedMT/vlsp_moe/scripts/create_validation_split.py \
    --data_dir "$DATA_DIR/processed"

# ==============================================================================
# --- MEDICAL KEYWORD EXTRACTION (Data-Driven) ---
# ==============================================================================

echo "Step 5: Analyzing medical keywords from data..."
python vlsp_moe/scripts/analyze_medical_keywords.py

echo "Step 6: Extracting and saving medical keywords..."
python vlsp_moe/scripts/extract_and_save_keywords.py

echo "Step 7: Validating data-driven extraction quality..."
python vlsp_moe/scripts/validate_data_driven_extraction.py

echo "Step 8: Enhancing tokenizer training data with medical keywords..."
python vlsp_moe/scripts/enhance_tokenizer_with_medical_keywords.py \
    --tokenizer_data "$DATA_DIR/tokenizer/training_data/combined_tokenizer_data.txt" \
    --medical_keywords "$DATA_DIR/../vlsp_moe/medical_keywords.json" \
    --output "$DATA_DIR/tokenizer/training_data/enhanced_tokenizer_data.txt"

echo "Step 9: Training SentencePiece tokenizer (with medical keywords)..."
python vlsp_moe/scripts/train_tokenizer.py \
    --config /home/users/sutd/1010047/VLSP-2025-MedMT/vlsp_moe/configs/moe_config.yaml \
    --output_dir "$DATA_DIR/tokenizer"

# ==============================================================================
# --- MODEL TRAINING AND EVALUATION ---
# ==============================================================================

echo "!!   train_file: ${DATA_DIR}/processed/train.jsonl"
echo "!!   val_file:   ${DATA_DIR}/processed/val.jsonl"
echo "!!   output_dir: ${OUTPUT_DIR}/moe_model_best"
echo "!!   logging_dir: ${LOGS_DIR}/moe_training"

echo "Step 10: Training MoE model..."
python /home/users/sutd/1010047/VLSP-2025-MedMT/vlsp_moe/scripts/train_moe.py

# Step 11: Evaluate the model (optional)
echo "Step 11: Evaluating model..."
if [ -f "${DATA_DIR}/processed/val.jsonl" ]; then
    python vlsp_moe/scripts/evaluate_moe.py --model_path ${OUTPUT_DIR}/moe_model --interactive
else
    echo "Validation file not found. Skipping evaluation."
fi

# Step 12: Test multi-model MoE setup
echo "Step 12: Testing multi-model MoE setup..."
python vlsp_moe/scripts/test_multi_model_moe.py

# Step 13: Run inference with multi-model MoE
echo "Step 13: Running inference with multi-model MoE..."
python vlsp_moe/scripts/inference_multi_model.py

echo "MoE Model Training Pipeline completed!"
echo "Model saved in: ${OUTPUT_DIR}/moe_model"
echo "Training logs saved in: ${LOGS_DIR}/moe_training"

# Display training summary
echo "Training Summary:"
echo "- Number of experts: 4"
echo "- Expert 1: Medical reasoning (prior model - Sculptor-Qwen3_Med-Reasoning)"
echo "- Expert 2: Medical domain translation (EN->VI)"
echo "- Expert 3: General translation (EN->VI)"
echo "- Expert 4: General translation (VI->EN)"
echo "- Base model: Qwen/Qwen3-1.7B"
echo "- Training technique: LoRA with multi-model MoE routing"
echo "- Medical keyword extraction: Fully data-driven (no hardcoded patterns)"