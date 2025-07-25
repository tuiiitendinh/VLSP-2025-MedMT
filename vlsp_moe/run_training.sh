#!/bin/bash

# Always run from the project root, regardless of where the script is called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
cd "$SCRIPT_DIR/.."

# MoE Model Training Pipeline
# This script runs the complete pipeline for training the MoE model

echo "Starting MoE Model Training Pipeline..."

# Step 1: Setup environment
echo "Step 1: Setting up environment..."
#pip install -r vlsp_moe/requirements.txt

# Step 1.1: Train SentencePiece tokenizer (if using SPM)
echo "Training SentencePiece tokenizer..."
python vlsp_moe/scripts/train_tokenizer.py --config vlsp_moe/configs/moe_config.yaml --output_dir vlsp_moe/tokenizer_output

# Step 1.2: Convert CSV to parallel TXT for MoE training
echo "Converting CSV to parallel TXT..."
python vlsp_moe/scripts/csv_to_parallel_txt.py

# Step 1.3: Convert CSV to JSONL for MoE training
echo "Converting CSV to JSONL..."
python vlsp_moe/scripts/convert_csv_to_jsonl.py

# Step 2: Prepare data for all experts
echo "Step 2: Preparing data for all experts..."
python vlsp_moe/scripts/prepare_data.py

# Step 3: Create processed data directory
echo "Step 3: Creating processed data directory..."
mkdir -p data/processed

# Step 4: Create validation split
echo "Step 4: Creating validation split..."
python vlsp_moe/scripts/create_validation_split.py

# Step 5: Train the MoE model
echo "Step 5: Training MoE model..."
python vlsp_moe/scripts/train_moe.py

# Step 6: Evaluate the model (optional)
echo "Step 6: Evaluating model..."
if [ -f "data/processed/val.jsonl" ]; then
    python vlsp_moe/scripts/evaluate_moe.py --model_path outputs/moe_model --interactive
else
    echo "Validation file not found. Skipping evaluation."
fi

echo "MoE Model Training Pipeline completed!"
echo "Model saved in: outputs/moe_model"
echo "Training logs saved in: logs/moe_training"