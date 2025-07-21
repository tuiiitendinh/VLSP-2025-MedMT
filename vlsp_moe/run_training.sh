#!/bin/bash

# MoE Model Training Pipeline
# This script runs the complete pipeline for training the MoE model

echo "Starting MoE Model Training Pipeline..."

# Step 1: Setup environment
echo "Step 1: Setting up environment..."
pip install -r requirements.txt

# Step 1.1: Convert CSV to JSONL for MoE training
echo "Converting CSV to JSONL..."
python vlsp_moe/scripts/convert_csv_to_jsonl.py

# Step 2: Prepare data for all experts
echo "Step 2: Preparing data for all experts..."
cd scripts
python prepare_data.py

# Step 3: Create processed data directory
echo "Step 3: Creating processed data directory..."
mkdir -p ../data/processed

# Step 4: Create validation split
echo "Step 4: Creating validation split..."
python create_validation_split.py

# Step 5: Train the MoE model
echo "Step 5: Training MoE model..."
python train_moe.py

# Step 6: Evaluate the model (optional)
echo "Step 6: Evaluating model..."
if [ -f "../data/processed/val.jsonl" ]; then
    python evaluate_moe.py --model_path ../outputs/moe_model --interactive
else
    echo "Validation file not found. Skipping evaluation."
fi

echo "MoE Model Training Pipeline completed!"
echo "Model saved in: ../outputs/moe_model"
echo "Training logs saved in: ../logs/moe_training"

# Display training summary
echo "Training Summary:"
echo "- Number of experts: 3"
echo "- Expert 1: Medical domain (EN->VI)"
echo "- Expert 2: General translation (EN->VI)"
echo "- Expert 3: General translation (VI->EN)"
echo "- Base model: Qwen/Qwen3-1.7B"
echo "- Training technique: LoRA with MoE routing"
