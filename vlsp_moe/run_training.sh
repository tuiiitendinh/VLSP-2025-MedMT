    #!/bin/bash
    # This script runs the full data preparation, keyword extraction, and MoE model training pipeline.
    # It is designed to be portable and robust, exiting immediately on any error.

    # Exit immediately if a command exits with a non-zero status.
    set -e

    # --- 1. DYNAMIC CONFIGURATION AND SETUP ---

    # Determine the project's root directory dynamically.
    # This makes the script portable and runnable from anywhere.
    readonly PROJECT_ROOT="/scratch/users/sutd/1010047/VLSP-2025-MedMT"

    # Define all other directories relative to the project root.
    readonly DATA_DIR="${PROJECT_ROOT}/data"
    readonly CACHE_DIR="${PROJECT_ROOT}/cache"
    readonly LOGS_DIR="${PROJECT_ROOT}/logs"
    readonly TMP_DIR="${PROJECT_ROOT}/tmp"
    readonly OUTPUT_DIR="${PROJECT_ROOT}/outputs"
    readonly MODEL_OUTPUT_DIR="${OUTPUT_DIR}/moe_0908_model_best"
    readonly COMBINED_TOKENIZER_DATA_FILE="${DATA_DIR}/combined.txt"

    # Define script and config locations for conciseness
    readonly SCRIPTS_DIR="${PROJECT_ROOT}/vlsp_moe/scripts"
    readonly CONFIG_FILE="${PROJECT_ROOT}/vlsp_moe/configs/moe_config.yaml"

    # Define key file paths for clarity
    readonly MEDICAL_KEYWORDS_FILE="${PROJECT_ROOT}/vlsp_moe/medical_keywords.json"
    readonly TOKENIZER_TRAIN_DATA_DIR="${DATA_DIR}/tokenizer/training_data"
    readonly ENHANCED_TOKENIZER_DATA_FILE="${TOKENIZER_TRAIN_DATA_DIR}/enhanced_tokenizer_data.txt"

    # Set environment variables for Hugging Face and other tools
    export HF_HOME=$CACHE_DIR
    export TMPDIR=$TMP_DIR

    # Create all necessary directories within the project
    mkdir -p "$DATA_DIR/processed"
    mkdir -p "$CACHE_DIR"
    mkdir -p "$LOGS_DIR"
    mkdir -p "$TMP_DIR"
    mkdir -p "$MODEL_OUTPUT_DIR"
    mkdir -p "$TOKENIZER_TRAIN_DATA_DIR"

    # Change to the project root directory for stable relative paths
    cd "$PROJECT_ROOT"

    # Helper function to run and log each step
    run_step() {
        local description="$1"
        echo "---------------------------------------------------------"
        echo "=> EXECUTING: $description"
        echo "---------------------------------------------------------"
        shift
        "$@"
        if [ $? -ne 0 ]; then
            echo "❌ ERROR: Step '$description' failed. Aborting pipeline."
            exit 1
        fi
        echo "✅ SUCCESS: Step '$description' completed."
        echo
    }

    echo "========================================================="
    echo "MoE Model Training Pipeline Started"
    echo "Project Root:    $PROJECT_ROOT"
    echo "All outputs will be saved within this directory."
    echo "========================================================="


    # --- 2. DATA PREPARATION ---

    # NOTE: Added --output_dir argument. Please verify it's correct for your script.
    run_step "Step 1: Converting CSV to JSONL for MoE training" \
        python "${SCRIPTS_DIR}/convert_csv_to_jsonl.py" \
        --output_dir "${DATA_DIR}/processed"

    run_step "Step 2: Converting CSV to parallel TXT for tokenizer" \
        python "${SCRIPTS_DIR}/csv_to_parallel_txt.py" \
        --output_dir "$DATA_DIR"

    run_step "Step 3: Preparing data for all experts" \
        python "${SCRIPTS_DIR}/prepare_data.py" \
        --input_dir "$DATA_DIR" \
        --output_dir "${DATA_DIR}/processed"

    run_step "Step 4: Creating validation split" \
        python "${SCRIPTS_DIR}/create_validation_split.py" \
        --data_dir "${DATA_DIR}/processed"


    # --- 3. DATA-DRIVEN MEDICAL KEYWORD EXTRACTION & TOKENIZER ENHANCEMENT ---

    # NOTE: Assumed these scripts also need input/output arguments. Please verify.
    run_step "Step 5: Analyzing medical keywords from data" \
        python "${SCRIPTS_DIR}/analyze_medical_keywords.py" \
        --input_file "${DATA_DIR}/processed/medical_train.jsonl" \
        --output_file "${DATA_DIR}/processed/keyword_analysis.json"

    run_step "Step 6: Extracting and saving medical keywords" \
        python "${SCRIPTS_DIR}/extract_and_save_keywords.py" \
        --input_file "${DATA_DIR}/processed/keyword_analysis.json" \
        --output_file "$MEDICAL_KEYWORDS_FILE"

    run_step "Step 7: Validating data-driven extraction quality" \
        python "${SCRIPTS_DIR}/validate_data_driven_extraction.py" \
        --keywords_file "$MEDICAL_KEYWORDS_FILE"

    # --- CREATE THE COMBINED TOKENIZER DATA ---
    cat "${DATA_DIR}/train.en.txt" "${DATA_DIR}/train.vi.txt" > "$COMBINED_TOKENIZER_DATA_FILE"

    run_step "Step 8: Enhancing tokenizer data with medical keywords" \
        python "${SCRIPTS_DIR}/enhance_tokenizer_with_medical_keywords.py" \
        --tokenizer_data "$COMBINED_TOKENIZER_DATA_FILE" \
        --medical_keywords "$MEDICAL_KEYWORDS_FILE" \
        --output "$ENHANCED_TOKENIZER_DATA_FILE"

    run_step "Step 9: Training SentencePiece tokenizer (with medical keywords)" \
        python "${SCRIPTS_DIR}/train_tokenizer.py" \
        --config "$CONFIG_FILE" \
        --output_dir "${DATA_DIR}/tokenizer" \
        --training_text_file "$ENHANCED_TOKENIZER_DATA_FILE"


    # --- 4. MODEL TRAINING AND EVALUATION ---

    echo "---------------------------------------------------------"
    echo "Pipeline configured with the following files:"
    echo "   Train File:   ${DATA_DIR}/processed/train.jsonl"
    echo "   Val File:     ${DATA_DIR}/processed/val.jsonl"
    echo "   Output Dir:   $MODEL_OUTPUT_DIR"
    echo "   Logging Dir:  $LOGS_DIR"
    echo "---------------------------------------------------------"

    run_step "Step 10: Training the MoE model" \
        python "${SCRIPTS_DIR}/train_moe.py" \
        --config "$CONFIG_FILE"
        # Ensure train_moe.py reads paths from the YAML config.

    run_step "Step 11: Evaluating the best model" \
        python "${SCRIPTS_DIR}/evaluate_moe.py" \
        --model_path "$MODEL_OUTPUT_DIR" \
        --test_file "${DATA_DIR}/processed/val.jsonl" \
        --interactive

    run_step "Step 12: Testing multi-model MoE setup" \
        python "${SCRIPTS_DIR}/test_multi_model_moe.py"

    run_step "Step 13: Running inference with multi-model MoE" \
        python "${SCRIPTS_DIR}/inference_multi_model.py"


    # --- 5. COMPLETION ---

    echo "========================================================="
    echo "✅ MoE Model Training Pipeline Completed Successfully!"
    echo "========================================================="
    echo "Model saved in:         $MODEL_OUTPUT_DIR"
    echo "Training logs saved in: $LOGS_DIR"
    echo
    echo "Training Summary:"
    echo "- Number of experts: 4"
    echo "- Expert 1: Medical reasoning (prior model - Sculptor-Qwen3_Med-Reasoning)"
    echo "- Expert 2: Medical domain translation (EN->VI)"
    echo "- Expert 3: General translation (EN->VI)"
    echo "- Expert 4: General translation (VI->EN)"
    echo "- Base model: Qwen/Qwen3-1.7B"
    echo "- Training technique: LoRA with multi-model MoE routing"
    echo "- Medical keyword extraction: Fully data-driven (no hardcoded patterns)"
    echo "========================================================="