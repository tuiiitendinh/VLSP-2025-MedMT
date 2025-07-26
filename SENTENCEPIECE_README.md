# SentencePiece Tokenizer Integration for MoE Model

This document describes how to use SentencePiece tokenizer with the MoE (Mixture of Experts) model instead of the default HuggingFace tokenizer.

## Overview

The project now supports two tokenizer types:
1. **HuggingFace Tokenizer** (default) - Uses the tokenizer from the pre-trained model
2. **SentencePiece Tokenizer** - Custom trained SentencePiece model with domain-specific vocabulary

## Features

- **Custom Vocabulary**: Train tokenizer specifically on your medical domain data
- **Special Tokens**: Support for task-specific markers (`<medical>`, `<en_vi>`, `<vi_en>`)
- **Seamless Integration**: Drop-in replacement for HuggingFace tokenizer
- **Configurable**: Easy switching between tokenizer types
- **Optimized**: Better compression ratio for domain-specific text

## Quick Start

### 1. Configure SentencePiece Tokenizer

```bash
# Switch to SentencePiece tokenizer
python vlsp_moe/scripts/tokenizer_manager.py --set sentencepiece

# Validate configuration
python vlsp_moe/scripts/tokenizer_manager.py --validate
```

### 2. Train the Tokenizer

```bash
# Train SentencePiece tokenizer on your data
python vlsp_moe/scripts/train_tokenizer.py --config vlsp_moe/configs/moe_config.yaml
```

### 3. Test the Tokenizer

```bash
# Test tokenizer functionality
python vlsp_moe/scripts/test_tokenizer.py \
    --tokenizer_path vlsp_moe/tokenizer_output/spm_model.model \
    --test_data data/processed/train.jsonl
```

### 4. Train MoE Model

```bash
# Train model with SentencePiece tokenizer
python vlsp_moe/scripts/train_moe.py
```

## Configuration

### SentencePiece Configuration in `moe_config.yaml`

```yaml
tokenizer:
  type: sentencepiece          # Set to 'sentencepiece' to enable
  vocab_size: 32000           # Vocabulary size
  model_prefix: spm_model     # Output model prefix
  model_type: unigram         # unigram, bpe, char, word
  character_coverage: 0.995   # Character coverage for rare characters
  
  # Tokenization settings
  split_by_unicode_script: true
  split_by_number: true
  split_by_whitespace: true
  split_digits: true
  
  # Special token IDs
  pad_id: 0
  unk_id: 1
  bos_id: 2
  eos_id: 3
  
  # Special symbols for domain tasks
  user_defined_symbols: 
    - "<|im_start|>"
    - "<|im_end|>"
    - "<medical>"
    - "<en_vi>"
    - "<vi_en>"
```

### Switch Back to HuggingFace Tokenizer

```bash
# Switch back to HuggingFace tokenizer
python vlsp_moe/scripts/tokenizer_manager.py --set huggingface
```

## Scripts Reference

### Core Scripts

| Script | Purpose |
|--------|---------|
| `tokenizer_manager.py` | Configure and manage tokenizer settings |
| `train_tokenizer.py` | Train SentencePiece tokenizer |
| `test_tokenizer.py` | Test and evaluate tokenizer |
| `prepare_tokenizer_data.py` | Extract training data from JSONL files |

### Usage Examples

#### Train Tokenizer with Custom Settings

```bash
python vlsp_moe/scripts/train_tokenizer.py \
    --config vlsp_moe/configs/moe_config.yaml \
    --output_dir vlsp_moe/tokenizer_output \
    --force  # Force retrain even if model exists
```

#### Test Tokenizer Performance

```bash
python vlsp_moe/scripts/test_tokenizer.py \
    --tokenizer_path vlsp_moe/tokenizer_output/spm_model.model \
    --config vlsp_moe/configs/moe_config.yaml \
    --test_data data/processed/val.jsonl \
    --compare_hf Qwen/Qwen3-1.7B \
    --limit 100
```

#### Check Current Configuration

```bash
# Show current tokenizer configuration
python vlsp_moe/scripts/tokenizer_manager.py --show

# Validate SentencePiece setup
python vlsp_moe/scripts/tokenizer_manager.py --validate
```

## Files Structure

```
vlsp_moe/
├── scripts/
│   ├── spm_tokenizer.py           # SentencePiece tokenizer wrapper
│   ├── spm_data_collator.py       # Data collator for SPM tokens
│   ├── train_tokenizer.py         # Train SPM tokenizer
│   ├── test_tokenizer.py          # Test SPM tokenizer
│   ├── tokenizer_manager.py       # Manage tokenizer configuration
│   ├── prepare_tokenizer_data.py  # Prepare training data
│   └── train_moe.py               # Updated MoE training script
├── configs/
│   └── moe_config.yaml            # Updated configuration
├── tokenizer_output/              # Trained tokenizer files
│   ├── spm_model.model           # Trained SentencePiece model
│   ├── spm_model.vocab           # Vocabulary file
│   └── training_data/            # Training data for tokenizer
└── tokenizer_data/               # Temporary data files
```

## Training Pipeline

The updated training pipeline now includes tokenizer training:

```bash
./vlsp_moe/run_training.sh
```

This will:
1. **Train SentencePiece tokenizer** (if configured)
2. Convert CSV to parallel text and JSONL
3. Prepare data for all experts
4. Create validation split
5. **Train MoE model** (with SPM tokenizer)
6. Evaluate the model

## Advanced Usage

### Custom Tokenizer Training

```python
from vlsp_moe.scripts.spm_tokenizer import create_spm_tokenizer

# Custom tokenizer configuration
config = {
    'vocab_size': 50000,
    'model_type': 'bpe',
    'character_coverage': 0.9995,
    'user_defined_symbols': ['<custom_token>']
}

# Train tokenizer
tokenizer = create_spm_tokenizer(
    config=config,
    data_files=['data1.txt', 'data2.txt'],
    model_path=None
)
```

### Using SPM Tokenizer in Code

```python
from vlsp_moe.scripts.smp_tokenizer import SPMTokenizer

# Load trained tokenizer
tokenizer = SPMTokenizer(
    model_path='vlsp_moe/tokenizer_output/spm_model.model',
    config=tokenizer_config
)

# Tokenize text
text = "Medical translation example"
encoded = tokenizer.encode(text, add_special_tokens=True)
decoded = tokenizer.decode(encoded, skip_special_tokens=True)

# Batch processing (transformers-like interface)
result = tokenizer(
    ["Text 1", "Text 2"],
    max_length=128,
    padding=True,
    truncation=True,
    return_tensors="pt"
)
```

## Benefits of SentencePiece

1. **Domain Adaptation**: Vocabulary optimized for medical translation
2. **Language Support**: Better handling of Vietnamese and English
3. **Compression**: More efficient token representation
4. **Consistency**: Language-agnostic tokenization
5. **Special Tokens**: Built-in support for task markers

## Troubleshooting

### Common Issues

1. **Import Error**: Install SentencePiece
   ```bash
   pip install sentencepiece
   ```

2. **Training Fails**: Check data files exist
   ```bash
   python vlsp_moe/scripts/prepare_tokenizer_data.py --config vlsp_moe/configs/moe_config.yaml
   ```

3. **Model Not Found**: Train tokenizer first
   ```bash
   python vlsp_moe/scripts/train_tokenizer.py
   ```

4. **Configuration Issues**: Validate setup
   ```bash
   python vlsp_moe/scripts/tokenizer_manager.py --validate
   ```

### Debug Mode

Enable verbose logging for debugging:

```bash
python vlsp_moe/scripts/train_tokenizer.py --verbose
python vlsp_moe/scripts/test_tokenizer.py --verbose
```

## Performance Comparison

Use the test script to compare SentencePiece vs HuggingFace tokenizer:

```bash
python vlsp_moe/scripts/test_tokenizer.py \
    --tokenizer_path vlsp_moe/tokenizer_output/spm_model.model \
    --compare_hf Qwen/Qwen3-1.7B \
    --test_data data/processed/val.jsonl
```

This will show:
- Token count comparison
- Compression ratios
- Roundtrip accuracy
- Processing speed

## Notes

- SentencePiece model training requires sufficient training data
- First run will take longer due to tokenizer training
- Trained models are cached and reused
- Configuration changes require tokenizer retraining
