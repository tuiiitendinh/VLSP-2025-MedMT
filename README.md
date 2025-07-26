# MoE Model for Machine Translation

This implementation provides a Mixture of Experts (MoE) model for machine translation tasks, specifically designed for Vietnamese-English translation with domain adaptation for medical texts.

## Overview

The MoE model consists of three specialized experts:

1. **Medical Expert**: Specializes in translating medical texts from English to Vietnamese
2. **EN-VI Expert**: Handles general English to Vietnamese translation
3. **VI-EN Expert**: Handles general Vietnamese to English translation

## Architecture

- **Base Model**: Qwen/Qwen1.5-1.8B
- **Training Technique**: LoRA (Low-Rank Adaptation) with MoE routing
- **Gating Network**: Neural network that routes inputs to appropriate experts
- **Expert Selection**: Based on task type and domain

## File Structure

```
vlsp_moe/
├── configs/
│   └── moe_config.yaml          # Configuration file
├── scripts/
│   ├── prepare_data.py          # Data preparation script
│   ├── train_moe.py            # MoE training script
│   ├── evaluate_moe.py         # Evaluation script
│   └── config.py               # Configuration loader
├── data/
│   ├── medical/                # Medical domain data
│   ├── en-vi/                  # EN-VI translation data
│   ├── vi-en/                  # VI-EN translation data
│   └── processed/              # Processed JSONL files
├── outputs/                    # Model checkpoints
├── logs/                       # Training logs
├── requirements.txt            # Python dependencies
└── run_training.sh            # Training pipeline script
```

## Usage

### 1. Setup Environment

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place your training data in the appropriate directories:
- Medical domain: `data/medical/train.{en,vi}.txt`
- General EN-VI: `data/en-vi/train.{en,vi}.txt`
- General VI-EN: `data/vi-en/train.{vi,en}.txt`

### 3. Run Training Pipeline

```bash
./run_training.sh
```

Or run individual steps:

```bash
# Prepare data
cd scripts
python prepare_data.py

# Train model
python train_moe.py

# Evaluate model
python evaluate_moe.py --model_path ../outputs/moe_model --interactive
```

### 4. Configuration

Edit `configs/moe_config.yaml` to adjust:
- Model parameters
- Training hyperparameters
- MoE-specific settings
- Data paths

## Key Features

### Data Preparation
- Converts parallel text files to JSONL format
- Adds expert labels for routing
- Combines multiple expert datasets

### MoE Training
- **Gating Network**: Routes inputs to appropriate experts
- **Expert-specific LoRA**: Separate adapters for each expert
- **Load Balancing**: Encourages equal expert usage
- **Routing Loss**: Promotes sparsity in expert selection

### Evaluation
- Supports both batch and interactive evaluation
- Tracks expert usage statistics
- Generates translation quality metrics

## Training Details

### Model Architecture
- **Base Model**: Qwen1.5-1.8B transformer
- **Experts**: 3 task-specific experts
- **Routing**: Learned gating network
- **Adaptation**: LoRA with r=8, alpha=16

### Training Parameters
- **Batch Size**: 4 per device
- **Learning Rate**: 2e-4
- **Max Steps**: 3000
- **Gradient Accumulation**: 2 steps
- **Warmup Steps**: 100

### Expert Routing
- **Medical Expert**: Activated for medical domain texts
- **EN-VI Expert**: Activated for English to Vietnamese translation
- **VI-EN Expert**: Activated for Vietnamese to English translation

## Inference

### Interactive Mode
```bash
python evaluate_moe.py --model_path outputs/moe_model --interactive
```

### Batch Evaluation
```bash
python evaluate_moe.py --model_path outputs/moe_model --test_file data/processed/val.jsonl
```

### Programmatic Usage
```python
from evaluate_moe import MoEInference

# Initialize model
inference = MoEInference("outputs/moe_model", "configs/moe_config.yaml")

# Translate text
translation = inference.translate(
    text="Hello world",
    source_lang="en",
    target_lang="vi",
    domain=None  # or "medical" for medical domain
)
```

## Customization

### Adding New Experts
1. Update `expert_mapping` in `train_moe.py`
2. Add expert configuration in `moe_config.yaml`
3. Prepare data with new expert labels
4. Update gating network for additional experts

### Modifying Routing Strategy
- Edit `MoEGatingNetwork` class for custom routing logic
- Adjust routing loss in `compute_routing_loss`
- Modify expert weight calculation

## Performance Considerations

- **Memory**: Each expert uses additional LoRA parameters
- **Compute**: Gating network adds minimal overhead
- **Storage**: Model size increases with number of experts
- **Inference**: Expert routing adds slight latency

## Troubleshooting

### Common Issues
1. **CUDA/MPS Setup**: Ensure proper device configuration
2. **Memory Issues**: Reduce batch size or use gradient accumulation
3. **Data Format**: Verify JSONL format and expert labels
4. **Model Loading**: Check model path and permissions

### Performance Tips
- Use mixed precision training (bf16/fp16)
- Implement gradient checkpointing for large models
- Monitor expert usage for load balancing
- Use appropriate learning rates for different experts

## Future Enhancements

- [ ] Dynamic expert addition/removal
- [ ] Hierarchical expert routing
- [ ] Cross-lingual expert knowledge transfer
- [ ] Adaptive load balancing strategies
- [ ] Multi-modal expert integration
