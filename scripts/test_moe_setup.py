#!/usr/bin/env python3
"""
Simple test script for MoE model setup.
"""

import torch
from config import Config
from train_moe import create_moe_model, MoEDataset
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def test_moe_setup():
    """Test MoE model setup without full training."""
    print("🧪 Testing MoE Model Setup...")
    
    # Load config
    config = Config(os.path.join(PROJECT_ROOT, "configs", "moe_config.yaml"))
    print(f"✅ Config loaded - Device: {config.device}")
    
    # Test if data exists
    train_file = config.dataset["train_file"]
    if not os.path.isabs(train_file):
        train_file = os.path.join(PROJECT_ROOT, train_file)
    if not os.path.exists(train_file):
        print(f"❌ Training file not found: {train_file}")
        return False
    
    # Test dataset loading
    print("📊 Testing dataset loading...")
    try:
        # Create a small sample for testing
        with open(train_file, 'r', encoding='utf-8') as f:
            sample_lines = [f.readline() for _ in range(5)]
        
        # Create temporary test file
        test_file = os.path.join(PROJECT_ROOT, "data", "processed", "test_sample.jsonl")
        with open(test_file, 'w', encoding='utf-8') as f:
            for line in sample_lines:
                if line.strip():
                    f.write(line)
        
        # Test tokenizer loading
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model["model_id_or_path"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"✅ Tokenizer loaded - Vocab size: {len(tokenizer)}")
        
        # Test dataset
        dataset = MoEDataset(test_file, tokenizer)
        print(f"✅ Dataset loaded - Size: {len(dataset)}")
        
        # Test sample
        sample = dataset[0]
        print(f"✅ Sample loaded - Keys: {list(sample.keys())}")
        print(f"   Expert: {sample['expert_type']}")
        print(f"   Input shape: {len(sample['input_ids'])}")
        
        # Clean up
        os.remove(test_file)
        
        return True
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

def test_model_loading():
    """Test model loading without training."""
    print("🤖 Testing Model Loading...")
    
    try:
        config = Config(os.path.join(PROJECT_ROOT, "configs", "moe_config.yaml"))
        
        # Test small model loading for speed
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, TaskType
        
        print("📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.model["model_id_or_path"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("📥 Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            config.model["model_id_or_path"],
            torch_dtype=torch.float16,  
            device_map="auto"
        )
        
        print("🔧 Applying LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora["r"],
            lora_alpha=config.lora["lora_alpha"],
            lora_dropout=config.lora["lora_dropout"],
            bias=config.lora["bias"]
        )
        model = get_peft_model(model, lora_config)
        
        print(f"✅ Model loaded successfully")
        print(f"   Model parameters: {model.num_parameters():,}")
        print(f"   Trainable parameters: {model.num_parameters(only_trainable=True):,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("MoE SETUP VALIDATION")
    print("="*60)
    
    data_ok = test_moe_setup()
    print()
    
    if data_ok:
        model_ok = test_model_loading()
        print()
        
        if model_ok and data_ok:
            print("🎉 All tests passed! Ready for MoE training.")
            print()
            print("Next steps:")
            print("1. Run: python train_moe.py")
            print("2. Monitor training in logs/moe_training/")
            print("3. Evaluate with: python evaluate_moe.py")
        else:
            print("⚠️  Some tests failed. Please fix issues before training.")
    else:
        print("❌ Data setup failed. Please run prepare_data.py first.")

if __name__ == "__main__":
    main()
