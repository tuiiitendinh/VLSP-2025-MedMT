#!/usr/bin/env python3
"""
Summary of the MoE data preparation results.
"""
import json
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def show_data_summary():
    """Show a summary of the prepared data."""
    print("="*60)
    print("MoE TRAINING DATA SUMMARY")
    print("="*60)
    
    # Check all processed files
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    files_info = {
        "medical_train.jsonl": "Medical Domain Expert",
        "en_vi_train.jsonl": "EN→VI Translation Expert", 
        "vi_en_train.jsonl": "VI→EN Translation Expert",
        "train.jsonl": "Combined Training Data",
        "val.jsonl": "Validation Data"
    }
    
    total_train = 0
    total_val = 0
    
    for filename, description in files_info.items():
        filepath = os.path.join(processed_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                count = sum(1 for line in f if line.strip())
            
            print(f"✅ {description}")
            print(f"   📁 {filename}")
            print(f"   📊 {count:,} samples")
            
            if filename == "train.jsonl":
                total_train = count
            elif filename == "val.jsonl":
                total_val = count
            
            # Show sample from each file
            with open(filepath, 'r', encoding='utf-8') as f:
                sample = json.loads(f.readline().strip())
                expert = sample.get('expert', 'unknown')
                user_content = sample['messages'][0]['content'][:80]
                print(f"   👤 Sample: {user_content}...")
                print(f"   🎯 Expert: {expert}")
            print()
        else:
            print(f"❌ {description}: File not found")
    
    print("="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"📚 Total Training Samples: {total_train:,}")
    print(f"🔍 Total Validation Samples: {total_val:,}")
    print(f"📈 Train/Val Ratio: {total_train/total_val:.1f}:1")
    print()
    
    # Show expert distribution in validation set
    if os.path.exists(os.path.join(processed_dir, "val.jsonl")):
        expert_counts = {}
        with open(os.path.join(processed_dir, "val.jsonl"), 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                expert = sample.get('expert', 'unknown')
                expert_counts[expert] = expert_counts.get(expert, 0) + 1
        
        print("🎯 Expert Distribution in Validation Set:")
        for expert, count in sorted(expert_counts.items()):
            percentage = (count / total_val) * 100
            print(f"   {expert}: {count:,} samples ({percentage:.1f}%)")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. 🚀 Start training: python train_moe.py")
    print("2. 📊 Monitor training: Check logs/moe_training/")
    print("3. 🔍 Evaluate model: python evaluate_moe.py")
    print("4. 💬 Interactive test: python evaluate_moe.py --interactive")
    print("="*60)

if __name__ == "__main__":
    show_data_summary()
