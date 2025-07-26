#!/usr/bin/env python3
"""
Test script to verify data preparation works correctly.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def test_data_preparation():
    """Test the data preparation process."""
    
    # Check if the main data files exist
    en_file = os.path.join(PROJECT_ROOT, "data", "train.en.txt")
    vi_file = os.path.join(PROJECT_ROOT, "data", "train.vi.txt")
    
    print("Testing data preparation...")
    
    if not os.path.exists(en_file):
        print(f"❌ English file not found: {en_file}")
        return False
    
    if not os.path.exists(vi_file):
        print(f"❌ Vietnamese file not found: {vi_file}")
        return False
    
    # Check file sizes
    with open(en_file, 'r', encoding='utf-8') as f:
        en_lines = len([line for line in f if line.strip()])
    
    with open(vi_file, 'r', encoding='utf-8') as f:
        vi_lines = len([line for line in f if line.strip()])
    
    print(f"✅ Found {en_lines} English sentences")
    print(f"✅ Found {vi_lines} Vietnamese sentences")
    
    if en_lines != vi_lines:
        print(f"⚠️  Warning: Mismatched line counts (EN: {en_lines}, VI: {vi_lines})")
        print("   Using minimum length for alignment")
    
    # Check if we can read the files
    try:
        with open(en_file, 'r', encoding='utf-8') as f:
            sample_en = f.readline().strip()
        with open(vi_file, 'r', encoding='utf-8') as f:
            sample_vi = f.readline().strip()
        
        print(f"✅ Sample EN: {sample_en[:50]}...")
        print(f"✅ Sample VI: {sample_vi[:50]}...")
        
    except Exception as e:
        print(f"❌ Error reading files: {e}")
        return False
    
    print("✅ Data files are ready for processing!")
    return True

def show_usage():
    """Show usage instructions."""
    print("\n" + "="*50)
    print("How to use your data with the MoE system:")
    print("="*50)
    print("1. Make sure your files are in the correct location:")
    print(f"   - {os.path.join(PROJECT_ROOT, 'data', 'train.en.txt')}")
    print(f"   - {os.path.join(PROJECT_ROOT, 'data', 'train.vi.txt')}")
    print("")
    print("2. Run the data preparation script:")
    print("   python prepare_data.py")
    print("")
    print("3. This will create three expert datasets:")
    print("   - Medical domain expert (EN->VI)")
    print("   - General EN->VI expert")
    print("   - General VI->EN expert")
    print("")
    print("4. The script will automatically:")
    print("   - Detect medical texts using keywords")
    print("   - Split remaining data for bidirectional translation")
    print("   - Create JSONL files for each expert")
    print("   - Combine all data into a single training file")
    print("")
    print("5. Start training with:")
    print("   python train_moe.py")
    print("="*50)

if __name__ == "__main__":
    if test_data_preparation():
        show_usage()
    else:
        print("\n❌ Please fix the data issues before proceeding.")
        sys.exit(1)
