import torch
import os
import sys
import json
import pickle
sys.path.append(os.path.dirname(__file__))

from train_moe import create_model, MultiModelMoE
from config import Config

def extract_and_save_keywords():
    """Extract medical keywords from data and save them for reuse."""
    
    # Load configuration
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    config = Config(os.path.join(PROJECT_ROOT, "vlsp_moe", "configs", "moe_config.yaml"))
    
    print("Creating multi-model MoE to extract medical keywords...")
    model, tokenizer = create_model(config)
    
    # Get the extracted medical keywords
    medical_keywords = model.get_medical_keywords()
    
    # Save keywords to file
    keywords_file = os.path.join(PROJECT_ROOT, "vlsp_moe", "medical_keywords.json")
    with open(keywords_file, 'w', encoding='utf-8') as f:
        json.dump({
            'medical_keywords': medical_keywords,
            'extraction_info': {
                'total_keywords': len(medical_keywords),
                'source': 'data_driven_extraction',
                'timestamp': str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
            }
        }, f, ensure_ascii=False, indent=2)
    
    print(f"Medical keywords saved to: {keywords_file}")
    print(f"Total keywords extracted: {len(medical_keywords)}")
    
    # Display sample keywords
    print("\nSample medical keywords:")
    for i, keyword in enumerate(sorted(medical_keywords)[:20]):
        print(f"  {i+1:2d}. {keyword}")
    
    if len(medical_keywords) > 20:
        print(f"  ... and {len(medical_keywords) - 20} more keywords")
    
    return medical_keywords

def load_keywords():
    """Load medical keywords from saved file."""
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    keywords_file = os.path.join(PROJECT_ROOT, "vlsp_moe", "medical_keywords.json")
    
    if os.path.exists(keywords_file):
        with open(keywords_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data['medical_keywords']
    else:
        print(f"Keywords file not found: {keywords_file}")
        return None

if __name__ == "__main__":
    extract_and_save_keywords() 