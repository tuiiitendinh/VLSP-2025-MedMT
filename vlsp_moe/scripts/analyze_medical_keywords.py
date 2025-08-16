import torch
import os
import sys
sys.path.append(os.path.dirname(__file__))

from train_moe import create_model, MultiModelMoE
from config import Config
import json
from collections import Counter

def analyze_medical_keywords():
    """Analyze and display the medical keywords extracted from data and tokenizer."""
    
    # Load configuration
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    config = Config(os.path.join(PROJECT_ROOT, "vlsp_moe", "configs", "moe_config.yaml"))
    
    print("Creating multi-model MoE to extract medical keywords...")
    model, tokenizer = create_model(config)
    
    # Get the extracted medical keywords
    medical_keywords = model.get_medical_keywords()
    
    print(f"\n=== Medical Keywords Analysis ===")
    print(f"Total medical keywords extracted: {len(medical_keywords)}")
    
    # Categorize keywords
    english_medical = []
    vietnamese_medical = []
    other_terms = []
    
    for keyword in medical_keywords:
        # Simple heuristic to categorize
        if any(char in keyword for char in ['ă', 'â', 'ê', 'ô', 'ơ', 'ư', 'đ', 'ệ', 'ế', 'ề', 'ể', 'ễ']):
            vietnamese_medical.append(keyword)
        elif any(term in keyword.lower() for term in ['patient', 'disease', 'treatment', 'diagnosis', 'symptom', 
                                                     'medicine', 'doctor', 'hospital', 'medical', 'clinical', 
                                                     'therapy', 'surgery', 'drug', 'medication', 'health', 
                                                     'illness', 'condition', 'infection', 'virus', 'bacteria', 
                                                     'cancer', 'tumor', 'blood', 'heart', 'lung', 'brain', 
                                                     'liver', 'kidney', 'bone', 'muscle', 'skin', 'eye']):
            english_medical.append(keyword)
        else:
            other_terms.append(keyword)
    
    print(f"\nEnglish medical terms: {len(english_medical)}")
    print("Top 20 English medical terms:")
    for term in sorted(english_medical)[:20]:
        print(f"  - {term}")
    
    print(f"\nVietnamese medical terms: {len(vietnamese_medical)}")
    print("Top 20 Vietnamese medical terms:")
    for term in sorted(vietnamese_medical)[:20]:
        print(f"  - {term}")
    
    print(f"\nOther extracted terms: {len(other_terms)}")
    print("Top 20 other terms:")
    for term in sorted(other_terms)[:20]:
        print(f"  - {term}")
    
    # Analyze training data distribution
    print(f"\n=== Training Data Analysis ===")
    train_file = os.path.join(PROJECT_ROOT, "data", "processed", "train.jsonl")
    
    if os.path.exists(train_file):
        expert_counts = Counter()
        medical_samples = []
        
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    expert = data.get('expert', 'unknown')
                    expert_counts[expert] += 1
                    
                    if expert == 'medical':
                        medical_samples.append(data)
                except json.JSONDecodeError:
                    continue
        
        print("Expert distribution in training data:")
        for expert, count in expert_counts.most_common():
            print(f"  - {expert}: {count} samples")
        
        print(f"\nMedical domain samples: {len(medical_samples)}")
        if medical_samples:
            print("Sample medical queries:")
            for i, sample in enumerate(medical_samples[:5]):
                messages = sample.get('messages', [])
                if messages:
                    user_msg = next((msg.get('content', '') for msg in messages if msg.get('role') == 'user'), '')
                    print(f"  {i+1}. {user_msg[:100]}...")
    
    # Test keyword detection
    print(f"\n=== Keyword Detection Test ===")
    test_queries = [
        "A patient has chest pain",
        "Translate this sentence to Vietnamese",
        "Bệnh nhân bị sốt cao",
        "Hello, how are you?",
        "The doctor prescribed medication",
        "Bác sĩ khám bệnh cho bệnh nhân",
        "What is the weather today?",
        "Treatment for diabetes includes"
    ]
    
    for query in test_queries:
        is_medical = model.is_medical_query(query)
        print(f"Query: '{query}' -> Medical: {is_medical}")

if __name__ == "__main__":
    analyze_medical_keywords() 