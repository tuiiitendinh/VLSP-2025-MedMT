import torch
import os
import sys
import json
import re
from collections import Counter
sys.path.append(os.path.dirname(__file__))

from train_moe import create_model, MultiModelMoE
from config import Config

def validate_data_driven_extraction():
    """Validate and analyze the data-driven medical keyword extraction process."""
    
    # Load configuration
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    config = Config(os.path.join(PROJECT_ROOT, "vlsp_moe", "configs", "moe_config.yaml"))
    
    print("Creating multi-model MoE to analyze data-driven extraction...")
    model, tokenizer = create_model(config)
    
    # Get the extracted medical keywords
    medical_keywords = model.get_medical_keywords()
    
    print(f"\n=== Data-Driven Medical Keyword Analysis ===")
    print(f"Total medical keywords extracted: {len(medical_keywords)}")
    
    # Analyze the extraction process
    train_file = os.path.join(PROJECT_ROOT, "data", "processed", "train.jsonl")
    
    if os.path.exists(train_file):
        print(f"\n=== Training Data Analysis ===")
        
        medical_terms = []
        general_terms = []
        medical_contexts = []
        general_contexts = []
        
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    expert = data.get('expert', '')
                    messages = data.get('messages', [])
                    
                    for msg in messages:
                        content = msg.get('content', '').lower()
                        words = re.findall(r'\b\w+\b', content)
                        
                        if expert == 'medical':
                            medical_terms.extend(words)
                            medical_contexts.append(content)
                        else:
                            general_terms.extend(words)
                            general_contexts.append(content)
                            
                except json.JSONDecodeError:
                    continue
        
        # Statistical analysis
        medical_counts = Counter(medical_terms)
        general_counts = Counter(general_terms)
        
        print(f"Medical domain samples: {len(medical_contexts)}")
        print(f"General domain samples: {len(general_contexts)}")
        print(f"Medical terms: {len(medical_terms)}")
        print(f"General terms: {len(general_terms)}")
        
        # Show top medical terms by frequency
        print(f"\n=== Top Medical Terms by Frequency ===")
        for term, count in medical_counts.most_common(20):
            general_freq = general_counts.get(term, 0)
            ratio = count / len(medical_terms) if len(medical_terms) > 0 else 0
            general_ratio = general_freq / len(general_terms) if len(general_terms) > 0 else 0
            print(f"  {term}: {count} (medical), {general_freq} (general), ratio: {ratio:.4f} vs {general_ratio:.4f}")
        
        # Show extracted keywords that are actually in medical data
        print(f"\n=== Extracted Keywords Validation ===")
        extracted_in_medical = []
        extracted_in_general = []
        not_found = []
        
        for keyword in medical_keywords:
            if keyword.lower() in medical_counts:
                extracted_in_medical.append((keyword, medical_counts[keyword.lower()]))
            elif keyword.lower() in general_counts:
                extracted_in_general.append((keyword, general_counts[keyword.lower()]))
            else:
                not_found.append(keyword)
        
        print(f"Keywords found in medical data: {len(extracted_in_medical)}")
        print(f"Keywords found in general data: {len(extracted_in_general)}")
        print(f"Keywords not found in data: {len(not_found)}")
        
        # Show top extracted keywords by medical frequency
        print(f"\n=== Top Extracted Keywords by Medical Frequency ===")
        for keyword, freq in sorted(extracted_in_medical, key=lambda x: x[1], reverse=True)[:15]:
            general_freq = general_counts.get(keyword.lower(), 0)
            print(f"  {keyword}: {freq} (medical), {general_freq} (general)")
        
        # Show potentially problematic keywords
        print(f"\n=== Keywords Found in General Data (Potential False Positives) ===")
        for keyword, freq in sorted(extracted_in_general, key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {keyword}: {freq} (general)")
    
    # Test the extraction on sample queries
    print(f"\n=== Extraction Quality Test ===")
    test_queries = [
        "A patient has chest pain and shortness of breath",
        "Translate this sentence to Vietnamese",
        "Bệnh nhân bị sốt cao và ho nhiều",
        "Hello, how are you today?",
        "The doctor prescribed medication for diabetes",
        "Bác sĩ khám bệnh cho bệnh nhân",
        "What is the weather like today?",
        "Treatment for heart disease includes",
        "I want to learn Vietnamese",
        "Bệnh viện đang điều trị cho bệnh nhân"
    ]
    
    for query in test_queries:
        is_medical = model.is_medical_query(query)
        medical_terms_found = [kw for kw in medical_keywords if kw.lower() in query.lower()]
        print(f"Query: '{query}'")
        print(f"  Medical: {is_medical}")
        print(f"  Medical terms found: {medical_terms_found}")
        print()

if __name__ == "__main__":
    validate_data_driven_extraction() 