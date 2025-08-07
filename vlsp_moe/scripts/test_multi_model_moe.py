import torch
import os
import sys
sys.path.append(os.path.dirname(__file__))

from train_moe import create_model, MultiModelMoE
from config import Config

def test_multi_model_moe():
    """Test the multi-model MoE setup with medical reasoning and translation."""
    
    # Load configuration
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    config = Config(os.path.join(PROJECT_ROOT, "vlsp_moe", "configs", "moe_config.yaml"))
    
    print("Creating multi-model MoE...")
    model, tokenizer = create_model(config)
    
    # Test medical reasoning
    print("\n=== Testing Medical Reasoning ===")
    medical_prompt = "A 45-year-old male presents with chest pain and shortness of breath. List possible diagnoses and explain the reasoning."
    
    messages = [
        {"role": "system", "content": "You are a clinical reasoning assistant trained on the Med Reason Dataset."},
        {"role": "user", "content": medical_prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"Medical Query: {medical_prompt}")
        print(f"Response: {response}")
    
    # Test translation
    print("\n=== Testing Translation ===")
    translation_prompt = "Translate the following English sentence to Vietnamese: The patient has a fever."
    
    model_inputs = tokenizer([translation_prompt], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=128,
            temperature=0.7,
            do_sample=True
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"Translation Query: {translation_prompt}")
        print(f"Response: {response}")
    
    print("\n=== Multi-Model MoE Test Completed ===")

if __name__ == "__main__":
    test_multi_model_moe() 