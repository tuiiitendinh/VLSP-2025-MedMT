import torch
import os
import sys
sys.path.append(os.path.dirname(__file__))

from train_moe import create_model, MultiModelMoE
from config import Config

def run_inference():
    """Run inference with the multi-model MoE setup."""
    
    # Load configuration
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    config = Config(os.path.join(PROJECT_ROOT, "vlsp_moe", "configs", "moe_config.yaml"))
    
    print("Loading multi-model MoE...")
    model, tokenizer = create_model(config)
    
    # Test cases
    test_cases = [
        {
            "type": "Medical Reasoning",
            "prompt": "A 45-year-old male presents with chest pain and shortness of breath. List possible diagnoses and explain the reasoning.",
            "system": "You are a clinical reasoning assistant trained on the Med Reason Dataset."
        },
        {
            "type": "Medical Translation EN->VI",
            "prompt": "Translate the following English medical text to Vietnamese: The patient has a fever and cough.",
            "system": "You are a medical translation assistant."
        },
        {
            "type": "General Translation EN->VI",
            "prompt": "Translate the following English sentence to Vietnamese: Hello, how are you today?",
            "system": "You are a translation assistant."
        },
        {
            "type": "General Translation VI->EN",
            "prompt": "Translate the following Vietnamese sentence to English: Xin chào, bạn khỏe không?",
            "system": "You are a translation assistant."
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n=== Test Case {i}: {test_case['type']} ===")
        
        messages = [
            {"role": "system", "content": test_case["system"]},
            {"role": "user", "content": test_case["prompt"]}
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
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Extract only the generated part
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            print(f"Input: {test_case['prompt']}")
            print(f"Response: {response}")
            print("-" * 80)

if __name__ == "__main__":
    run_inference() 