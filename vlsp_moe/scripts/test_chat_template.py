#!/usr/bin/env python3
"""
Test script to verify chat template formatting consistency between training and inference.
"""

import os
import sys
import torch

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from config import Config
from infer import create_model, format_input_for_inference

def test_chat_template():
    """Test that inference chat template matches training format."""
    
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    config_path = os.path.join(PROJECT_ROOT, "vlsp_moe", "configs", "moe_config.yaml")
    
    print("Testing chat template formatting...")
    print("=" * 60)
    
    # Load model and tokenizer (just for tokenizer, no need to run inference)
    try:
        config = Config(config_path)
        model, tokenizer = create_model(config)
        print(f"✓ Model and tokenizer loaded successfully")
        print(f"✓ EOS token: {repr(tokenizer.eos_token)}")
        print(f"✓ EOS token ID: {tokenizer.eos_token_id}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False
    
    # Test various prompts
    test_prompts = [
        "Translate the following English sentence to Vietnamese: Hello, how are you?",
        "Translate the following Vietnamese sentence to English: Xin chào, bạn khỏe không?",
        "Translate the following English medical text to Vietnamese: The patient has a fever and cough.",
    ]
    
    print("\nTesting chat template formatting:")
    print("-" * 60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}:")
        print(f"Original prompt: {prompt}")
        
        # Format using our training-consistent method
        formatted = format_input_for_inference(prompt, tokenizer)
        print(f"Formatted input: {repr(formatted)}")
        
        # Show tokenization
        tokens = tokenizer.encode(formatted)
        print(f"Token count: {len(tokens)}")
        
        # Verify EOS token is present
        if tokenizer.eos_token_id in tokens:
            eos_position = tokens.index(tokenizer.eos_token_id)
            print(f"✓ EOS token found at position {eos_position}")
        else:
            print("✗ EOS token not found in tokenized input!")
            return False
    
    # Compare with training format
    print("\n" + "=" * 60)
    print("Training format comparison:")
    print("-" * 60)
    
    # Simulate training format: {user_content}<eos_token>{assistant_content}<eos_token>
    user_content = test_prompts[0]
    assistant_content = "Xin chào, bạn khỏe không?"
    training_format = f"{user_content}{tokenizer.eos_token}{assistant_content}{tokenizer.eos_token}"
    
    print(f"Training format example:")
    print(f"User: {user_content}")
    print(f"Assistant: {assistant_content}")
    print(f"Training text: {repr(training_format)}")
    
    # Show inference format
    inference_format = format_input_for_inference(user_content, tokenizer)
    print(f"\nInference format:")
    print(f"Input text: {repr(inference_format)}")
    print(f"Expected generation: {repr(assistant_content + tokenizer.eos_token)}")
    
    # Verify they match up to the first EOS token
    training_prefix = training_format.split(tokenizer.eos_token)[0] + tokenizer.eos_token
    if training_prefix == inference_format:
        print("✓ Inference format matches training format prefix!")
    else:
        print("✗ Format mismatch!")
        print(f"  Training prefix: {repr(training_prefix)}")
        print(f"  Inference format: {repr(inference_format)}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! Chat template is consistent with training format.")
    return True

if __name__ == "__main__":
    success = test_chat_template()
    exit(0 if success else 1)
