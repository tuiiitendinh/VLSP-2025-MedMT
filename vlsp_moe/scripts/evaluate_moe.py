import torch
import json
import os
from config import Config
from swift.llm import get_model_tokenizer, get_template
from peft import PeftModel
import argparse
from typing import Dict
import logging

# Set environment variable to force left padding for all tokenizers
os.environ["TOKENIZERS_PADDING_SIDE"] = "left"

# Import transformers after setting environment variable
import transformers

# Configure transformers logging to reduce noise
transformers.logging.set_verbosity_error()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def configure_tokenizer_for_generation(tokenizer):
    """
    Properly configure tokenizer for decoder-only generation.
    This ensures correct padding behavior and eliminates warnings.
    """
    if tokenizer is None:
        return None
    
    # Set padding side to left for decoder-only models
    tokenizer.padding_side = 'left'
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Set pad token id to avoid conflicts
    if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info(f"Configured tokenizer: padding_side='{tokenizer.padding_side}', pad_token='{tokenizer.pad_token}'")
    return tokenizer

class MoEInference:
    """Inference class for MoE model."""
    
    def __init__(self, model_path: str, config_path: str):
        self.config = Config(config_path)
        self.model_path = model_path
        self.model, self.tokenizer = get_model_tokenizer(
            self.config.model["model_id_or_path"]
        )
        # Configure tokenizer properly for decoder-only generation
        self.tokenizer = configure_tokenizer_for_generation(self.tokenizer)
        
        self.model = PeftModel.from_pretrained(self.model, model_path)
        self.model.eval()
        self.template = get_template(self.config.model["template"], self.tokenizer)
        self.expert_mapping = {
            'medical': 0,
            'en_vi': 1,
            'vi_en': 2
        }
        
        logger.info("MoE model loaded successfully")
    
    def translate(self, text: str, source_lang: str, target_lang: str, 
                  domain: str = None, max_length: int = 512) -> str:
        """
        Translate text using the MoE model.
        
        Args:
            text: Input text to translate
            source_lang: Source language code ('en' or 'vi')
            target_lang: Target language code ('en' or 'vi')
            domain: Domain type ('medical' or None for general)
            max_length: Maximum output length
            
        Returns:
            Translated text
        """
        if domain == "medical":
            expert_type = "medical"
            task_desc = "Translate the following English medical text to Vietnamese"
        elif source_lang == "en" and target_lang == "vi":
            expert_type = "en_vi"
            task_desc = "Translate the following English sentence to Vietnamese"
        elif source_lang == "vi" and target_lang == "en":
            expert_type = "vi_en"
            task_desc = "Translate the following Vietnamese sentence to English"
        else:
            expert_type = "en_vi"  
            task_desc = "Translate the following sentence"
        
        messages = [
            {
                "role": "user",
                "content": f"{task_desc}: {text}"
            }
        ]
        
        # Encode input
        inputs = self.template.encode(messages)
        input_ids = torch.tensor([inputs['input_ids']])
        attention_mask = torch.tensor([inputs['attention_mask']])
        
        # Generate translation with optimized parameters for speed
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=min(128, max_length//2),  # Reduced for faster inference
                num_beams=1,  # Greedy decoding for speed
                do_sample=False,  # Deterministic for consistent results
                use_cache=True,  # Enable KV cache
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in generated_text:
            translation = generated_text.split("assistant")[-1].strip()
        else:
            translation = generated_text.strip()
        
        return translation
    
    def evaluate_on_testset(self, test_file: str) -> Dict:
        """Evaluate model on test set."""
        results = {
            'total_samples': 0,
            'expert_usage': {'medical': 0, 'en_vi': 0, 'vi_en': 0},
            'translations': []
        }
        
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                expert_type = item.get('expert', 'general')
                
                user_message = item['messages'][0]['content']
                target_text = item['messages'][1]['content']
                
                if ":" in user_message:
                    source_text = user_message.split(":", 1)[1].strip()
                else:
                    source_text = user_message.strip()
                
                if expert_type == "medical":
                    source_lang, target_lang = "en", "vi"
                    domain = "medical"
                elif expert_type == "en_vi":
                    source_lang, target_lang = "en", "vi"
                    domain = None
                elif expert_type == "vi_en":
                    source_lang, target_lang = "vi", "en"
                    domain = None
                else:
                    source_lang, target_lang = "en", "vi"
                    domain = None
                
                translation = self.translate(
                    source_text, source_lang, target_lang, domain
                )
                
                results['translations'].append({
                    'source': source_text,
                    'target': target_text,
                    'translation': translation,
                    'expert': expert_type
                })
                
                results['expert_usage'][expert_type] += 1
                results['total_samples'] += 1
        
        return results

def main():
    parser = argparse.ArgumentParser(description="MoE Model Inference")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the fine-tuned MoE model")
    parser.add_argument("--config_path", type=str, 
                       default=os.path.join(PROJECT_ROOT, "configs", "moe_config.yaml"),
                       help="Path to the configuration file")
    parser.add_argument("--test_file", type=str, 
                       default=os.path.join(PROJECT_ROOT, "data", "processed", "val.jsonl"),
                       help="Path to the test file")
    parser.add_argument("--output_file", type=str, 
                       default=os.path.join(PROJECT_ROOT, "evaluation_results.json"),
                       help="Path to save evaluation results")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = MoEInference(args.model_path, args.config_path)
    
    if args.interactive:
        # Interactive mode
        print("MoE Translation Model - Interactive Mode")
        print("Type 'quit' to exit")
        
        while True:
            text = input("\nEnter text to translate: ")
            if text.lower() == 'quit':
                break
            
            source_lang = input("Source language (en/vi): ").lower()
            target_lang = input("Target language (en/vi): ").lower()
            domain = input("Domain (medical/general): ").lower()
            domain = domain if domain == "medical" else None
            
            translation = inference.translate(text, source_lang, target_lang, domain)
            print(f"\nTranslation: {translation}")
    
    else:
        # Evaluation mode
        logger.info("Starting evaluation...")
        results = inference.evaluate_on_testset(args.test_file)
        
        # Save results
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Print summary
        print(f"\nEvaluation Results:")
        print(f"Total samples: {results['total_samples']}")
        print(f"Expert usage:")
        for expert, count in results['expert_usage'].items():
            percentage = (count / results['total_samples']) * 100
            print(f"  {expert}: {count} ({percentage:.1f}%)")
        
        print(f"\nResults saved to: {args.output_file}")

if __name__ == "__main__":
    main()
