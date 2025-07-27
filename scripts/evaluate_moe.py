import torch
import json
from config import Config
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
from typing import Dict
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class MoEInference:
    """Inference class for MoE model."""
    
    def __init__(self, model_path: str, config_path: str):
        self.config = Config(config_path)
        self.model_path = model_path
        
        # Load base model and tokenizer using transformers directly
        base_model = self.config.model["model_id_or_path"]
        logger.info(f"Loading base model: {base_model}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load PEFT adapter
        logger.info(f"Loading PEFT adapter from: {model_path}")
        self.model = PeftModel.from_pretrained(model, model_path).eval()
        
        # Remove template loading since we're not using swift.llm anymore
        # self.template = get_template(self.config.model["template"], self.tokenizer)
        
        self.expert_mapping = {
            'medical': 0,
            'en_vi': 1,
            'vi_en': 2
        }
        
        logger.info("MoE model loaded successfully")
    
    def _format_prompt(self, text: str, task_desc: str) -> str:
        """Format prompt for the model."""
        # Simple prompt formatting (adjust based on your model's expected format)
        return f"{task_desc}: {text}"
    
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
        
        # Format prompt
        prompt = self._format_prompt(text, task_desc)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate translation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_beams=4,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract translation (remove original prompt)
        if prompt in generated_text:
            translation = generated_text.replace(prompt, "").strip()
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
    
    # Set default path to outputs/moe_model
    default_model_path = os.path.join(PROJECT_ROOT, "outputs", "moe_model")
    
    parser.add_argument("--model_path", type=str, 
                       default=default_model_path,
                       help="Path to the fine-tuned MoE model (default: outputs/moe_model)")
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
    
    # Print the model path being used for clarity
    print(f"Using model path: {args.model_path}")
    
    # Check if the model path exists
    if not os.path.exists(args.model_path):
        print(f"⚠️ Warning: Model path does not exist: {args.model_path}")
        print("Make sure you have trained and saved a model to this location.")
        return
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
