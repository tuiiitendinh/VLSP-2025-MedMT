import torch
import os
import logging
from swift.llm import get_model_tokenizer, get_template
from peft import PeftModel
from config import Config

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

# Define paths
MODEL_PATH = "/home/users/sutd/1010042/VLSP-2025-MedMT/checkpoint-33750/"
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "moe_config.yaml")
INPUT_FILE = "/home/users/sutd/1010042/VLSP-2025-MedMT/test_data/test.en.txt"  # File containing input sentences
OUTPUT_FILE = "/home/users/sutd/1010042/VLSP-2025-MedMT/test_data_output/test.vi.txt"  # File to save the inference results


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


# Load model and tokenizer
def load_model_and_tokenizer(model_path, config_path):
    logger.info("Loading model and tokenizer...")
    
    # Load configuration
    config = Config(config_path)
    
    # Load base model and tokenizer using Swift
    model, tokenizer = get_model_tokenizer(
        config.model["model_id_or_path"]
    )
    
    # Configure tokenizer properly for decoder-only generation
    tokenizer = configure_tokenizer_for_generation(tokenizer)
    
    # Load fine-tuned weights using PEFT
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    
    # Get template for formatting
    template = get_template(config.model["template"], tokenizer)
    
    logger.info("Model and tokenizer loaded successfully")
    return model, tokenizer, template

# Perform inference
def infer(model, tokenizer, template, input_file, output_file):
    logger.info("Starting inference...")
    
    # Read input sentences
    with open(input_file, "r", encoding="utf-8") as f:
        input_sentences = [line.strip() for line in f if line.strip()]

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Prepare output file
    with open(output_file, "w", encoding="utf-8") as outf:
        for i, sentence in enumerate(input_sentences):
            logger.info(f"Processing sentence {i+1}/{len(input_sentences)}")
            
            # Format input using template (following evaluate_moe.py pattern)
            messages = [
                {
                    "role": "user", 
                    "content": f"Translate the following English sentence to Vietnamese: {sentence}"
                }
            ]
            
            # Encode input using template
            inputs = template.encode(messages)
            input_ids = torch.tensor([inputs['input_ids']])
            attention_mask = torch.tensor([inputs['attention_mask']])
            
            # Generate translation with optimized parameters
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=256,
                    num_beams=1,  # Greedy decoding for speed
                    do_sample=False,  # Deterministic for consistent results
                    use_cache=True,  # Enable KV cache
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract translation (following evaluate_moe.py pattern)
            if "assistant" in generated_text:
                translation = generated_text.split("assistant")[-1].strip()
            else:
                translation = generated_text.strip()
            
            outf.write(translation + "\n")

    logger.info(f"Inference completed. Results saved to {output_file}")

if __name__ == "__main__":
    # Ensure paths are updated before running
    assert os.path.exists(MODEL_PATH), f"Model path does not exist: {MODEL_PATH}"
    assert os.path.exists(CONFIG_PATH), f"Config path does not exist: {CONFIG_PATH}"
    assert os.path.exists(INPUT_FILE), f"Input file does not exist: {INPUT_FILE}"

    model, tokenizer, template = load_model_and_tokenizer(MODEL_PATH, CONFIG_PATH)
    infer(model, tokenizer, template, INPUT_FILE, OUTPUT_FILE)
