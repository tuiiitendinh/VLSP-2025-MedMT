import os
import torch
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

# Define paths
MODEL_PATH = "<path_to_trained_model>"
TOKENIZER_PATH = "<path_to_tokenizer>"
INPUT_FILE = "<path_to_input_file>"  # File containing input sentences
OUTPUT_FILE = "<path_to_output_file>"  # File to save the inference results

# Load model and tokenizer
def load_model_and_tokenizer(model_path, tokenizer_path):
    print("Loading model and tokenizer...")
    model = FastLanguageModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

# Perform inference
def infer(model, tokenizer, input_file, output_file):
    print("Starting inference...")
    
    # Read input sentences
    with open(input_file, "r", encoding="utf-8") as f:
        input_sentences = [line.strip() for line in f if line.strip()]

    # Prepare output file
    with open(output_file, "w", encoding="utf-8") as outf:
        for sentence in input_sentences:
            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=256, num_beams=1)
            decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            outf.write(decoded_output + "\n")

    print(f"Inference completed. Results saved to {output_file}")

if __name__ == "__main__":
    # Ensure paths are updated before running
    assert os.path.exists(MODEL_PATH), "Model path does not exist. Update MODEL_PATH."
    assert os.path.exists(TOKENIZER_PATH), "Tokenizer path does not exist. Update TOKENIZER_PATH."
    assert os.path.exists(INPUT_FILE), "Input file does not exist. Update INPUT_FILE."

    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, TOKENIZER_PATH)
    infer(model, tokenizer, INPUT_FILE, OUTPUT_FILE)
