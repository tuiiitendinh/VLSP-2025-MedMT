import os
import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

class MoEInference:
    def __init__(self, model_path: str, config_path: str):
        """
        Initialize the MoEInference class.

        :param model_path: Path to the trained model directory.
        :param config_path: Path to the configuration file.
        """
        self.model_path = model_path
        self.config_path = config_path
        self.model, self.tokenizer = self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """
        Load the model and tokenizer from the specified paths.

        :return: Tuple of (model, tokenizer).
        """
        print("Loading model and tokenizer...")
        model = FastLanguageModel.from_pretrained(self.model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return model, tokenizer

    def translate(self, text: str, source_lang: str, target_lang: str, domain: str = None) -> str:
        """
        Translate the given text using the MoE model.

        :param text: Input text to translate.
        :param source_lang: Source language (e.g., 'en').
        :param target_lang: Target language (e.g., 'vi').
        :param domain: Optional domain for translation (e.g., 'medical').
        :return: Translated text.
        """
        # Add domain-specific tokens if provided
        if domain:
            text = f"<{domain}> {text}"

        # Tokenize input text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

        # Generate translation
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=2048,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # Decode the output
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text

if __name__ == "__main__":
    # Example usage
    MODEL_PATH = "<path_to_trained_model>"
    CONFIG_PATH = "<path_to_config_file>"

    # Ensure paths are updated before running
    assert os.path.exists(MODEL_PATH), "Model path does not exist. Update MODEL_PATH."
    assert os.path.exists(CONFIG_PATH), "Config path does not exist. Update CONFIG_PATH."

    inference = MoEInference(MODEL_PATH, CONFIG_PATH)
    result = inference.translate(
        text="Hello world",
        source_lang="en",
        target_lang="vi",
        domain=None  # or "medical" for medical domain
    )
    print("Translated text:", result)
