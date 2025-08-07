import torch
import torch.nn as nn
from config import Config
import json
import logging
import os
from sacrebleu import corpus_bleu
import numpy as np

# Import the 'datasets' library and use SFTTrainer
import datasets
from unsloth.trainer import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling, AutoModelForCausalLM, AutoTokenizer
import unsloth
from unsloth import FastLanguageModel
import wandb

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Custom Trainer ---
class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processing_class = self.tokenizer  # Define processing_class using tokenizer

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | list[int]],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        with torch.no_grad():
            pad_token_id = self.processing_class.pad_token_id
            eos_token_id = self.processing_class.eos_token_id

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=256,
                num_beams=1,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
        return (None, generated_ids, labels)


# --- Multi-Model MoE Architecture ---
class MultiModelMoE(nn.Module):
    def __init__(self, medical_model, translation_model, tokenizer, config):
        super().__init__()
        self.medical_model = medical_model
        self.translation_model = translation_model
        self.tokenizer = tokenizer
        self.config = config
        
        # Extract medical keywords from data and tokenizer
        self.medical_keywords = self._extract_medical_keywords()
    
    def _extract_medical_keywords(self):
        """Extract medical keywords entirely from training data and tokenizer."""
        import json
        import re
        from collections import Counter
        
        # First, try to load keywords from saved file
        keywords_file = os.path.join(PROJECT_ROOT, "vlsp_moe", "medical_keywords.json")
        if os.path.exists(keywords_file):
            try:
                with open(keywords_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    medical_keywords = data['medical_keywords']
                    print(f"Loaded {len(medical_keywords)} medical keywords from saved file")
                    return medical_keywords
            except Exception as e:
                print(f"Warning: Could not load keywords from file: {e}")
                print("Will extract keywords from data instead")
        
        medical_keywords = set()
        
        # Extract medical terms from training data
        try:
            # Load training data to extract domain-specific terms
            train_file = os.path.join(PROJECT_ROOT, "data", "processed", "train.jsonl")
            if os.path.exists(train_file):
                print("Extracting medical keywords from training data...")
                
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
                                # Extract words using regex
                                words = re.findall(r'\b\w+\b', content)
                                
                                if expert == 'medical':
                                    medical_terms.extend(words)
                                    medical_contexts.append(content)
                                else:
                                    general_terms.extend(words)
                                    general_contexts.append(content)
                                    
                        except json.JSONDecodeError:
                            continue
                
                # Calculate term frequencies and statistical significance
                if medical_terms and general_terms:
                    medical_counts = Counter(medical_terms)
                    general_counts = Counter(general_terms)
                    
                    # Calculate TF-IDF-like scores for medical relevance
                    total_medical_terms = len(medical_terms)
                    total_general_terms = len(general_terms)
                    
                    for term, medical_freq in medical_counts.most_common(200):
                        if len(term) > 2:  # Minimum length
                            general_freq = general_counts.get(term, 0)
                            
                            # Calculate medical relevance score
                            medical_ratio = medical_freq / total_medical_terms if total_medical_terms > 0 else 0
                            general_ratio = general_freq / total_general_terms if total_general_terms > 0 else 0
                            
                            # Add term if it's more frequent in medical context
                            if medical_freq >= 2 and (medical_ratio > general_ratio or medical_freq >= 5):
                                medical_keywords.add(term)
                    
                    # Extract n-grams and phrases from medical contexts
                    medical_phrases = self._extract_medical_phrases(medical_contexts)
                    medical_keywords.update(medical_phrases)
                    
                    print(f"Extracted {len(medical_keywords)} medical keywords from training data")
                else:
                    print("No medical domain samples found in training data")
        
        except Exception as e:
            print(f"Warning: Could not extract medical keywords from data: {e}")
        
        # Extract medical-related tokens from tokenizer using data-driven patterns
        try:
            # Look for medical-related tokens in the tokenizer vocabulary
            vocab = self.tokenizer.get_vocab()
            medical_tokens = []
            
            # Use the medical keywords already extracted from data to identify medical tokens
            # This ensures we're using data-driven patterns, not hardcoded ones
            if medical_keywords:
                # Create a set of medical keyword stems for matching
                medical_stems = set()
                for keyword in medical_keywords:
                    # Add the keyword itself and common variations
                    medical_stems.add(keyword.lower())
                    medical_stems.add(keyword.lower().replace(' ', ''))
                    # Add common prefixes/suffixes
                    if len(keyword) > 3:
                        medical_stems.add(keyword.lower()[:3])
                        medical_stems.add(keyword.lower()[-3:])
                
                # Scan vocabulary for tokens that match medical patterns from data
                for token in vocab.keys():
                    token_lower = token.lower()
                    
                    # Check if token contains any medical stems from our data
                    if any(stem in token_lower for stem in medical_stems):
                        medical_tokens.append(token)
                    
                    # Also check for exact matches with medical keywords
                    elif any(keyword.lower() in token_lower for keyword in medical_keywords):
                        medical_tokens.append(token)
                
                # Add medical tokens to keywords
                medical_keywords.update(medical_tokens)
                print(f"Added {len(medical_tokens)} medical tokens from tokenizer using data-driven patterns")
            else:
                print("No medical keywords available for tokenizer analysis")
            
        except Exception as e:
            print(f"Warning: Could not extract medical tokens from tokenizer: {e}")
        
        # If no medical keywords found, use a minimal fallback set
        if not medical_keywords:
            print("Warning: No medical keywords extracted. Using minimal fallback set.")
            # Only use the most basic medical terms as fallback
            medical_keywords = {'patient', 'doctor', 'hospital', 'bệnh', 'thuốc', 'bác sĩ'}
        
        return list(medical_keywords)
    
    def _extract_medical_phrases(self, medical_contexts):
        """Extract medical phrases and n-grams from medical contexts."""
        import re
        from collections import Counter
        
        phrases = set()
        
        for context in medical_contexts:
            # Extract 2-3 word phrases that might be medical terms
            words = re.findall(r'\b\w+\b', context)
            
            # Extract bigrams and trigrams
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                if len(bigram) > 5:  # Minimum phrase length
                    phrases.add(bigram)
            
            for i in range(len(words) - 2):
                trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                if len(trigram) > 8:  # Minimum phrase length
                    phrases.add(trigram)
        
        # Filter phrases by frequency and relevance
        phrase_counts = Counter(phrases)
        relevant_phrases = set()
        
        for phrase, count in phrase_counts.most_common(50):
            if count >= 2:  # Minimum frequency
                relevant_phrases.add(phrase)
        
        return relevant_phrases
    
    def is_medical_query(self, text):
        """Determine if the input is medical-related using extracted keywords."""
        text_lower = text.lower()
        
        # Check for medical keywords
        keyword_match = any(keyword.lower() in text_lower for keyword in self.medical_keywords)
        
        # Additional data-driven heuristics
        # Count medical terms in the text
        medical_term_count = sum(1 for keyword in self.medical_keywords if keyword.lower() in text_lower)
        
        # Consider it medical if:
        # 1. Contains any medical keywords, OR
        # 2. Contains multiple medical terms (indicating medical context)
        return keyword_match or medical_term_count >= 2
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # For training, we'll use the translation model as the primary model
        # since it has LoRA adapters that can be trained
        # The medical model will be used during inference for medical queries
        return self.translation_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
    
    def generate(self, input_ids, **kwargs):
        input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        if self.is_medical_query(input_text):
            return self.medical_model.generate(input_ids=input_ids, **kwargs)
        else:
            return self.translation_model.generate(input_ids=input_ids, **kwargs)
    
    def print_trainable_parameters(self):
        """Print trainable parameters for the translation model only."""
        self.translation_model.print_trainable_parameters()
    
    def get_medical_keywords(self):
        """Return the extracted medical keywords for inspection."""
        return self.medical_keywords


# --- Custom Trainer for Multi-Model MoE ---
class MultiModelSFTTrainer(CustomSFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation that handles the multi-model setup.
        During training, we use the translation model (with LoRA) for all samples.
        """
        # Use the translation model for training
        outputs = model.translation_model(**inputs)
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | list[int]],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        
        with torch.no_grad():
            pad_token_id = self.processing_class.pad_token_id
            eos_token_id = self.processing_class.eos_token_id

            # For evaluation, use the full multi-model routing
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=256,
                num_beams=1,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
        return (None, generated_ids, labels)


# --- Model Creation ---
def create_model(config):
    logger.info("Loading medical reasoning model...")
    medical_model_name = "prithivMLmods/Sculptor-Qwen3_Med-Reasoning"
    
    # Load medical reasoning model
    medical_model = AutoModelForCausalLM.from_pretrained(
        medical_model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    medical_tokenizer = AutoTokenizer.from_pretrained(medical_model_name)
    
    logger.info("Loading translation model with Unsloth...")
    # Load translation model (original setup)
    translation_model, translation_tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model["model_id_or_path"],
        max_seq_length=config.data.get("max_length", 2048),
        dtype=None,
        load_in_4bit=True,
    )
    
    # Use translation tokenizer as primary tokenizer
    tokenizer = translation_tokenizer
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    special_tokens_to_add = ["<medical>", "<en_vi>", "<vi_en>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})
    
    # Resize embeddings for both models
    medical_model.resize_token_embeddings(len(tokenizer))
    translation_model.resize_token_embeddings(len(tokenizer))
    
    # Apply LoRA to translation model only (medical model stays as is)
    translation_model = FastLanguageModel.get_peft_model(
        translation_model,
        r=int(config.lora["r"]),
        lora_alpha=int(config.lora["lora_alpha"]),
        lora_dropout=float(config.lora["lora_dropout"]),
        bias=config.lora["bias"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    # Create the multi-model MoE
    model = MultiModelMoE(medical_model, translation_model, tokenizer, config)
    model.print_trainable_parameters()
    
    return model, tokenizer


# # --- Metrics Calculation ---
# def compute_metrics(eval_preds, tokenizer):
#     predictions, labels = eval_preds
#     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#     predictions[predictions == -100] = tokenizer.pad_token_id

#     labels[labels == -100] = tokenizer.pad_token_id
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#     cleaned_preds = [pred.split("assistant\n")[-1].strip() if "assistant\n" in pred else pred for pred in decoded_preds]
#     decoded_labels_sacrebleu = [[label] for label in decoded_labels]
#     bleu = corpus_bleu(cleaned_preds, decoded_labels_sacrebleu)
#     return {"bleu": bleu.score}

def compute_metrics(eval_preds, tokenizer):
    """
    Computes BLEU score during evaluation.
    This function is called by the Trainer after our custom prediction_step.
    """
    predictions, labels = eval_preds

    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    # --- THE FIX ---
    # The 'predictions' tensor, after being gathered by the Trainer, contains
    # padding values (often -100). The tokenizer's low-level decoder cannot handle
    # negative numbers and will crash with an OverflowError.
    # We must replace these invalid padding values with the tokenizer's actual
    # pad_token_id before decoding.
    predictions[predictions == -100] = tokenizer.pad_token_id

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels[labels == -100] = tokenizer.pad_token_id
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Post-processing: remove prompt remnants from predictions if necessary.
    cleaned_preds = [pred.split("assistant\n")[-1].strip() if "assistant\n" in pred else pred for pred in decoded_preds]
    
    # Sacrebleu expects a list of reference strings for each prediction.
    decoded_labels_sacrebleu = [[label] for label in decoded_labels]

    # Calculate the BLEU score
    bleu = corpus_bleu(cleaned_preds, decoded_labels_sacrebleu)
    
    return {"bleu": bleu.score}

# --- Main Training Function ---
def train_model():
    logger.info("Starting training process...")
    config = Config(os.path.join(PROJECT_ROOT, "vlsp_moe", "configs", "moe_config.yaml"))
    
    wandb.init(project="moe_medmt", name=config.training.get("run_name", "moe_run"))
    wandb.config.update(config.__dict__, allow_val_change=True)

    model, tokenizer = create_model(config)
    max_len = config.data.get("max_length", 2048)

    logger.info("Loading datasets using the 'datasets' library...")
    raw_datasets = datasets.load_dataset('json', data_files={
        'train': config.dataset["train_file"],
        'validation': config.dataset["val_file"],
    })

    def format_for_sft(example):
        messages = example['messages']
        source_text = messages[0]["content"]
        target_text = messages[1]["content"]
        return {"text": f"{source_text}{tokenizer.eos_token}{target_text}{tokenizer.eos_token}"}

    num_proc = config.data.get('num_proc', 8)
    logger.info(f"Formatting datasets into 'text' column with {num_proc} processes...")
    formatted_datasets = raw_datasets.map(format_for_sft, num_proc=num_proc, remove_columns=raw_datasets["train"].column_names)

    # validation_subset_size = 10000
    validation_subset_size = config.data.get('validation_subset_size', 10000)
    logger.info(f"Creating a validation subset of {validation_subset_size} samples.")
    shuffled_val_dataset = formatted_datasets["validation"].shuffle(seed=42)
    validation_subset = shuffled_val_dataset.select(range(validation_subset_size))

    training_args_dict = config.training.copy()
    unsupported_args = ["predict_with_generate", "gradient_checkpointing_use_reentrant"]
    for arg in unsupported_args:
        if arg in training_args_dict:
            del training_args_dict[arg]
            
    training_args_dict["gradient_checkpointing"] = True
    training_args_dict["per_device_eval_batch_size"] = 1
    
    for key in list(training_args_dict.keys()):
        if key in ['learning_rate', 'weight_decay']: training_args_dict[key] = float(training_args_dict[key])
        elif key in ['num_train_epochs', 'max_steps', 'warmup_steps', 'logging_steps', 'save_steps', 
                     'eval_steps', 'per_device_train_batch_size', 'gradient_accumulation_steps', 'save_total_limit']:
            training_args_dict[key] = int(training_args_dict[key])

    training_args = TrainingArguments(**training_args_dict)
    compute_metrics_fn = lambda eval_preds: compute_metrics(eval_preds, tokenizer)

    trainer = MultiModelSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_datasets["train"],
        eval_dataset=validation_subset,
        dataset_text_field="text",
        args=training_args,
        compute_metrics=compute_metrics_fn,
        max_seq_length=max_len,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Training completed.")
    trainer.save_model(config.training["output_dir"])
    logger.info(f"Model saved to {config.training['output_dir']}")
    wandb.finish()


if __name__ == "__main__":
    train_model()