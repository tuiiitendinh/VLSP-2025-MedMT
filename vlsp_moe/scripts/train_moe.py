import unsloth
import torch
import torch.nn as nn
from config import Config
import json
import logging
import os
from sacrebleu import corpus_bleu
import numpy as np
import datasets
import argparse
from unsloth.trainer import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling, AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel
import wandb

# Set environment variable to force left padding for all tokenizers
os.environ["TOKENIZERS_PADDING_SIDE"] = "left"
# Ensure progress bars are enabled in both transformers/tqdm and datasets
os.environ.setdefault("TQDM_DISABLE", "0")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BAR", "0")
# Disable Hugging Face tokenizers parallelism to avoid fork warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Import transformers after setting environment variable
from transformers import set_seed
import transformers

# Configure transformers logging to reduce noise
transformers.logging.set_verbosity_error()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processing_class = self.tokenizer

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | list[int]],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        
        # Use mixed precision for faster inference
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            pad_token_id = self.processing_class.pad_token_id
            eos_token_id = self.processing_class.eos_token_id

            # Optimized generation parameters for speed
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=2048,
                num_beams=1,  # Keep greedy decoding for speed
                do_sample=False,  # Disable sampling for deterministic fast results
                use_cache=True,  # Enable KV cache for faster generation
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
        return (None, generated_ids, labels)


class MultiModelMoE(nn.Module):
    def __init__(self, medical_model, translation_model, tokenizer, config):
        super().__init__()
        self.medical_model = medical_model
        self.translation_model = translation_model
        self.tokenizer = tokenizer
        
        # Use translation model's config as primary config
        self.config = translation_model.config
        
        # Ensure config has necessary attributes for compatibility
        if not hasattr(self.config, "_name_or_path"):
            setattr(self.config, "_name_or_path", "unsloth/qwen3-1.7b-unsloth-bnb-4bit")
        if not hasattr(self.config, "_attn_implementation"):
            setattr(self.config, "_attn_implementation", "flash_attention_2")
        if not hasattr(self.config, "attn_implementation"):
            setattr(self.config, "attn_implementation", "flash_attention_2")
        
        # Ensure both models have the same config attributes for compatibility
        if hasattr(medical_model, 'config'):
            medical_config = medical_model.config
            if not hasattr(medical_config, "_name_or_path"):
                setattr(medical_config, "_name_or_path", "prithivMLmods/Sculptor-Qwen3_Med-Reasoning")
            if not hasattr(medical_config, "_attn_implementation"):
                setattr(medical_config, "_attn_implementation", "flash_attention_2")
            if not hasattr(medical_config, "attn_implementation"):
                setattr(medical_config, "attn_implementation", "flash_attention_2")
        
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
        # Handle batch processing more efficiently
        batch_size = input_ids.shape[0]
        
        if batch_size == 1:
            # Single sample - use existing logic
            input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            
            if self.is_medical_query(input_text):
                return self.medical_model.generate(input_ids=input_ids, **kwargs)
            else:
                return self.translation_model.generate(input_ids=input_ids, **kwargs)
        else:
            # Batch processing - route all to translation model for speed
            # During training/validation, prioritize speed over expert routing
            return self.translation_model.generate(input_ids=input_ids, **kwargs)
    
    def print_trainable_parameters(self):
        """Print trainable parameters for the translation model only."""
        self.translation_model.print_trainable_parameters()
    
    def get_medical_keywords(self):
        """Return the extracted medical keywords for inspection."""
        return self.medical_keywords
    
    def get_input_embeddings(self):
        """Return the input embedding layer of the translation model, monkey-patched to have a .dtype property."""
        embedding = self.translation_model.get_input_embeddings()
        # Monkey-patch: add a .dtype property to the embedding layer
        if not hasattr(embedding, 'dtype'):
            embedding.dtype = embedding.weight.dtype
        return embedding

    def get_output_embeddings(self):
        """Return the output embedding (lm head) of the translation model."""
        return self.translation_model.get_output_embeddings()

    # Trainer/HF compatibility shims
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.translation_model, "gradient_checkpointing_enable"):
            # HF signature allows optional kwargs
            if gradient_checkpointing_kwargs is not None:
                self.translation_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                )
            else:
                self.translation_model.gradient_checkpointing_enable()
        if hasattr(self.medical_model, "gradient_checkpointing_enable"):
            try:
                if gradient_checkpointing_kwargs is not None:
                    self.medical_model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                    )
                else:
                    self.medical_model.gradient_checkpointing_enable()
            except TypeError:
                self.medical_model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        if hasattr(self.translation_model, "gradient_checkpointing_disable"):
            self.translation_model.gradient_checkpointing_disable()
        if hasattr(self.medical_model, "gradient_checkpointing_disable"):
            self.medical_model.gradient_checkpointing_disable()

    def enable_input_require_grads(self):
        if hasattr(self.translation_model, "enable_input_require_grads"):
            self.translation_model.enable_input_require_grads()
        if hasattr(self.medical_model, "enable_input_require_grads"):
            self.medical_model.enable_input_require_grads()
    
    def save_pretrained(self, save_directory, **kwargs):
        """Custom save method to handle shared tensors properly."""
        import os
        import torch
        
        # Create the save directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the translation model (which has the LoRA adapters)
        if hasattr(self.translation_model, 'save_pretrained'):
            try:
                # Try to save with safe serialization first, including embedding layers
                self.translation_model.save_pretrained(
                    save_directory,
                    safe_serialization=True,
                    save_embedding_layers=True,
                    **kwargs,
                )
            except Exception as e:
                print(f"Warning: Could not save with safe_serialization: {e}")
                # Fallback: save with embedding layers explicitly
                if hasattr(self.translation_model, 'peft_config'):
                    # Save with embedding layers for PEFT models
                    self.translation_model.save_pretrained(save_directory, save_embedding_layers=True, **kwargs)
                else:
                    # Last resort: save only the state dict
                    state_dict = self.translation_model.state_dict()
                    torch.save(state_dict, os.path.join(save_directory, 'pytorch_model.bin'))
        
        # Save the medical model's embedding layers separately to preserve MoE integrity
        medical_embeddings_dir = os.path.join(save_directory, 'medical_embeddings')
        os.makedirs(medical_embeddings_dir, exist_ok=True)
        
        # Extract and save medical model's embedding layers
        if hasattr(self.medical_model, 'get_input_embeddings'):
            medical_embeddings = self.medical_model.get_input_embeddings()
            if hasattr(medical_embeddings, 'weight'):
                torch.save(medical_embeddings.weight, os.path.join(medical_embeddings_dir, 'embed_tokens.weight'))
                print("Saved medical model embedding layers")
        
        if hasattr(self.medical_model, 'get_output_embeddings'):
            medical_lm_head = self.medical_model.get_output_embeddings()
            if hasattr(medical_lm_head, 'weight'):
                torch.save(medical_lm_head.weight, os.path.join(medical_embeddings_dir, 'lm_head.weight'))
                print("Saved medical model language model head")
        
        # Save the tokenizer
        if hasattr(self.tokenizer, 'save_pretrained'):
            self.tokenizer.save_pretrained(save_directory, **kwargs)
        
        # Save medical keywords and model configuration
        config_data = {
            'medical_keywords': self.medical_keywords,
            'medical_model_name': 'prithivMLmods/Sculptor-Qwen3_Med-Reasoning',
            'translation_model_name': getattr(self.config, '_name_or_path', 'Qwen/Qwen3-1.7B'),
            'vocab_size': len(self.tokenizer),
            'embedding_size': getattr(self.config, 'hidden_size', 2048),
        }
        
        import json
        with open(os.path.join(save_directory, 'moe_config.json'), 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"MultiModelMoE saved to {save_directory}")
        print("Note: Translation model (with LoRA) and medical model embeddings were saved.")
        print("The medical model will be reloaded from its original checkpoint during inference.")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Custom load method to reconstruct the MultiModelMoE."""
        from transformers import AutoTokenizer
        
        # Load the saved configuration
        import json
        import os
        
        config_path = os.path.join(pretrained_model_name_or_path, 'moe_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            medical_model_name = config_data.get('medical_model_name', 'prithivMLmods/Sculptor-Qwen3_Med-Reasoning')
            translation_model_name = config_data.get('translation_model_name', 'Qwen/Qwen3-1.7B')
            medical_keywords = config_data.get('medical_keywords', [])
            vocab_size = config_data.get('vocab_size', 151936)  # Default Qwen3 vocab size
            embedding_size = config_data.get('embedding_size', 2048)
            
            # Load the translation model (which includes LoRA adapters)
            translation_model, translation_tokenizer = FastLanguageModel.from_pretrained(
                model_name=pretrained_model_name_or_path,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
                attn_implementation="flash_attention_2",
            )
            # Configure tokenizer properly for decoder-only generation
            translation_tokenizer = configure_tokenizer_for_generation(translation_tokenizer)
            
            # Load the medical model from its original checkpoint
            medical_model, medical_tokenizer_temp = FastLanguageModel.from_pretrained(
                model_name=medical_model_name,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
                attn_implementation="flash_attention_2",
            )
            # Configure tokenizer properly for decoder-only generation
            medical_tokenizer_temp = configure_tokenizer_for_generation(medical_tokenizer_temp)
            
            # Resize both models to match the saved vocabulary size
            if hasattr(medical_model, 'resize_token_embeddings'):
                medical_model.resize_token_embeddings(vocab_size)
            if hasattr(translation_model, 'resize_token_embeddings'):
                translation_model.resize_token_embeddings(vocab_size)
            
            # Load saved medical model embeddings if they exist
            medical_embeddings_dir = os.path.join(pretrained_model_name_or_path, 'medical_embeddings')
            if os.path.exists(medical_embeddings_dir):
                try:
                    # Load medical model embedding layers
                    embed_path = os.path.join(medical_embeddings_dir, 'embed_tokens.weight')
                    if os.path.exists(embed_path):
                        embed_weight = torch.load(embed_path, map_location='cpu')
                        if hasattr(medical_model, 'get_input_embeddings'):
                            medical_embeddings = medical_model.get_input_embeddings()
                            if hasattr(medical_embeddings, 'weight'):
                                medical_embeddings.weight.data = embed_weight.to(medical_embeddings.weight.device)
                                print("Loaded medical model embedding layers")
                    
                    # Load medical model language model head
                    lm_head_path = os.path.join(medical_embeddings_dir, 'lm_head.weight')
                    if os.path.exists(lm_head_path):
                        lm_head_weight = torch.load(lm_head_path, map_location='cpu')
                        if hasattr(medical_model, 'get_output_embeddings'):
                            medical_lm_head = medical_model.get_output_embeddings()
                            if hasattr(medical_lm_head, 'weight'):
                                medical_lm_head.weight.data = lm_head_weight.to(medical_lm_head.weight.device)
                                print("Loaded medical model language model head")
                except Exception as e:
                    print(f"Warning: Could not load medical model embeddings: {e}")
            
            # Create a dummy config for compatibility
            class DummyConfig:
                def __init__(self):
                    self._name_or_path = translation_model_name
                    self._attn_implementation = "flash_attention_2"
                    self.attn_implementation = "flash_attention_2"
                    self.hidden_size = embedding_size
            
            config = DummyConfig()
            
            # Create the MultiModelMoE instance
            model = cls(medical_model, translation_model, translation_tokenizer, config)
            model.medical_keywords = medical_keywords
            
            return model
        else:
            raise ValueError(f"Could not find moe_config.json in {pretrained_model_name_or_path}")


# --- Custom Trainer for Multi-Model MoE ---
class MultiModelSFTTrainer(CustomSFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch: int | None = None, **kwargs):
        """
        Custom loss computation that handles the multi-model setup.
        During training, we use the translation model (with LoRA) for all samples.
        """
        # Use the translation model for training
        outputs = model.translation_model(**inputs)
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss
    
    def _save_checkpoint(self, model, trial, metrics=None):
        """Override save checkpoint to use custom save method."""
        # Use the custom save_pretrained method of MultiModelMoE
        if hasattr(model, 'save_pretrained'):
            checkpoint_folder = f"{self.args.output_dir}/checkpoint-{self.state.global_step}"
            run_dir = self.args.output_dir if self.args.should_save else None
            os.makedirs(checkpoint_folder, exist_ok=True)
            
            # Save the model using custom method
            model.save_pretrained(checkpoint_folder)
            
            # Save training arguments
            if self.args.should_save:
                torch.save(self.args, os.path.join(run_dir, "training_args.bin"))
                
            # Save optimizer state
            if self.args.should_save and hasattr(self, "optimizer"):
                torch.save(self.optimizer.state_dict(), os.path.join(run_dir, "optimizer.bin"))
                
            # Save scheduler state
            if self.args.should_save and hasattr(self, "lr_scheduler"):
                torch.save(self.lr_scheduler.state_dict(), os.path.join(run_dir, "scheduler.bin"))
            
            # Save metrics for best model tracking
            if metrics is not None:
                torch.save(metrics, os.path.join(checkpoint_folder, "metrics.bin"))
        else:
            # Fallback to default save method
            super()._save_checkpoint(model, trial, metrics)
    
    def save_model(self, output_dir=None, _internal_call=False):
        """Override save_model to handle shared tensors."""
        if output_dir is None:
            output_dir = self.args.output_dir
        
        if hasattr(self.model, 'save_pretrained'):
            # Use custom save method
            self.model.save_pretrained(output_dir)
        else:
            # Fallback to default method
            super().save_model(output_dir, _internal_call)
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | list[int]],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        
        # Use mixed precision for faster inference
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            pad_token_id = self.processing_class.pad_token_id
            eos_token_id = self.processing_class.eos_token_id

            # Optimized generation parameters for speed
            # For evaluation, use the full multi-model routing with speed optimizations
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=2048,
                num_beams=1,  # Keep greedy decoding for speed
                do_sample=False,  # Disable sampling for deterministic fast results
                use_cache=True,  # Enable KV cache for faster generation
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
        return (None, generated_ids, labels)


# --- Model Creation ---
def create_model(config):
    logger.info("Loading medical reasoning model...")
    medical_model_name = "prithivMLmods/Sculptor-Qwen3_Med-Reasoning"
    
    # Load medical reasoning model with Unsloth for compatibility
    medical_model, medical_tokenizer = FastLanguageModel.from_pretrained(
        model_name=medical_model_name,
        max_seq_length=config.data.get("max_length", 2048),
        dtype=None,
        load_in_4bit=True,
        attn_implementation="flash_attention_2",
    )
    # Configure tokenizer properly for decoder-only generation
    medical_tokenizer = configure_tokenizer_for_generation(medical_tokenizer)
    
    logger.info("Loading translation model with Unsloth...")
    # Load translation model with Flash Attention 2 for faster training
    translation_model, translation_tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model["model_id_or_path"],
        max_seq_length=config.data.get("max_length", 2048),
        dtype=None,
        load_in_4bit=True,
        attn_implementation="flash_attention_2",
    )
    # Configure tokenizer properly for decoder-only generation
    translation_tokenizer = configure_tokenizer_for_generation(translation_tokenizer)
    
    # Use translation tokenizer as primary tokenizer (already configured)
    tokenizer = translation_tokenizer
    
    special_tokens_to_add = ["<medical>", "<en_vi>", "<vi_en>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})
    
    # Resize embeddings for both models to match the tokenizer
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


# --- Metrics Calculation ---
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

    # Replace invalid padding values (-100) with the tokenizer's actual pad_token_id.
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
    
    return {"eval_bleu": bleu.score}

# --- Main Training Function ---
def train_model():
    logger.info("Starting training process...")
    # Allow overriding config path via CLI to match run scripts
    default_config_path = os.path.join(PROJECT_ROOT, "vlsp_moe", "configs", "moe_config.yaml")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config_path)
    args, _ = parser.parse_known_args()

    config_path = args.config if args and args.config else default_config_path
    config = Config(config_path)

    model, tokenizer = create_model(config)
    max_len = config.data.get("max_length", 2048)

    logger.info("Preparing datasets (with caching) using the 'datasets' library...")

    formatted_datasets = None
    # 1) Prefer our primary cache location first
    primary_cache_dir = os.path.join(PROJECT_ROOT, "cache", "hf_formatted")
    if os.path.exists(primary_cache_dir):
        try:
            logger.info(f"Loading formatted datasets from cache: {primary_cache_dir}")
            formatted_datasets = datasets.load_from_disk(primary_cache_dir)
        except Exception as e:
            logger.warning(f"Could not load formatted datasets from '{primary_cache_dir}': {e}")

    # 2) Else try user-provided tokenized path (if it is a HF dataset dir)
    if formatted_datasets is None:
        tokenized_dir = config.data.get("tokenized") if hasattr(config, "data") else None
        if tokenized_dir:
            tokenized_abs = os.path.join(PROJECT_ROOT, tokenized_dir) if not os.path.isabs(tokenized_dir) else tokenized_dir
            if os.path.exists(tokenized_abs):
                try:
                    logger.info(f"Found preprocessed dataset at '{tokenized_abs}', attempting to load...")
                    formatted_datasets = datasets.load_from_disk(tokenized_abs)
                except Exception as e:
                    logger.warning(f"Could not load preprocessed dataset from '{tokenized_abs}': {e}")

    # 3) Else load raw JSON and format once, then save to primary cache
    if formatted_datasets is None:
        raw_datasets = datasets.load_dataset(
            'json',
            data_files={
                'train': config.dataset["train_file"],
                'validation': config.dataset["val_file"],
            },
        )

    def format_for_sft(example):
        messages = example['messages']
        source_text = messages[0]["content"]
        target_text = messages[1]["content"]
        return {"text": f"{source_text}{tokenizer.eos_token}{target_text}{tokenizer.eos_token}"}
    
    # Build formatted dataset only if we couldn't load any cache
    if formatted_datasets is None:
        num_proc = config.data.get('num_proc', 8)
        logger.info(f"Formatting datasets into 'text' column with {num_proc} processes...")
        formatted_datasets = raw_datasets.map(
            format_for_sft,
            num_proc=num_proc,
            remove_columns=raw_datasets["train"].column_names,
            load_from_cache_file=True,
            desc="Formatting to text",
        )

        # Save formatted dataset to disk cache for future fast starts
        try:
            os.makedirs(primary_cache_dir, exist_ok=True)
            formatted_datasets.save_to_disk(primary_cache_dir)
            logger.info(f"Saved formatted datasets to cache: {primary_cache_dir}")
        except Exception as e:
            logger.warning(f"Could not save formatted datasets to cache: {e}")

    validation_subset_size = config.data.get('validation_subset_size', None)
    
    if validation_subset_size is not None and validation_subset_size > 0:
        logger.info(f"Creating a validation subset of {validation_subset_size} samples.")
        shuffled_val_dataset = formatted_datasets["validation"].shuffle(seed=42)
        validation_dataset = shuffled_val_dataset.select(range(validation_subset_size))
    else:
        logger.info("Using the full validation dataset.")
        validation_dataset = formatted_datasets["validation"]
        logger.info(f"Full validation dataset size: {len(validation_dataset)} samples.")

    training_args_dict = config.training.copy()
    unsupported_args = ["predict_with_generate", "gradient_checkpointing_use_reentrant"]
    for arg in unsupported_args:
        if arg in training_args_dict:
            del training_args_dict[arg]
            
    # Optionally speed up first start by disabling heavy compile/checkpointing via env FAST_START=1
    fast_start = os.environ.get("FAST_START", "0") == "1"
    if fast_start:
        logger.info("FAST_START enabled: Disabling torch_compile and gradient_checkpointing for quicker startup.")
        training_args_dict["torch_compile"] = False
        training_args_dict["gradient_checkpointing"] = False
        # Reduce early eval/save frequency to avoid stalls
        if "eval_steps" in training_args_dict and isinstance(training_args_dict["eval_steps"], int):
            training_args_dict["eval_steps"] = max(200, int(training_args_dict["eval_steps"]))
        if "save_steps" in training_args_dict and isinstance(training_args_dict["save_steps"], int):
            training_args_dict["save_steps"] = max(200, int(training_args_dict["save_steps"]))
    else:
        training_args_dict["gradient_checkpointing"] = True
    # Use the eval batch size from config instead of hardcoding to 1
    # training_args_dict["per_device_eval_batch_size"] = 1  # Removed to use config value
    
    # Additional optimizations for faster evaluation
    training_args_dict["eval_accumulation_steps"] = 4  # Accumulate eval batches to reduce memory transfers
    training_args_dict["dataloader_pin_memory"] = True  # Faster data loading
    training_args_dict["dataloader_num_workers"] = min(8, os.cpu_count() or 4)  # Optimized workers
    # Force-enable tqdm progress bar in Trainer
    training_args_dict["disable_tqdm"] = False
    
    for key in list(training_args_dict.keys()):
        if key in ['learning_rate', 'weight_decay']:
            if training_args_dict[key] is not None:
                training_args_dict[key] = float(training_args_dict[key])
        elif key in ['num_train_epochs', 'max_steps', 'warmup_steps', 'logging_steps', 'save_steps', 
                     'eval_steps', 'per_device_train_batch_size', 'gradient_accumulation_steps', 'save_total_limit']:
            if training_args_dict[key] is not None:
                training_args_dict[key] = int(training_args_dict[key])

    training_args = TrainingArguments(**training_args_dict)
    compute_metrics_fn = lambda eval_preds: compute_metrics(eval_preds, tokenizer)

    trainer = MultiModelSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_datasets["train"],
        eval_dataset=validation_dataset,
        dataset_text_field="text",
        args=training_args,
        compute_metrics=compute_metrics_fn,
        max_seq_length=max_len,
    )
    
    # Log the best model saving configuration
    logger.info(f"Best model saving configuration:")
    logger.info(f"  - load_best_model_at_end: {training_args.load_best_model_at_end}")
    logger.info(f"  - metric_for_best_model: {training_args.metric_for_best_model}")
    logger.info(f"  - greater_is_better: {training_args.greater_is_better}")
    logger.info(f"  - save_total_limit: {training_args.save_total_limit}")

    # Initialize Weights & Biases after data and trainer are ready to avoid network delays at start
    try:
        wandb_mode = os.environ.get("WANDB_MODE", "online")
        wandb.init(project="moe_medmt", name=config.training.get("run_name", "moe_run"))
        wandb.config.update(config.__dict__, allow_val_change=True)
        logger.info(f"Initialized W&B (mode={wandb_mode}).")
    except Exception as e:
        logger.warning(f"W&B init failed or skipped: {e}")

    logger.info("Starting training...")
    trainer.train()

    logger.info("Training completed.")
    # The trainer will automatically save the best model if load_best_model_at_end is True
    # We don't need to manually save here as the trainer handles it
    logger.info(f"Best model should be saved to {config.training['output_dir']}")
    wandb.finish()


if __name__ == "__main__":
    train_model()