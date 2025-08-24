import unsloth
import torch
from config import Config
from sacrebleu import corpus_bleu
from unsloth import FastLanguageModel
import os
import torch.nn as nn
from pathlib import Path
from config import Config

# Set environment variable to force left padding for all tokenizers
os.environ["TOKENIZERS_PADDING_SIDE"] = "left"
# Ensure progress bars are enabled in both transformers/tqdm and datasets
os.environ.setdefault("TQDM_DISABLE", "0")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BAR", "0")
# Disable Hugging Face tokenizers parallelism to avoid fork warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
    return tokenizer

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
            vocab_size = config_data.get('vocab_size', 32000)
            embedding_size = config_data.get('embedding_size', 151672)
            
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

def create_model(config):
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

    # Load translation model with Flash Attention 2 for faster training
    translation_model, translation_tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model["model_id_or_path"],
        max_seq_length=config.data.get("max_length", 2048),
        dtype=None,
        load_in_4bit=True,
        attn_implementation="flash_attention_2",
        resize_model_vocab=151672
    )
    # Configure tokenizer properly for decoder-only generation
    translation_tokenizer = configure_tokenizer_for_generation(translation_tokenizer)
    
    # Use translation tokenizer as primary tokenizer (already configured)
    tokenizer = translation_tokenizer

    # fix: add PAD token
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token = '[PAD]'
        # fix: update medical_tokenizer if needed
        if medical_tokenizer.pad_token is None or medical_tokenizer.pad_token == medical_tokenizer.eos_token:
            medical_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            medical_tokenizer.pad_token = '[PAD]'

    special_tokens_to_add = ["<medical>", "<en_vi>", "<vi_en>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})

    # fix: resize embeddings for both models to match the tokenizer
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

    return model, tokenizer

if __name__ == "__main__":
    # Allow overriding config path via CLI to match run scripts
    default_config_path = os.path.join(PROJECT_ROOT, "vlsp_moe", "configs", "moe_config.yaml")
    config_path = default_config_path
    config = Config(config_path)

    model, tokenizer = create_model(config)
    model.to(DEVICE)
    model.eval()

    # input_ids = tokenizer.encode("Translate the following English sentence to Vietnamese, do not say anything else: For HFrEF patients with acute HF exacerbation already taking a beta-blocker, the dose should not be decreased or stopped unless absolutely necessary.", return_tensors="pt").to(DEVICE)
    input_ids = tokenizer.encode("Translate the following Vietnamese sentence to English: Đây chính là cơ sở cho việc vận dụng các tiêu chuẩn đã được xây dựng để kiểm soát chất lượng dược liệu lá đinh lăng và các dạng bào chế từ lá đinh lăng, góp phần vào công tác đánh giá chất lượng của dược liệu.", return_tensors="pt").to(DEVICE)
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }

    # Generate response
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=2048,
            num_beams=1,  # Keep greedy decoding for speed
            do_sample=False,  # Disable sampling for deterministic fast results
            use_cache=True,  # Enable KV cache for faster generation
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the full output
    full_output = tokenizer.decode(output[0], skip_special_tokens=False)
    print(full_output)

# if __name__ == "__main__":
#     # Allow overriding config path via CLI to match run scripts
#     default_config_path = os.path.join(PROJECT_ROOT, "vlsp_moe", "configs", "moe_config.yaml")
#     config_path = default_config_path
#     config = Config(config_path)

#     # Create model and tokenizer
#     model, tokenizer = create_model(config)
#     model.to(DEVICE)
#     model.eval()
    
#     # Test with a medical translation example
#     test_prompt = "Translate the following English sentence to Vietnamese: For HFrEF patients with acute HF exacerbation already taking a beta-blocker, the dose should not be decreased or stopped unless absolutely necessary."
    
#     print("Input prompt:")
#     print(test_prompt)
#     print("\nFormatted input (with chat template):")
#     formatted = format_input_for_inference(test_prompt, tokenizer)
#     print(repr(formatted))  # Show the actual format with eos_token
    
#     print("\nGenerated translation:")
#     result = run_inference(model, tokenizer, test_prompt)
#     print(result)
