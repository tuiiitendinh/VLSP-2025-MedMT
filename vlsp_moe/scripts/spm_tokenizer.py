import sentencepiece as spm
import os
import logging
from typing import List, Dict, Any, Optional, Union
import torch

logger = logging.getLogger(__name__)


class SPMTokenizer:
    """SentencePiece tokenizer wrapper for compatibility with transformers interface."""
    
    def __init__(self, model_path: str = None, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model_path = model_path
        self.sp = spm.SentencePieceProcessor()
        
        # Special token IDs from config
        self.pad_token_id = self.config.get("pad_id", 0)
        self.unk_token_id = self.config.get("unk_id", 1)
        self.bos_token_id = self.config.get("bos_id", 2)
        self.eos_token_id = self.config.get("eos_id", 3)
        
        # Special tokens
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
    def train_tokenizer(self, input_files: List[str], output_dir: str, vocab_size: int = 32000):
        """Train a new SentencePiece model."""
        os.makedirs(output_dir, exist_ok=True)
        
        model_prefix = os.path.join(output_dir, self.config.get("model_prefix", "spm_model"))
        
        # Prepare training arguments
        train_args = [
            f"--input={','.join(input_files)}",
            f"--model_prefix={model_prefix}",
            f"--vocab_size={vocab_size}",
            f"--model_type={self.config.get('model_type', 'unigram')}",
            f"--input_sentence_size={self.config.get('input_sentence_size', 10000000)}",
            f"--character_coverage={self.config.get('character_coverage', 0.995)}",
            f"--split_by_unicode_script={self.config.get('split_by_unicode_script', True)}",
            f"--split_by_number={self.config.get('split_by_number', True)}",
            f"--split_by_whitespace={self.config.get('split_by_whitespace', True)}",
            f"--treat_whitespace_as_suffix={self.config.get('treat_whitespace_as_suffix', False)}",
            f"--allow_whitespace_only_pieces={self.config.get('allow_whitespace_only_pieces', True)}",
            f"--split_digits={self.config.get('split_digits', True)}",
            f"--pad_id={self.pad_token_id}",
            f"--unk_id={self.unk_token_id}",
            f"--bos_id={self.bos_token_id}",
            f"--eos_id={self.eos_token_id}",
        ]
        
        # Add user defined symbols if any
        user_symbols = self.config.get("user_defined_symbols", [])
        if user_symbols:
            train_args.append(f"--user_defined_symbols={','.join(user_symbols)}")
        
        logger.info(f"Training SentencePiece model with args: {' '.join(train_args)}")
        
        # Train the model
        spm.SentencePieceTrainer.train(' '.join(train_args),
                                        input_sentence_size=1000000,
                                        shuffle_input_sentence=True,
                                        verbose=True)
        
        # Load the trained model
        self.model_path = f"{model_prefix}.model"
        self.load_model(self.model_path)
        
        logger.info(f"SentencePiece model trained and saved to {self.model_path}")
        return self.model_path
    
    def load_model(self, model_path: str):
        """Load a trained SentencePiece model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SentencePiece model not found: {model_path}")
        
        self.sp.load(model_path)
        self.model_path = model_path
        logger.info(f"Loaded SentencePiece model from {model_path}")
        
        # Update vocab size
        self.vocab_size = self.sp.get_piece_size()
        
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        if not hasattr(self.sp, 'encode'):
            raise RuntimeError("SentencePiece model not loaded")
        
        if add_special_tokens:
            text = f"{self.bos_token} {text} {self.eos_token}"
        
        return self.sp.encode(text, out_type=int)
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        if not hasattr(self.sp, 'decode'):
            raise RuntimeError("SentencePiece model not loaded")
        
        text = self.sp.decode(token_ids)
        
        if skip_special_tokens:
            # Remove special tokens
            for special_token in [self.bos_token, self.eos_token, self.pad_token]:
                text = text.replace(special_token, "")
            text = text.strip()
        
        return text
    
    def __call__(self, 
                 text: Union[str, List[str]], 
                 max_length: int = None,
                 truncation: bool = True,
                 padding: Union[str, bool] = False,
                 return_tensors: str = None,
                 add_special_tokens: bool = True) -> Dict[str, Any]:
        """Tokenize text with transformers-like interface."""
        
        if isinstance(text, str):
            text = [text]
        
        encoded_batch = []
        attention_masks = []
        
        for t in text:
            # Encode text
            token_ids = self.encode(t, add_special_tokens=add_special_tokens)
            
            # Apply truncation
            if truncation and max_length and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
                # Ensure we end with EOS token if it was truncated
                if add_special_tokens and token_ids[-1] != self.eos_token_id:
                    token_ids[-1] = self.eos_token_id
            
            # Create attention mask
            attention_mask = [1] * len(token_ids)
            
            # Apply padding
            if padding and max_length:
                pad_length = max_length - len(token_ids)
                if pad_length > 0:
                    token_ids.extend([self.pad_token_id] * pad_length)
                    attention_mask.extend([0] * pad_length)
            
            encoded_batch.append(token_ids)
            attention_masks.append(attention_mask)
        
        result = {
            "input_ids": encoded_batch,
            "attention_mask": attention_masks
        }
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            result["input_ids"] = torch.tensor(result["input_ids"])
            result["attention_mask"] = torch.tensor(result["attention_mask"])
        elif return_tensors is None:
            # Return lists for single item
            if len(encoded_batch) == 1:
                result["input_ids"] = encoded_batch[0]
                result["attention_mask"] = attention_masks[0]
        
        return result
    
    def batch_decode(self, sequences: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """Batch decode sequences."""
        return [self.decode(seq, skip_special_tokens) for seq in sequences]
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return getattr(self, 'vocab_size', self.sp.get_piece_size() if hasattr(self.sp, 'get_piece_size') else 32000)
    
    def save_pretrained(self, save_directory: str):
        """Save tokenizer (copy the model file)."""
        os.makedirs(save_directory, exist_ok=True)
        
        if self.model_path and os.path.exists(self.model_path):
            import shutil
            target_path = os.path.join(save_directory, "spm_model.model")
            shutil.copy2(self.model_path, target_path)
            logger.info(f"Tokenizer saved to {save_directory}")
        else:
            logger.warning("No trained model to save")


def create_spm_tokenizer(config: Dict[str, Any], data_files: List[str] = None, model_path: str = None) -> SPMTokenizer:
    """Create and optionally train a SentencePiece tokenizer."""
    tokenizer = SPMTokenizer(config=config)
    
    if model_path and os.path.exists(model_path):
        # Load existing model
        tokenizer.load_model(model_path)
    elif data_files:
        # Train new model
        output_dir = config.get("output_dir", "./tokenizer_output")
        vocab_size = config.get("vocab_size", 32000)
        model_path = tokenizer.train_tokenizer(data_files, output_dir, vocab_size)
    else:
        raise ValueError("Either provide existing model_path or data_files for training")
    
    return tokenizer
