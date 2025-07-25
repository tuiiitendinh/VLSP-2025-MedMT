import torch
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from transformers.data.data_collator import DataCollatorMixin
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForSPMLanguageModeling(DataCollatorMixin):
    """
    Data collator for SentencePiece causal language modeling.
    """
    
    tokenizer: Any
    mlm: bool = False  # Set to False for causal LM
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def __post_init__(self):
        if self.mlm and not hasattr(self.tokenizer, "mask_token"):
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling."
            )

    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Handle the input features
        batch = {}
        
        # Extract input_ids and attention_mask
        input_ids = []
        attention_mask = []
        labels = []
        expert_types = []
        
        for example in examples:
            input_ids.append(example["input_ids"])
            attention_mask.append(example["attention_mask"])
            labels.append(example.get("labels", example["input_ids"]))
            expert_types.append(example.get("expert_type", "general"))
        
        # Pad sequences to the same length
        max_length = max(len(ids) for ids in input_ids)
        
        # Apply pad_to_multiple_of if specified
        if self.pad_to_multiple_of is not None:
            max_length = (
                (max_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )
        
        # Pad input_ids
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        
        pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)
        
        for i in range(len(input_ids)):
            current_length = len(input_ids[i])
            pad_length = max_length - current_length
            
            # Pad input_ids
            padded_ids = input_ids[i] + [pad_token_id] * pad_length
            padded_input_ids.append(padded_ids)
            
            # Pad attention_mask
            padded_mask = attention_mask[i] + [0] * pad_length
            padded_attention_mask.append(padded_mask)
            
            # Pad labels
            padded_label = labels[i] + [-100] * pad_length
            padded_labels.append(padded_label)
        
        batch["input_ids"] = torch.tensor(padded_input_ids, dtype=torch.long)
        batch["attention_mask"] = torch.tensor(padded_attention_mask, dtype=torch.long)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        batch["expert_type"] = expert_types
        
        return batch

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        if return_tensors is None:
            return_tensors = self.return_tensors
        
        if return_tensors == "pt":
            return self.torch_call(features)
        else:
            raise ValueError(f"Framework '{return_tensors}' not recognized!")


class SPMDataCollatorForLanguageModeling:
    """Simple data collator for SentencePiece language modeling."""
    
    def __init__(self, tokenizer, mlm: bool = False, pad_to_multiple_of: Optional[int] = None):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.pad_to_multiple_of = pad_to_multiple_of
        
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {}
        
        # Extract features
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features] 
        labels = [f.get("labels", f["input_ids"]) for f in features]
        expert_types = [f.get("expert_type", "general") for f in features]
        
        # Get max length
        max_length = max(len(ids) for ids in input_ids)
        
        # Apply pad_to_multiple_of
        if self.pad_to_multiple_of is not None:
            max_length = ((max_length + self.pad_to_multiple_of - 1) 
                         // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        # Pad sequences
        pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)
        
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        
        for i in range(len(input_ids)):
            pad_length = max_length - len(input_ids[i])
            
            # Pad input_ids
            padded_ids = input_ids[i] + [pad_token_id] * pad_length
            padded_input_ids.append(padded_ids)
            
            # Pad attention_mask  
            padded_mask = attention_mask[i] + [0] * pad_length
            padded_attention_mask.append(padded_mask)
            
            # Pad labels
            padded_label = labels[i] + [-100] * pad_length
            padded_labels.append(padded_label)
        
        batch["input_ids"] = torch.tensor(padded_input_ids, dtype=torch.long)
        batch["attention_mask"] = torch.tensor(padded_attention_mask, dtype=torch.long)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        batch["expert_type"] = expert_types
        
        return batch
