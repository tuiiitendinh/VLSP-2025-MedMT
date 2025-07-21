import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from config import Config
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MoEGatingNetwork(nn.Module):
    """Gating network for MoE that routes inputs to appropriate experts."""

    def __init__(self, input_size: int, num_experts: int, hidden_size: int = 256):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_experts),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size)
        Returns:
            gate_weights: Tensor of shape (batch_size, num_experts)
        """
        return self.gate(x)


class MoEExpertRouter(nn.Module):
    """Routes inputs to different experts based on task type."""

    def __init__(self, model, tokenizer, num_experts: int = 3):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.num_experts = num_experts
        self.expert_mapping = {"medical": 0, "en_vi": 1, "vi_en": 2}

        # Create separate LoRA adapters for each expert
        self.expert_adapters = {}
        for expert_name, expert_id in self.expert_mapping.items():
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.expert_adapters[expert_name] = lora_config

        # Initialize gating network
        # Use model's hidden size for gating
        hidden_size = (
            model.config.hidden_size if hasattr(model.config, "hidden_size") else 1024
        )
        self.gating_network = MoEGatingNetwork(hidden_size, num_experts)

    def get_expert_weights(self, input_ids, expert_labels=None):
        """Get routing weights for experts."""
        batch_size = input_ids.size(0)

        if expert_labels is not None:
            # During training, use ground truth expert labels
            expert_weights = torch.zeros(
                batch_size, self.num_experts, device=input_ids.device
            )
            for i, expert_label in enumerate(expert_labels):
                if expert_label in self.expert_mapping:
                    expert_id = self.expert_mapping[expert_label]
                    expert_weights[i, expert_id] = 1.0
        else:
            # During inference, use gating network
            with torch.no_grad():
                embeddings = self.model.get_input_embeddings()(input_ids)
                pooled_embeddings = embeddings.mean(dim=1)
                
                if self.gating_network.gate[0].weight.device != pooled_embeddings.device:
                    self.gating_network = self.gating_network.to(pooled_embeddings.device)
                
                expert_weights = self.gating_network(pooled_embeddings)

        return expert_weights


class MoEDataset(Dataset):
    """Dataset class for MoE training."""

    def __init__(self, data_path: str, tokenizer):
        self.data = []
        self.tokenizer = tokenizer

        # Load data
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        expert_type = item.get("expert", "general")
        messages = item["messages"]

        source_text = messages[0]["content"]
        target_text = messages[1]["content"]
        formatted_text = f"{source_text} {target_text}"

        encoding = self.tokenizer(
            formatted_text,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )

        # For causal LM, input_ids and labels are the same
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Create labels (same as input_ids for causal LM)
        labels = input_ids.copy()

        # Mask the source part in labels to only compute loss on target
        source_encoding = self.tokenizer(
            source_text,
            max_length=512,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        source_len = len(source_encoding["input_ids"])

        # Set labels for source tokens to -100 (ignored in loss)
        labels[:source_len] = [-100] * source_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "expert_type": expert_type,
        }


class MoETrainer(Trainer):
    """Custom trainer for MoE model."""

    def __init__(self, moe_router, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.moe_router = moe_router

    def _move_model_to_device(self, model, device):
        """Override to also move MoE router to device."""
        model = super()._move_model_to_device(model, device)
        if hasattr(self, 'moe_router') and self.moe_router is not None:
            self.moe_router = self.moe_router.to(device)
        return model

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss with expert routing."""
        expert_types = inputs.pop("expert_type", None)
        expert_weights = self.moe_router.get_expert_weights(
            inputs["input_ids"], expert_types
        )
        
        outputs = model(**inputs)

        loss = outputs.loss

        if expert_weights is not None:
            routing_loss = self.compute_routing_loss(expert_weights)
            loss = loss + 0.01 * routing_loss 

        return (loss, outputs) if return_outputs else loss

    def compute_routing_loss(self, expert_weights):
        """Compute routing loss to encourage sparsity."""
        batch_size = expert_weights.size(0)
        expert_usage = expert_weights.sum(dim=0)
        target_usage = batch_size / self.moe_router.num_experts
        load_loss = F.mse_loss(
            expert_usage, torch.full_like(expert_usage, target_usage)
        )

        return load_loss


def create_moe_model(config):
    """Create MoE model with multiple experts."""
    logger.info("Loading base model and tokenizer...")

    # Load base model and tokenizer using standard transformers
    model_name = config.model["model_id_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Add special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA to the base model
    lora_config = LoraConfig(**config.lora)
    model = get_peft_model(model, lora_config)

    # Create MoE router after LoRA is applied
    moe_router = MoEExpertRouter(model, tokenizer, num_experts=3)
    
    # Move MoE router to the same device as the model
    device = next(model.parameters()).device
    moe_router = moe_router.to(device)

    logger.info("MoE model created successfully")
    return model, tokenizer, moe_router


def train_moe_model():
    """Main training function for MoE model."""
    logger.info("Starting MoE training...")
    config = Config("../configs/moe_config.yaml")

    model, tokenizer, moe_router = create_moe_model(config)

    logger.info("Loading training dataset...")
    train_dataset = MoEDataset(config.dataset["train_file"], tokenizer)
    
    logger.info("Loading validation dataset...")
    eval_dataset = MoEDataset(config.dataset["val_file"], tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8  
    )
    
    training_config = config.training.copy()
    training_config["learning_rate"] = float(training_config["learning_rate"])
    training_config["weight_decay"] = float(training_config["weight_decay"])
    training_config["warmup_steps"] = int(training_config["warmup_steps"])
    training_config["max_steps"] = int(training_config["max_steps"])
    training_config["logging_steps"] = int(training_config["logging_steps"])
    training_config["save_steps"] = int(training_config["save_steps"])
    training_config["eval_steps"] = int(training_config["eval_steps"])
    training_config["per_device_train_batch_size"] = int(training_config["per_device_train_batch_size"])
    training_config["per_device_eval_batch_size"] = int(training_config["per_device_eval_batch_size"])
    training_config["gradient_accumulation_steps"] = int(training_config["gradient_accumulation_steps"])
    training_config["save_total_limit"] = int(training_config["save_total_limit"])
    training_config["num_train_epochs"] = int(training_config["num_train_epochs"])
    
    training_args = TrainingArguments(**training_config)
    trainer = MoETrainer(
        moe_router=moe_router,
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving model...")
    trainer.save_model()

    logger.info("MoE training completed!")


if __name__ == "__main__":
    train_moe_model()
