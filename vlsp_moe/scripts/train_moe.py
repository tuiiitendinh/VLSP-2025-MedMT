import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from config import Config
from spm_tokenizer import SPMTokenizer, create_spm_tokenizer
from spm_data_collator import SPMDataCollatorForLanguageModeling
import json
import logging
import os
from sacrebleu import corpus_bleu

# Thêm các import cần thiết
import unsloth
from transformers import DataCollatorForLanguageModeling, TrainingArguments
from peft import TaskType
from unsloth import FastLanguageModel
from unsloth.trainer import SFTTrainer
import wandb

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Auto-select device
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        DEVICE = torch.device("cuda")
        logger.info(f"Using {n_gpus} GPUs: {[torch.cuda.get_device_name(i) for i in range(n_gpus)]}")
    else:
        DEVICE = torch.device("cuda")
        logger.info("Using GPU: %s", torch.cuda.get_device_name(DEVICE))
else:
    DEVICE = torch.device("cpu")
    logger.info("Using CPU")


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

        # Unsloth handles LoRA globally; no per-expert configs needed
        self.expert_adapters = {}

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
                # Handle both HuggingFace and SPM tokenizers
                if hasattr(self.model, 'get_input_embeddings'):
                    embeddings = self.model.get_input_embeddings()(input_ids)
                else:
                    # For models without get_input_embeddings, use embedding layer directly
                    embeddings = self.model.model.embed_tokens(input_ids)
                
                pooled_embeddings = embeddings.mean(dim=1)
                
                if self.gating_network.gate[0].weight.device != pooled_embeddings.device:
                    self.gating_network = self.gating_network.to(pooled_embeddings.device)
                
                expert_weights = self.gating_network(pooled_embeddings)

        return expert_weights


class MoEDataset(Dataset):
    """Dataset class for MoE training."""

    def __init__(self, data_path: str, tokenizer, sample_rate: float = 0.9, seed: int = 42):
        import random
        self.data = []
        self.tokenizer = tokenizer

        with open(data_path, "r", encoding="utf-8") as f:
            all_items = [json.loads(line.strip()) for line in f]
        
        random.seed(seed)
        sample_size = int(len(all_items) * sample_rate)
        self.data = random.sample(all_items, sample_size) if sample_size < len(all_items) else all_items

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

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        labels = input_ids.copy()

        source_encoding = self.tokenizer(
            source_text,
            max_length=512,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        source_len = len(source_encoding["input_ids"])

        labels[:source_len] = [-100] * source_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "expert_type": expert_type,
        }

def create_moe_model(config):
    """Create MoE model with multiple experts."""
    logger.info("Loading base model and tokenizer...")

    tokenizer_config = getattr(config, 'tokenizer', {})
    model_prefix = tokenizer_config.get('model_prefix', 'spm_model')
    spm_model_path = f"{model_prefix}.model"
    
    if os.path.exists(spm_model_path):
        custom_tokenizer = SPMTokenizer(model_path=spm_model_path, config=tokenizer_config)
        logger.info(f"Loaded existing SentencePiece model from {spm_model_path}")
    else:
        logger.info("Training new SentencePiece model...")
        from prepare_tokenizer_data import prepare_tokenizer_data
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "..", "configs", "moe_config.yaml")
        tokenizer_data_dir = os.path.join(current_dir, "..", "tokenizer_data")
        data_files = prepare_tokenizer_data(config_path, tokenizer_data_dir)
        if not data_files:
            raise ValueError("Failed to prepare training data for SentencePiece tokenizer")
        custom_tokenizer = create_spm_tokenizer(
            config=tokenizer_config,
            data_files=data_files,
            model_path=None
        )
    
    model_name = config.model["model_id_or_path"]
    max_seq_length = config.data.get("max_length", 2048)
    
    # Always use custom_tokenizer
    model, _ = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,
        device_map="auto" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else None,
    )
    tokenizer = custom_tokenizer
    
    # SỬA LỖI: Xóa bỏ tham số `task_type` vì Unsloth sẽ tự động xác định nó.
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(config.lora["r"]),
        lora_alpha=int(config.lora["lora_alpha"]),
        lora_dropout=float(config.lora["lora_dropout"]),
        bias=config.lora["bias"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    if config.training.get("bf16", False):
        model = model.to(dtype=torch.bfloat16)

    moe_router = MoEExpertRouter(model, tokenizer, num_experts=3)
    
    model = model.to(DEVICE)
    moe_router = moe_router.to(DEVICE)

    logger.info("MoE model created successfully")
    return model, tokenizer, moe_router

def train_moe_model():
    # If using multiple GPUs, recommend launching with torchrun
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info("Multi-GPU detected. Please launch with: torchrun --nproc_per_node=2 vlsp_moe/scripts/train_moe.py")
    """Main training function for MoE model."""
    logger.info("Starting MoE training...")
    config = Config(os.path.join(PROJECT_ROOT, "vlsp_moe", "configs", "moe_config.yaml"))

    # Initialize wandb and log config
    wandb.init(project="moe_medmt", name=config.training.get("run_name", "moe_run"))
    wandb.config.update(config.__dict__, allow_val_change=True)

    model, tokenizer, moe_router = create_moe_model(config)

    logger.info("Loading training dataset...")
    train_file = config.dataset["train_file"]
    if not os.path.isabs(train_file):
        train_file = os.path.join(PROJECT_ROOT, train_file)
    train_dataset = MoEDataset(train_file, tokenizer)
    
    logger.info("Loading validation dataset...")
    val_file = config.dataset["val_file"]
    if not os.path.isabs(val_file):
        val_file = os.path.join(PROJECT_ROOT, val_file)
    eval_dataset = MoEDataset(val_file, tokenizer)

    tokenizer_config = getattr(config, 'tokenizer', {})
    tokenizer_type = tokenizer_config.get('type', 'huggingface')
    
    if tokenizer_type == 'sentencepiece':
        data_collator = SPMDataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=False, 
            pad_to_multiple_of=8
        )
    else:
        # SỬA LỖI: Đảm bảo import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=False, 
            pad_to_multiple_of=8  
        )
    
    # SỬA LỖI: Tạo đối tượng TrainingArguments từ dictionary config
    training_args_dict = config.training.copy()
    # Chuyển đổi kiểu dữ liệu cho các tham số cần thiết
    training_args_dict["learning_rate"] = float(training_args_dict["learning_rate"])
    training_args_dict["weight_decay"] = float(training_args_dict["weight_decay"])
    training_args_dict["warmup_steps"] = int(training_args_dict["warmup_steps"])
    training_args_dict["max_steps"] = int(training_args_dict["max_steps"])
    training_args_dict["logging_steps"] = int(training_args_dict["logging_steps"])
    training_args_dict["save_steps"] = int(training_args_dict["save_steps"])
    training_args_dict["eval_steps"] = int(training_args_dict["eval_steps"])
    training_args_dict["per_device_train_batch_size"] = int(training_args_dict["per_device_train_batch_size"])
    training_args_dict["per_device_eval_batch_size"] = int(training_args_dict["per_device_eval_batch_size"])
    training_args_dict["gradient_accumulation_steps"] = int(training_args_dict["gradient_accumulation_steps"])
    training_args_dict["save_total_limit"] = int(training_args_dict["save_total_limit"])
    training_args_dict["num_train_epochs"] = int(training_args_dict["num_train_epochs"])
    
    # Thêm output_dir nếu chưa có
    if "output_dir" not in training_args_dict:
        training_args_dict["output_dir"] = "./results"

    training_args = TrainingArguments(**training_args_dict)

    def compute_bleu(eval_preds):
        predictions, labels = eval_preds
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_labels = [[label] for label in decoded_labels]
        bleu = corpus_bleu(decoded_preds, decoded_labels)
        return {"bleu": bleu.score}

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_bleu,
        max_seq_length=config.data.get("max_length", 2048),
        # device_map="auto" is recommended for multi-GPU
        device_map="auto" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else None,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving model...")
    trainer.save_model()

    logger.info("MoE training completed!")


if __name__ == "__main__":
    train_moe_model()