import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import logging
import pandas as pd
from datasets import Dataset
from scripts.unslothsft import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from config import Config
from spm_tokenizer import SPMTokenizer, create_spm_tokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Auto-select device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    logger.info("Using GPU: %s", torch.cuda.get_device_name(DEVICE))
else:
    DEVICE = torch.device("cpu")
    logger.info("Using CPU")


class MoEGatingNetwork(nn.Module):
    """Gating network for MoE that routes inputs to appropriate experts."""

    def __init__(self, input_size: int, num_experts: int, hidden_size: int = 256, dtype=torch.float32):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Sequential(
            nn.Linear(input_size, hidden_size, dtype=dtype),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_experts, dtype=dtype),
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

        # Initialize gating network
        # Use model's hidden size for gating
        hidden_size = (
            model.config.hidden_size if hasattr(model.config, "hidden_size") else 1024
        )
        
        # Detect model dtype
        model_dtype = next(model.parameters()).dtype
        
        self.gating_network = MoEGatingNetwork(hidden_size, num_experts, dtype=model_dtype)

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
                    self.gating_network = self.gating_network.to(pooled_embeddings.device, dtype=pooled_embeddings.dtype)
                
                expert_weights = self.gating_network(pooled_embeddings)

        return expert_weights


class UnslothMoETrainer(SFTTrainer):
    """Custom Unsloth trainer for MoE model."""

    def __init__(self, moe_router, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.moe_router = moe_router

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss with expert routing."""
        # Extract expert information from the text if available
        expert_types = self.extract_expert_types_from_batch(inputs)
        
        # Get input_ids from tokenized inputs
        if 'input_ids' in inputs:
            input_ids = inputs['input_ids']
        else:
            # If inputs are text, tokenize them
            texts = inputs.get('text', inputs.get('input', []))
            if isinstance(texts, list):
                tokenized = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
                input_ids = tokenized['input_ids'].to(model.device)
            else:
                input_ids = None
        
        if input_ids is not None:
            expert_weights = self.moe_router.get_expert_weights(input_ids, expert_types)
        else:
            expert_weights = None
        
        # Compute standard loss
        outputs = super().compute_loss(model, inputs, return_outputs=True)
        loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss

        # Add routing loss if expert weights are available
        if expert_weights is not None:
            routing_loss = self.compute_routing_loss(expert_weights)
            loss = loss + 0.01 * routing_loss

        return (loss, outputs[1]) if return_outputs and isinstance(outputs, tuple) else loss

    def extract_expert_types_from_batch(self, inputs):
        """Extract expert types from input text."""
        # This is a simplified extraction - in practice, you might want to
        # parse the text more carefully or include expert type in metadata
        expert_types = []
        
        texts = inputs.get('text', inputs.get('input', []))
        if not isinstance(texts, list):
            if isinstance(texts, str):
                texts = [texts]
            else:
                return None
                
        for text in texts:
            if isinstance(text, str):
                # Simple heuristic to determine expert type
                if any(keyword in text.lower() for keyword in ['medical', 'doctor', 'patient', 'medicine']):
                    expert_types.append('medical')
                elif 'vietnamese:' in text.lower() or 'tiếng việt:' in text.lower():
                    expert_types.append('en_vi')
                elif 'english:' in text.lower() or 'tiếng anh:' in text.lower():
                    expert_types.append('vi_en')
                else:
                    expert_types.append('medical')  # default
            else:
                expert_types.append('medical')  # default
                
        return expert_types if expert_types else None

    def compute_routing_loss(self, expert_weights):
        """Compute routing loss to encourage sparsity."""
        batch_size = expert_weights.size(0)
        expert_usage = expert_weights.sum(dim=0)
        target_usage = batch_size / self.moe_router.num_experts
        load_loss = F.mse_loss(
            expert_usage, torch.full_like(expert_usage, target_usage)
        )
        return load_loss


def prepare_moe_dataset(data_path: str, tokenizer):
    """Prepare dataset for MoE training with Unsloth format."""
    conversations = []
    
    logger.info(f"Loading data from {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            expert_type = item.get("expert", "medical")
            messages = item["messages"]
            
            # Format as conversation
            source_text = messages[0]["content"]
            target_text = messages[1]["content"]
            
            # Add expert type indicator to the conversation
            formatted_text = f"Expert: {expert_type}\nUser: {source_text}\nAssistant: {target_text}"
            conversations.append(formatted_text)
    
    # Convert to pandas Series and then to Dataset
    data = pd.Series(conversations)
    data.name = "text"
    
    combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
    combined_dataset = combined_dataset.shuffle(seed=3407)
    
    logger.info(f"Prepared {len(combined_dataset)} samples")
    return combined_dataset


def create_unsloth_moe_model(config):
    """Create MoE model using Unsloth."""
    logger.info("Creating Unsloth MoE model...")
    
    # Get model configuration
    model_name = config.model["model_id_or_path"]
    max_seq_length = config.model.get("max_length", 2048)
    
    logger.info(f"Loading model: {model_name}")
    
    # Load model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,  # Use 4bit for memory efficiency
        load_in_8bit=False,
        full_finetuning=False,
    )
    
    # Configure LoRA with Unsloth optimizations
    lora_config = config.lora
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config.get("r", 32),
        target_modules=lora_config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        lora_alpha=lora_config.get("lora_alpha", 32),
        lora_dropout=lora_config.get("lora_dropout", 0),
        bias=lora_config.get("bias", "none"),
        use_gradient_checkpointing="unsloth",  # Unsloth optimization
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    # Create MoE router
    moe_router = MoEExpertRouter(model, tokenizer, num_experts=3)
    
    # Move to device with correct dtype
    model_dtype = next(model.parameters()).dtype
    moe_router = moe_router.to(DEVICE, dtype=model_dtype)
    
    logger.info("Unsloth MoE model created successfully")
    return model, tokenizer, moe_router


def train_moe_unsloth():
    """Main training function for MoE model using Unsloth."""
    logger.info("Starting MoE training with Unsloth...")
    
    # Load configuration
    config = Config(os.path.join(PROJECT_ROOT, "configs", "moe_config.yaml"))
    
    # Create model
    model, tokenizer, moe_router = create_unsloth_moe_model(config)
    
    # Prepare datasets
    train_file = config.dataset["train_file"]
    if not os.path.isabs(train_file):
        train_file = os.path.join(PROJECT_ROOT, train_file)
    
    val_file = config.dataset["val_file"]
    if not os.path.isabs(val_file):
        val_file = os.path.join(PROJECT_ROOT, val_file)
    
    train_dataset = prepare_moe_dataset(train_file, tokenizer)
    eval_dataset = prepare_moe_dataset(val_file, tokenizer)
    
    # Show memory stats
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        logger.info(f"{start_gpu_memory} GB of memory reserved.")
    
    # Configure training arguments
    training_config = config.training.copy()
    
    # Create SFTConfig for Unsloth
    sft_config = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        warmup_steps=training_config.get("warmup_steps", 5),
        max_steps=training_config.get("max_steps", 100),
        learning_rate=float(training_config.get("learning_rate", 2e-4)),
        logging_steps=training_config.get("logging_steps", 1),
        optim="adamw_8bit",  # Unsloth optimization
        weight_decay=float(training_config.get("weight_decay", 0.01)),
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=training_config.get("output_dir", "./results"),
        save_steps=training_config.get("save_steps", 50),
        eval_steps=training_config.get("eval_steps", 50),
        save_total_limit=training_config.get("save_total_limit", 2),
        report_to="none",  # Disable wandb for now
        remove_unused_columns=False,  # Keep all columns for MoE routing
    )
    
    # Create custom trainer
    trainer = UnslothMoETrainer(
        moe_router=moe_router,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
    )
    
    logger.info("Starting training...")
    trainer_stats = trainer.train()
    
    # Show final memory and time stats
    if torch.cuda.is_available():
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        
        logger.info(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        logger.info(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
        logger.info(f"Peak reserved memory = {used_memory} GB.")
        logger.info(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        logger.info(f"Peak reserved memory % of max memory = {used_percentage} %.")
        logger.info(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
    
    # Save model
    logger.info("Saving model...")
    output_dir = sft_config.output_dir
    
    # Save LoRA adapter
    model.save_pretrained(f"{output_dir}/lora_model")
    tokenizer.save_pretrained(f"{output_dir}/lora_model")
    
    # Save MoE router state
    torch.save(moe_router.state_dict(), f"{output_dir}/moe_router.pt")
    
    logger.info("MoE training with Unsloth completed!")
    return model, tokenizer, moe_router


def test_inference(model, tokenizer, moe_router):
    """Test inference with the trained MoE model."""
    logger.info("Testing inference...")
    
    # Test messages for different experts
    test_messages = [
        {
            "expert": "medical",
            "text": "What are the symptoms of diabetes?"
        },
        {
            "expert": "en_vi", 
            "text": "Translate to Vietnamese: Good morning, how are you?"
        },
        {
            "expert": "vi_en",
            "text": "Translate to English: Xin chào, bạn khỏe không?"
        }
    ]
    
    for test_msg in test_messages:
        logger.info(f"\nTesting {test_msg['expert']} expert:")
        logger.info(f"Input: {test_msg['text']}")
        
        # Format for generation
        messages = [{"role": "user", "content": test_msg['text']}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Generate response
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Output: {response}")


if __name__ == "__main__":
    try:
        model, tokenizer, moe_router = train_moe_unsloth()
        
        # Optional: Test inference
        test_inference(model, tokenizer, moe_router)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise 