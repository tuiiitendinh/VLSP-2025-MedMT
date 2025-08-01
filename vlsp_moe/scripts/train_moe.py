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
from transformers import TrainingArguments, DataCollatorForLanguageModeling
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


# --- Model Creation ---
def create_model(config):
    logger.info("Loading base model and tokenizer with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model["model_id_or_path"],
        max_seq_length=config.data.get("max_length", 2048),
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    special_tokens_to_add = ["<medical>", "<en_vi>", "<vi_en>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})
    model.resize_token_embeddings(len(tokenizer))
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(config.lora["r"]),
        lora_alpha=int(config.lora["lora_alpha"]),
        lora_dropout=float(config.lora["lora_dropout"]),
        bias=config.lora["bias"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
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

    trainer = CustomSFTTrainer(
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