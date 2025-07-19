from datasets import Dataset
from config import Config

config = Config("../configs/moe_config.yaml")

with open(config.data["train_path"]) as f:
    src_lines = f.read().splitlines()
with open(config.data["target_path"]) as f:
    tgt_lines = f.read().splitlines()

raw_data = [{"translation": {"en": en, "vi": vi}} for en, vi in zip(src_lines, tgt_lines)]
dataset = Dataset.from_list(raw_data)
dataset.save_to_disk(config.data["tokenized"])
print("✅ Raw dataset saved to disk.")
