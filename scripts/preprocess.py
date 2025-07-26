from datasets import Dataset
from config import Config
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

config = Config(os.path.join(PROJECT_ROOT, "configs", "moe_config.yaml"))

with open(os.path.join(PROJECT_ROOT, config.data["train_path"])) as f:
    src_lines = f.read().splitlines()
with open(os.path.join(PROJECT_ROOT, config.data["target_path"])) as f:
    tgt_lines = f.read().splitlines()

raw_data = [{"translation": {"en": en, "vi": vi}} for en, vi in zip(src_lines, tgt_lines)]
dataset = Dataset.from_list(raw_data)
dataset.save_to_disk(os.path.join(PROJECT_ROOT, config.data["tokenized"]))
print("✅ Raw dataset saved to disk.")
