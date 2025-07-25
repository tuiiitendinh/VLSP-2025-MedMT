import yaml
import torch
import os


class Config:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        
        self._apply_env_overrides()
        
        self.device_str = self.cfg.get("device", "cpu").lower()
        self.device = self._resolve_device(self.device_str)
        self.model = self.cfg.get("model", {})
        self.tokenizer = self.cfg.get("tokenizer", {})
        self.lora = self.cfg.get("lora", {})
        self.training = self.cfg.get("training", {})
        self.data = self.cfg.get("data", {})
        self.dataset = self.cfg.get("dataset", {})
        self.moe = self.cfg.get("moe", {})
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides to config."""
        # Device override
        if os.getenv("MOE_DEVICE"):
            self.cfg["device"] = os.getenv("MOE_DEVICE")
        
        # Training overrides
        if os.getenv("MOE_LEARNING_RATE"):
            if "training" not in self.cfg:
                self.cfg["training"] = {}
            self.cfg["training"]["learning_rate"] = float(os.getenv("MOE_LEARNING_RATE"))
        
        if os.getenv("MOE_BATCH_SIZE"):
            if "training" not in self.cfg:
                self.cfg["training"] = {}
            self.cfg["training"]["per_device_train_batch_size"] = int(os.getenv("MOE_BATCH_SIZE"))
        
        if os.getenv("MOE_MAX_STEPS"):
            if "training" not in self.cfg:
                self.cfg["training"] = {}
            self.cfg["training"]["max_steps"] = int(os.getenv("MOE_MAX_STEPS"))
        
        if os.getenv("MOE_OUTPUT_DIR"):
            if "training" not in self.cfg:
                self.cfg["training"] = {}
            self.cfg["training"]["output_dir"] = os.getenv("MOE_OUTPUT_DIR")
        
        # Model overrides
        if os.getenv("MOE_MODEL_PATH"):
            if "model" not in self.cfg:
                self.cfg["model"] = {}
            self.cfg["model"]["model_id_or_path"] = os.getenv("MOE_MODEL_PATH")
        
        # LoRA overrides
        if os.getenv("MOE_LORA_R"):
            if "lora" not in self.cfg:
                self.cfg["lora"] = {}
            self.cfg["lora"]["r"] = int(os.getenv("MOE_LORA_R"))

    def _resolve_device(self, device_name: str):
        if device_name == "cuda" and torch.cuda.is_available():
            print("✅ Using CUDA")
            return torch.device("cuda")
        elif device_name == "mps" and torch.backends.mps.is_available():
            print("✅ Using MPS")
            return torch.device("mps")
        else:
            print("⚠️ Falling back to CPU")
            return torch.device("cpu")

    def get(self, key_path, default=None):
        keys = key_path.split(".")
        val = self.cfg
        for k in keys:
            if not isinstance(val, dict) or k not in val:
                return default
            val = val[k]
        return val
    
    def set(self, key_path, value):
        """Set a configuration value using dot notation."""
        keys = key_path.split(".")
        val = self.cfg
        for k in keys[:-1]:
            if k not in val or not isinstance(val[k], dict):
                val[k] = {}
            val = val[k]
        val[keys[-1]] = value
        
        # Update parsed attributes if they exist
        if keys[0] == "device":
            self.device_str = str(value).lower()
            self.device = self._resolve_device(self.device_str)
        elif keys[0] == "model":
            self.model = self.cfg.get("model", {})
        elif keys[0] == "tokenizer":
            self.tokenizer = self.cfg.get("tokenizer", {})
        elif keys[0] == "lora":
            self.lora = self.cfg.get("lora", {})
        elif keys[0] == "training":
            self.training = self.cfg.get("training", {})
        elif keys[0] == "data":
            self.data = self.cfg.get("data", {})
        elif keys[0] == "dataset":
            self.dataset = self.cfg.get("dataset", {})
        elif keys[0] == "moe":
            self.moe = self.cfg.get("moe", {})
    
    def update(self, overrides: dict):
        """Update multiple configuration values."""
        for key_path, value in overrides.items():
            self.set(key_path, value)
