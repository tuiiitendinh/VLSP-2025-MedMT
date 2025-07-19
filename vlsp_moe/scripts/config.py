import yaml
import torch


class Config:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.device_str = self.cfg.get("device", "cpu").lower()
        self.device = self._resolve_device(self.device_str)
        self.model = self.cfg.get("model", {})
        self.lora = self.cfg.get("lora", {})
        self.training = self.cfg.get("training", {})
        self.data = self.cfg.get("data", {})
        self.dataset = self.cfg.get("dataset", {})
        self.moe = self.cfg.get("moe", {})

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
