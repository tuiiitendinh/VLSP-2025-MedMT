"""
Secret loader for handling API keys and tokens securely in notebooks.
"""
import os
import getpass
from pathlib import Path
import json


class SecretLoader:
    """Handles loading of secrets from environment variables or user input."""
    
    def __init__(self, secrets_file=None):
        """
        Initialize the secret loader.
        
        Args:
            secrets_file: Optional path to a JSON file containing secrets
        """
        self.secrets_file = secrets_file or Path.home() / ".vlsp_secrets.json"
        self.secrets = self._load_secrets()
    
    def _load_secrets(self):
        """Load secrets from file if it exists."""
        if self.secrets_file.exists():
            try:
                with open(self.secrets_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}
    
    def _save_secrets(self):
        """Save secrets to file."""
        self.secrets_file.parent.mkdir(exist_ok=True)
        with open(self.secrets_file, 'w') as f:
            json.dump(self.secrets, f, indent=2)
        # Make file readable only by owner
        os.chmod(self.secrets_file, 0o600)
    
    def get_secret(self, key, prompt=None, required=True):
        """
        Get a secret value from environment, file, or user input.
        
        Args:
            key: The secret key name
            prompt: Custom prompt for user input
            required: Whether the secret is required
            
        Returns:
            The secret value or None if not required and not found
        """
        # Check environment variable first
        env_value = os.getenv(key)
        if env_value:
            return env_value
        
        # Check stored secrets
        if key in self.secrets:
            return self.secrets[key]
        
        # If not required and not found, return None
        if not required:
            return None
        
        # Prompt user for input
        if prompt is None:
            prompt = f"Enter {key}: "
        
        value = getpass.getpass(prompt)
        if value:
            # Ask if user wants to save it
            save_choice = input(f"Save {key} for future use? (y/n): ").lower().strip()
            if save_choice in ['y', 'yes']:
                self.secrets[key] = value
                self._save_secrets()
                print(f"Secret saved to {self.secrets_file}")
        
        return value
    
    def set_secret(self, key, value, save_to_file=True):
        """
        Set a secret value.
        
        Args:
            key: The secret key name
            value: The secret value
            save_to_file: Whether to save to file
        """
        self.secrets[key] = value
        if save_to_file:
            self._save_secrets()
    
    def setup_wandb(self, project_name=None, entity=None, disabled=False):
        """
        Setup Weights & Biases with proper authentication.
        
        Args:
            project_name: W&B project name
            entity: W&B entity/team name
            disabled: Whether to disable W&B entirely
        """
        if disabled:
            os.environ["WANDB_DISABLED"] = "true"
            return
        
        # Get W&B API key
        wandb_key = self.get_secret(
            "WANDB_API_KEY", 
            "Enter your Weights & Biases API key: ",
            required=False
        )
        
        if wandb_key:
            os.environ["WANDB_API_KEY"] = wandb_key
            print("✓ W&B API key set")
        else:
            os.environ["WANDB_DISABLED"] = "true"
            print("W&B disabled - no API key provided")
            return
        
        if project_name:
            os.environ["WANDB_PROJECT"] = project_name
        
        if entity:
            os.environ["WANDB_ENTITY"] = entity
        
        # Set non-interactive mode
        os.environ["WANDB_MODE"] = "online"  
        
        print(f"✓ W&B configured for project: {project_name or 'default'}")


# Convenience function for quick setup
def setup_wandb_quick(project_name="vlsp-medmt", disabled=False):
    """Quick setup for W&B in notebooks."""
    loader = SecretLoader()
    loader.setup_wandb(project_name=project_name, disabled=disabled)
    return loader
