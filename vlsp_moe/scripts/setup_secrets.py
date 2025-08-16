#!/usr/bin/env python3
"""
Setup script to help users create their secrets file.
"""
import os
import json
import getpass
from pathlib import Path


def setup_secrets_file():
    """Interactive setup for creating a secrets file."""
    
    home_dir = Path.home()
    secrets_file = home_dir / ".vlsp_secrets.json"
    
    print("🔐 VLSP Secrets Setup")
    print("=" * 50)
    print(f"This will create a secrets file at: {secrets_file}")
    print("You can skip any fields by pressing Enter without typing anything.\n")
    
    # Check if file already exists
    if secrets_file.exists():
        overwrite = input(f"Secrets file already exists. Overwrite? (y/n): ").lower().strip()
        if overwrite not in ['y', 'yes']:
            print("Setup cancelled.")
            return
    
    secrets = {}
    
    # W&B Setup
    print("📊 Weights & Biases (W&B) Setup")
    print("-" * 30)
    wandb_key = getpass.getpass("Enter your W&B API key (from https://wandb.ai/authorize): ")
    if wandb_key:
        secrets["WANDB_API_KEY"] = wandb_key
        
        project = input("Enter W&B project name (default: vlsp-medmt): ").strip()
        secrets["WANDB_PROJECT"] = project if project else "vlsp-medmt"
        
        entity = input("Enter W&B entity/team (optional): ").strip()
        if entity:
            secrets["WANDB_ENTITY"] = entity
    
    # HuggingFace Setup
    print("\n🤗 HuggingFace Setup")
    print("-" * 20)
    hf_token = getpass.getpass("Enter your HuggingFace token (from https://huggingface.co/settings/tokens): ")
    if hf_token:
        secrets["HUGGINGFACE_TOKEN"] = hf_token
    
    # OpenAI Setup
    print("\n🧠 OpenAI Setup")
    print("-" * 15)
    openai_key = getpass.getpass("Enter your OpenAI API key (optional): ")
    if openai_key:
        secrets["OPENAI_API_KEY"] = openai_key
    
    # Save the secrets file
    try:
        with open(secrets_file, 'w') as f:
            json.dump(secrets, f, indent=2)
        
        # Set secure permissions (only owner can read/write)
        os.chmod(secrets_file, 0o600)
        
        print(f"\n✅ Secrets file created successfully at: {secrets_file}")
        print(f"📁 File permissions set to 600 (owner read/write only)")
        print(f"🔢 Saved {len(secrets)} secrets")
        
        if secrets:
            print("\n📋 Saved secrets:")
            for key in secrets.keys():
                print(f"  ✓ {key}")
        
        print("\n💡 Usage in your code:")
        print("from secret_loader import setup_wandb_quick")
        print("setup_wandb_quick(disabled=False)")
        
    except Exception as e:
        print(f"❌ Error creating secrets file: {e}")


def show_current_secrets():
    """Show what secrets are currently stored (without values)."""
    home_dir = Path.home()
    secrets_file = home_dir / ".vlsp_secrets.json"
    
    if not secrets_file.exists():
        print("❌ No secrets file found.")
        return
    
    try:
        with open(secrets_file, 'r') as f:
            secrets = json.load(f)
        
        print(f"📁 Secrets file: {secrets_file}")
        print(f"🔢 Number of secrets: {len(secrets)}")
        
        if secrets:
            print("\n📋 Available secrets:")
            for key in secrets.keys():
                print(f"  ✓ {key}")
        else:
            print("📭 No secrets stored.")
            
    except Exception as e:
        print(f"❌ Error reading secrets file: {e}")


def main():
    """Main interactive menu."""
    while True:
        print("\n🔐 VLSP Secrets Manager")
        print("=" * 25)
        print("1. Setup new secrets file")
        print("2. Show current secrets")
        print("3. Exit")
        
        choice = input("\nChoose an option (1-3): ").strip()
        
        if choice == "1":
            setup_secrets_file()
        elif choice == "2":
            show_current_secrets()
        elif choice == "3":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
