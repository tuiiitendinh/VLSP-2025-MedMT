# Secrets Management Guide

This directory contains tools for securely managing API keys and tokens in your VLSP project.

## Quick Start

### Option 1: Interactive Setup (Recommended)
```bash
python setup_secrets.py
```
This will guide you through setting up your secrets file interactively.

### Option 2: Manual Setup
1. Copy the sample file to your home directory:
   ```bash
   cp .vlsp_secrets_sample.json ~/.vlsp_secrets.json
   ```

2. Edit the file with your actual API keys:
   ```bash
   # Use your preferred editor
   nano ~/.vlsp_secrets.json
   # or
   code ~/.vlsp_secrets.json
   ```

3. Set secure permissions:
   ```bash
   chmod 600 ~/.vlsp_secrets.json
   ```

## Usage in Notebooks

### For W&B (Weights & Biases)
```python
# Option 1: Disable W&B (no prompts)
import os
os.environ["WANDB_DISABLED"] = "true"

# Option 2: Use W&B with automatic setup
from secret_loader import setup_wandb_quick
setup_wandb_quick(project_name="vlsp-medmt", disabled=False)
```

### For Other APIs
```python
from secret_loader import SecretLoader

loader = SecretLoader()
hf_token = loader.get_secret("HUGGINGFACE_TOKEN")
openai_key = loader.get_secret("OPENAI_API_KEY", required=False)
```

## Files Overview

- **`secret_loader.py`** - Main secrets management class
- **`setup_secrets.py`** - Interactive setup script
- **`.vlsp_secrets_sample.json`** - Sample secrets file template
- **`README_secrets.md`** - This guide

## Security Notes

- ✅ Secrets file is stored in your home directory (`~/.vlsp_secrets.json`)
- ✅ File permissions are automatically set to 600 (owner read/write only)
- ✅ Never commit secrets files to git
- ✅ Environment variables take precedence over stored secrets
- ✅ All user input for secrets uses `getpass` (hidden input)

## Supported Secrets

| Key | Description | Where to Get |
|-----|-------------|--------------|
| `WANDB_API_KEY` | Weights & Biases API key | https://wandb.ai/authorize |
| `HUGGINGFACE_TOKEN` | HuggingFace access token | https://huggingface.co/settings/tokens |
| `OPENAI_API_KEY` | OpenAI API key | https://platform.openai.com/api-keys |
| `ANTHROPIC_API_KEY` | Anthropic API key | https://console.anthropic.com/ |

## Troubleshooting

### W&B Still Prompting?
Make sure you run one of these before training:
```python
# Disable completely
os.environ["WANDB_DISABLED"] = "true"

# Or setup with your API key
setup_wandb_quick(disabled=False)
```

### Secrets Not Loading?
1. Check file location: `ls -la ~/.vlsp_secrets.json`
2. Check file permissions: Should be `-rw-------`
3. Check JSON syntax: `python -m json.tool ~/.vlsp_secrets.json`

### Permission Errors?
```bash
chmod 600 ~/.vlsp_secrets.json
```

## Environment Variables

You can also set secrets as environment variables (they take precedence):
```bash
export WANDB_API_KEY="your_key_here"
export WANDB_PROJECT="vlsp-medmt"
```
