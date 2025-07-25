#!/usr/bin/env python3
"""
Script to switch tokenizer configuration between HuggingFace and SentencePiece.
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


def update_config_tokenizer(config_path: str, tokenizer_type: str, backup: bool = True):
    """Update config file to use specified tokenizer type."""
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return False
    
    # Backup original config
    if backup:
        backup_path = f"{config_path}.backup"
        import shutil
        shutil.copy2(config_path, backup_path)
        logger.info(f"Backed up config to: {backup_path}")
    
    # Load current config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if tokenizer_type == 'sentencepiece':
        # Set SentencePiece configuration
        config['tokenizer'] = {
            'type': 'sentencepiece',
            'vocab_size': 32000,
            'model_prefix': 'spm_model',
            'input_sentence_size': 10000000,
            'character_coverage': 0.995,
            'model_type': 'unigram',
            'split_by_unicode_script': True,
            'split_by_number': True,
            'split_by_whitespace': True,
            'treat_whitespace_as_suffix': False,
            'allow_whitespace_only_pieces': True,
            'split_digits': True,
            'pad_id': 0,
            'unk_id': 1,
            'bos_id': 2,
            'eos_id': 3,
            'user_defined_symbols': ["<|im_start|>", "<|im_end|>", "<medical>", "<en_vi>", "<vi_en>"]
        }
        logger.info("✅ Updated config to use SentencePiece tokenizer")
        
    elif tokenizer_type == 'huggingface':
        # Remove SentencePiece configuration
        if 'tokenizer' in config:
            del config['tokenizer']
        logger.info("✅ Updated config to use HuggingFace tokenizer")
        
    else:
        logger.error(f"Unknown tokenizer type: {tokenizer_type}")
        return False
    
    # Save updated config
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    logger.info(f"Config updated: {config_path}")
    return True


def show_current_config(config_path: str):
    """Show current tokenizer configuration."""
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info("=" * 50)
    logger.info("Current Tokenizer Configuration")
    logger.info("=" * 50)
    
    if 'tokenizer' in config:
        tokenizer_config = config['tokenizer']
        tokenizer_type = tokenizer_config.get('type', 'unknown')
        logger.info(f"Type: {tokenizer_type}")
        
        if tokenizer_type == 'sentencepiece':
            logger.info(f"Vocabulary size: {tokenizer_config.get('vocab_size', 'unknown')}")
            logger.info(f"Model type: {tokenizer_config.get('model_type', 'unknown')}")
            logger.info(f"Character coverage: {tokenizer_config.get('character_coverage', 'unknown')}")
            logger.info(f"Special symbols: {tokenizer_config.get('user_defined_symbols', [])}")
    else:
        logger.info("Type: huggingface (default)")
        logger.info("Using model's default tokenizer")


def validate_sentencepiece_setup(config_path: str, tokenizer_dir: str = None):
    """Validate SentencePiece setup."""
    
    logger.info("=" * 50)
    logger.info("Validating SentencePiece Setup")
    logger.info("=" * 50)
    
    # Check if sentencepiece is installed
    try:
        import sentencepiece
        logger.info("✅ SentencePiece package is installed")
    except ImportError:
        logger.error("❌ SentencePiece package not found. Install with: pip install sentencepiece")
        return False
    
    # Check config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if 'tokenizer' not in config or config['tokenizer'].get('type') != 'sentencepiece':
        logger.warning("⚠️ Config is not set to use SentencePiece tokenizer")
        return False
    
    logger.info("✅ Config is set to use SentencePiece tokenizer")
    
    # Check for trained model
    if tokenizer_dir:
        model_prefix = config['tokenizer'].get('model_prefix', 'spm_model')
        model_path = os.path.join(tokenizer_dir, f"{model_prefix}.model")
        
        if os.path.exists(model_path):
            logger.info(f"✅ Trained SentencePiece model found: {model_path}")
            
            # Try to load the model
            try:
                sp = sentencepiece.SentencePieceProcessor()
                sp.load(model_path)
                vocab_size = sp.get_piece_size()
                logger.info(f"✅ Model loads successfully, vocab size: {vocab_size}")
            except Exception as e:
                logger.error(f"❌ Error loading model: {e}")
                return False
        else:
            logger.warning(f"⚠️ Trained model not found: {model_path}")
            logger.info("Run 'python train_tokenizer.py' to train the tokenizer")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Manage tokenizer configuration")
    parser.add_argument("--config", type=str,
                       default="../configs/moe_config.yaml",
                       help="Path to config file")
    parser.add_argument("--set", type=str, choices=['sentencepiece', 'huggingface'],
                       help="Set tokenizer type")
    parser.add_argument("--show", action="store_true",
                       help="Show current tokenizer configuration")
    parser.add_argument("--validate", action="store_true",
                       help="Validate SentencePiece setup")
    parser.add_argument("--tokenizer_dir", type=str,
                       default="../tokenizer_output",
                       help="Directory containing trained tokenizer")
    parser.add_argument("--no-backup", action="store_true",
                       help="Don't backup config file when updating")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Convert relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.isabs(args.config):
        args.config = os.path.join(script_dir, args.config)
    
    if args.tokenizer_dir and not os.path.isabs(args.tokenizer_dir):
        args.tokenizer_dir = os.path.join(script_dir, args.tokenizer_dir)
    
    logger.info("Tokenizer Configuration Manager")
    logger.info(f"Config file: {args.config}")
    
    success = True
    
    try:
        if args.show:
            show_current_config(args.config)
        
        if args.set:
            success = update_config_tokenizer(args.config, args.set, backup=not args.no_backup)
        
        if args.validate:
            success = validate_sentencepiece_setup(args.config, args.tokenizer_dir) and success
        
        if not any([args.show, args.set, args.validate]):
            # Default action: show current config
            show_current_config(args.config)
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
