#!/usr/bin/env python3
"""
Script to train SentencePiece tokenizer for the MoE model.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the scripts directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from config import Config
from spm_tokenizer import create_spm_tokenizer
from prepare_tokenizer_data import prepare_tokenizer_data

logger = logging.getLogger(__name__)


def train_tokenizer(config_path: str, output_dir: str, force_retrain: bool = False):
    """Train SentencePiece tokenizer from config."""
    
    # Load config
    config = Config(config_path)
    tokenizer_config = getattr(config, 'tokenizer', {})
    
    if tokenizer_config.get('type') != 'sentencepiece':
        logger.error("Config does not specify SentencePiece tokenizer")
        return False
    
    # Setup paths
    os.makedirs(output_dir, exist_ok=True)
    model_prefix = tokenizer_config.get('model_prefix', 'spm_model')
    model_path = os.path.join(output_dir, f"{model_prefix}.model")
    
    # Check if model already exists
    if os.path.exists(model_path) and not force_retrain:
        logger.info(f"SentencePiece model already exists at {model_path}")
        logger.info("Use --force to retrain")
        return True
    
    # Prepare training data
    logger.info("Preparing training data...")
    tokenizer_data_dir = os.path.join(output_dir, "training_data")
    data_files = prepare_tokenizer_data(config_path, tokenizer_data_dir)
    
    if not data_files:
        logger.error("Failed to prepare training data")
        return False
    
    # Update config with output directory
    tokenizer_config['output_dir'] = output_dir
    
    # Train tokenizer
    logger.info("Training SentencePiece tokenizer...")
    try:
        tokenizer = create_spm_tokenizer(
            config=tokenizer_config,
            data_files=data_files,
            model_path=None
        )
        
        logger.info(f"✅ SentencePiece tokenizer trained successfully!")
        logger.info(f"Model saved to: {tokenizer.model_path}")
        logger.info(f"Vocabulary size: {tokenizer.get_vocab_size()}")
        
        # Test the tokenizer
        test_texts = [
            "Hello, this is a test sentence.",
            "Xin chào, đây là câu thử nghiệm.",
            "Medical terminology: diabetes, hypertension, cardiovascular disease."
        ]
        
        logger.info("\nTesting tokenizer:")
        for text in test_texts:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded)
            logger.info(f"Original: {text}")
            logger.info(f"Encoded:  {encoded}")
            logger.info(f"Decoded:  {decoded}")
            logger.info("-" * 50)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to train tokenizer: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer for MoE model")
    parser.add_argument("--config", type=str, 
                       default="../configs/moe_config.yaml",
                       help="Path to config YAML file")
    parser.add_argument("--output_dir", type=str, 
                       default="../tokenizer_output",
                       help="Output directory for trained tokenizer")
    parser.add_argument("--force", action="store_true",
                       help="Force retrain even if model exists")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Convert relative paths to absolute
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.isabs(args.config):
        args.config = os.path.join(script_dir, args.config)
    
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(script_dir, args.output_dir)
    
    logger.info("=" * 60)
    logger.info("SentencePiece Tokenizer Training")
    logger.info("=" * 60)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Force retrain: {args.force}")
    
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        return 1
    
    try:
        success = train_tokenizer(args.config, args.output_dir, args.force)
        if success:
            logger.info("🎉 Tokenizer training completed successfully!")
            return 0
        else:
            logger.error("❌ Tokenizer training failed!")
            return 1
    except Exception as e:
        logger.error(f"Error during tokenizer training: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
