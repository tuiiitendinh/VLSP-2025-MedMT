#!/usr/bin/env python3
"""
Script to prepare data for training SentencePiece tokenizer.
This script extracts text from JSONL files and creates training data.
"""

import json
import os
import argparse
import logging
from pathlib import Path
from config import Config

logger = logging.getLogger(__name__)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def extract_text_from_jsonl(jsonl_file: str, output_file: str):
    """Extract text content from JSONL file for tokenizer training."""
    texts = []
    
    if not os.path.exists(jsonl_file):
        logger.warning(f"File not found: {jsonl_file}")
        return 0
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                if 'messages' in item:
                    # Extract both source and target text
                    for message in item['messages']:
                        if 'content' in message:
                            texts.append(message['content'])
                elif 'text' in item:
                    texts.append(item['text'])
                elif 'source' in item and 'target' in item:
                    texts.append(item['source'])
                    texts.append(item['target'])
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line: {e}")
                continue
    
    # Write extracted texts
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text.strip() + '\n')
    
    logger.info(f"Extracted {len(texts)} texts to {output_file}")
    return len(texts)


def prepare_tokenizer_data(config_path: str, output_dir: str):
    """Prepare training data for SentencePiece tokenizer."""
    config = Config(config_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get dataset files from config
    dataset_files = []
    if hasattr(config, 'dataset'):
        for key, file_path in config.dataset.items():
            if key.endswith('_file') and file_path:
                if os.path.exists(file_path):
                    dataset_files.append(file_path)
                else:
                    logger.warning(f"Dataset file not found: {file_path}")
    
    if not dataset_files:
        logger.error("No valid dataset files found in config")
        return []
    
    # Extract text from each file
    extracted_files = []
    total_texts = 0
    
    for i, jsonl_file in enumerate(dataset_files):
        output_file = os.path.join(output_dir, f"tokenizer_data_{i}.txt")
        count = extract_text_from_jsonl(jsonl_file, output_file)
        if count > 0:
            extracted_files.append(output_file)
            total_texts += count
    
    # Create combined file
    combined_file = os.path.join(output_dir, "combined_tokenizer_data.txt")
    with open(combined_file, 'w', encoding='utf-8') as outf:
        for ext_file in extracted_files:
            with open(ext_file, 'r', encoding='utf-8') as inf:
                outf.write(inf.read())
    
    logger.info(f"Combined {total_texts} texts into {combined_file}")
    
    return [combined_file]


def main():
    parser = argparse.ArgumentParser(description="Prepare data for SentencePiece tokenizer training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")

    default_output_path = os.path.join(PROJECT_ROOT, "tokenizer_data")
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=default_output_path,
        help="Output directory for tokenizer training data"
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Preparing tokenizer training data...")
    
    try:
        extracted_files = prepare_tokenizer_data(args.config, args.output_dir)
        if extracted_files:
            logger.info(f"Successfully prepared tokenizer data: {extracted_files}")
            return 0
        else:
            logger.error("Failed to prepare tokenizer data")
            return 1
    except Exception as e:
        logger.error(f"Error preparing tokenizer data: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
