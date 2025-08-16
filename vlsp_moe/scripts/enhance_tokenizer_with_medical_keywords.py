#!/usr/bin/env python3
"""
Script to enhance tokenizer training data with medical keywords.
This ensures the tokenizer learns medical-specific terms.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Add the scripts directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from config import Config

logger = logging.getLogger(__name__)


def enhance_tokenizer_data_with_medical_keywords(tokenizer_data_file: str, medical_keywords_file: str, output_file: str):
    """Enhance tokenizer training data with medical keywords."""
    
    # Load medical keywords
    if not os.path.exists(medical_keywords_file):
        logger.warning(f"Medical keywords file not found: {medical_keywords_file}")
        return False
    
    try:
        with open(medical_keywords_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            medical_keywords = data.get('medical_keywords', [])
    except Exception as e:
        logger.error(f"Failed to load medical keywords: {e}")
        return False
    
    if not medical_keywords:
        logger.warning("No medical keywords found")
        return False
    
    # Read existing tokenizer data
    if not os.path.exists(tokenizer_data_file):
        logger.error(f"Tokenizer data file not found: {tokenizer_data_file}")
        return False
    
    with open(tokenizer_data_file, 'r', encoding='utf-8') as f:
        existing_data = f.read()
    
    # Create enhanced data with medical keywords
    enhanced_data = existing_data + "\n"
    
    # Add medical keywords as individual lines to ensure they're learned
    for keyword in medical_keywords:
        if len(keyword.strip()) > 0:
            enhanced_data += keyword.strip() + "\n"
    
    # Add medical phrases and sentences
    medical_phrases = [
        "medical terminology",
        "patient diagnosis",
        "treatment plan",
        "clinical symptoms",
        "healthcare provider",
        "medical examination",
        "prescription medication",
        "surgical procedure",
        "laboratory test",
        "medical consultation"
    ]
    
    for phrase in medical_phrases:
        enhanced_data += phrase + "\n"
    
    # Write enhanced data
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(enhanced_data)
    
    logger.info(f"Enhanced tokenizer data with {len(medical_keywords)} medical keywords")
    logger.info(f"Enhanced data saved to: {output_file}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Enhance tokenizer training data with medical keywords")
    parser.add_argument("--tokenizer_data", type=str, required=True,
                       help="Path to original tokenizer training data file")
    parser.add_argument("--medical_keywords", type=str, required=True,
                       help="Path to medical keywords JSON file")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to enhanced tokenizer training data file")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Enhancing tokenizer training data with medical keywords...")
    
    try:
        success = enhance_tokenizer_data_with_medical_keywords(
            args.tokenizer_data, 
            args.medical_keywords, 
            args.output
        )
        
        if success:
            logger.info("✅ Successfully enhanced tokenizer data with medical keywords!")
            return 0
        else:
            logger.error("❌ Failed to enhance tokenizer data")
            return 1
            
    except Exception as e:
        logger.error(f"Error enhancing tokenizer data: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 