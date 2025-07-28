#!/usr/bin/env python3
"""
Script to test and evaluate SentencePiece tokenizer performance.
"""

import os
import sys
import argparse
import logging
import json
from typing import List, Dict

# Add the scripts directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from config import Config
from spm_tokenizer import SPMTokenizer
import sentencepiece as spm

logger = logging.getLogger(__name__)


def load_test_data(data_file: str, limit: int = 100) -> List[str]:
    """Load test data from JSONL file."""
    texts = []
    
    if not os.path.exists(data_file):
        logger.warning(f"Test data file not found: {data_file}")
        return []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        count = 0
        for line in f:
            if count >= limit:
                break
            
            try:
                item = json.loads(line.strip())
                if 'messages' in item:
                    for message in item['messages']:
                        if 'content' in message:
                            texts.append(message['content'])
                            count += 1
                            if count >= limit:
                                break
            except json.JSONDecodeError:
                continue
    
    return texts


def test_tokenizer_basic(tokenizer: SPMTokenizer, test_texts: List[str]):
    """Test basic tokenizer functionality."""
    logger.info("=" * 60)
    logger.info("Basic Tokenizer Tests")
    logger.info("=" * 60)
    
    for i, text in enumerate(test_texts[:5]):  # Test first 5 texts
        logger.info(f"\nTest {i+1}:")
        logger.info(f"Original: {text}")
        
        # Encode
        encoded = tokenizer.encode(text, add_special_tokens=True)
        logger.info(f"Encoded ({len(encoded)} tokens): {encoded}")
        
        # Decode
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
        logger.info(f"Decoded: {decoded}")
        
        # Check if roundtrip is successful
        if text.strip() == decoded.strip():
            logger.info("✅ Roundtrip successful")
        else:
            logger.warning("⚠️ Roundtrip mismatch")


def test_tokenizer_batch(tokenizer: SPMTokenizer, test_texts: List[str]):
    """Test batch tokenization."""
    logger.info("=" * 60)
    logger.info("Batch Tokenization Tests")
    logger.info("=" * 60)
    
    batch_size = 5
    batch_texts = test_texts[:batch_size]
    
    # Test batch encoding
    result = tokenizer(
        batch_texts,
        max_length=128,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Input IDs shape: {result['input_ids'].shape}")
    logger.info(f"Attention mask shape: {result['attention_mask'].shape}")
    
    # Test batch decoding
    decoded_batch = tokenizer.batch_decode(
        result['input_ids'].tolist(),
        skip_special_tokens=True
    )
    
    logger.info("\nBatch decoding results:")
    for i, (original, decoded) in enumerate(zip(batch_texts, decoded_batch)):
        logger.info(f"{i+1}. Original: {original}")
        logger.info(f"   Decoded:  {decoded}")
        logger.info(f"   Match: {'✅' if original.strip() == decoded.strip() else '⚠️'}")


def analyze_tokenizer_statistics(tokenizer: SPMTokenizer, test_texts: List[str]):
    """Analyze tokenizer statistics."""
    logger.info("=" * 60)
    logger.info("Tokenizer Statistics")
    logger.info("=" * 60)
    
    vocab_size = tokenizer.get_vocab_size()
    logger.info(f"Vocabulary size: {vocab_size:,}")
    
    # Analyze token lengths
    token_lengths = []
    char_lengths = []
    
    for text in test_texts:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        token_lengths.append(len(encoded))
        char_lengths.append(len(text))
    
    if token_lengths:
        avg_tokens = sum(token_lengths) / len(token_lengths)
        avg_chars = sum(char_lengths) / len(char_lengths)
        compression_ratio = avg_chars / avg_tokens if avg_tokens > 0 else 0
        
        logger.info(f"Average tokens per text: {avg_tokens:.2f}")
        logger.info(f"Average characters per text: {avg_chars:.2f}")
        logger.info(f"Compression ratio (chars/token): {compression_ratio:.2f}")
        logger.info(f"Min tokens: {min(token_lengths)}")
        logger.info(f"Max tokens: {max(token_lengths)}")


def test_special_tokens(tokenizer: SPMTokenizer):
    """Test special token handling."""
    logger.info("=" * 60)
    logger.info("Special Token Tests")
    logger.info("=" * 60)
    
    # Test individual special tokens
    special_tokens = [
        ("<medical>", "Medical domain marker"),
        ("<en_vi>", "English to Vietnamese marker"),
        ("<vi_en>", "Vietnamese to English marker"),
        ("<|im_start|>", "Instruction start marker"),
        ("<|im_end|>", "Instruction end marker"),
    ]
    
    for token, description in special_tokens:
        try:
            encoded = tokenizer.encode(token, add_special_tokens=False)
            decoded = tokenizer.decode(encoded, skip_special_tokens=False)
            logger.info(f"{description}:")
            logger.info(f"  Token: {token}")
            logger.info(f"  Encoded: {encoded}")
            logger.info(f"  Decoded: {decoded}")
            logger.info(f"  Match: {'✅' if token == decoded else '⚠️'}")
        except Exception as e:
            logger.warning(f"Error testing {token}: {e}")


def compare_with_huggingface(tokenizer: SPMTokenizer, test_texts: List[str], model_name: str):
    """Compare with Unsloth tokenizer if available."""
    try:
        from unsloth import FastLanguageModel
        _, hf_tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = 2048,
            dtype = "auto",
            load_in_4bit = False,
        )
        
        logger.info("=" * 60)
        logger.info("Comparison with HuggingFace Tokenizer")
        logger.info("=" * 60)
        
        for i, text in enumerate(test_texts[:3]):
            logger.info(f"\nComparison {i+1}:")
            logger.info(f"Text: {text}")
            
            # SPM tokenization
            spm_encoded = tokenizer.encode(text, add_special_tokens=False)
            smp_decoded = tokenizer.decode(spm_encoded, skip_special_tokens=True)
            
            # HF tokenization
            hf_encoded = hf_tokenizer.encode(text, add_special_tokens=False)
            hf_decoded = hf_tokenizer.decode(hf_encoded, skip_special_tokens=True)
            
            logger.info(f"SPM tokens ({len(spm_encoded)}): {spm_encoded}")
            logger.info(f"HF tokens ({len(hf_encoded)}): {hf_encoded}")
            logger.info(f"SPM decoded: {smp_decoded}")
            logger.info(f"HF decoded: {hf_decoded}")
            
    except ImportError:
        logger.info("Transformers not available, skipping comparison")
    except Exception as e:
        logger.warning(f"Error in comparison: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test SentencePiece tokenizer")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                       help="Path to trained SentencePiece model")
    parser.add_argument("--config", type=str,
                       default="../configs/moe_config.yaml",
                       help="Path to config file")
    parser.add_argument("--test_data", type=str,
                       help="Path to test data JSONL file")
    parser.add_argument("--limit", type=int, default=50,
                       help="Limit number of test texts")
    parser.add_argument("--compare_hf", type=str,
                       help="HuggingFace model name for comparison")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Convert relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.isabs(args.config):
        args.config = os.path.join(script_dir, args.config)
    
    logger.info("=" * 80)
    logger.info("SentencePiece Tokenizer Testing")
    logger.info("=" * 80)
    
    try:
        # Load config
        config = Config(args.config)
        tokenizer_config = getattr(config, 'tokenizer', {})
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from: {args.tokenizer_path}")
        tokenizer = SPMTokenizer(model_path=args.tokenizer_path, config=tokenizer_config)
        
        # Load test data
        test_texts = []
        if args.test_data:
            test_texts = load_test_data(args.test_data, args.limit)
            logger.info(f"Loaded {len(test_texts)} test texts")
        
        # Add some default test texts if no data provided
        if not test_texts:
            test_texts = [
                "Hello, this is a test sentence for tokenization.",
                "Xin chào, đây là câu thử nghiệm tokenization.",
                "Medical terminology: diabetes mellitus, hypertension, cardiovascular disease.",
                "Thuật ngữ y tế: tiểu đường, tăng huyết áp, bệnh tim mạch.",
                "The patient presents with acute chest pain and shortness of breath.",
                "Bệnh nhân có triệu chứng đau ngực cấp và khó thở.",
            ]
        
        # Run tests
        test_tokenizer_basic(tokenizer, test_texts)
        test_tokenizer_batch(tokenizer, test_texts)
        analyze_tokenizer_statistics(tokenizer, test_texts)
        test_special_tokens(tokenizer)
        
        if args.compare_hf:
            compare_with_huggingface(tokenizer, test_texts, args.compare_hf)
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 Tokenizer testing completed successfully!")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
