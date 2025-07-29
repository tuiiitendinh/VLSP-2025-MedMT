import sacrebleu
from typing import List, Optional, Dict, Any
import pandas as pd

class BLEUEvaluator:
    """
    A class for evaluating BLEU scores from a DataFrame with original, reference, and destination columns.
    
    Expected DataFrame columns:
    - original: Source text (e.g., Vietnamese medical text)
    - reference: Reference translation(s) (can be single string or list of strings)
    - destination: Model's translation to evaluate
    """
    
    def __init__(self, tokenizer: str = 'intl'):
        """
        Initialize the BLEU evaluator.
        
        Args:
            tokenizer: Tokenizer to use ('intl' for international, '13a' for English)
        """
        self.tokenizer = tokenizer
        
    def evaluate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate BLEU scores from a DataFrame.
        
        Args:
            df: DataFrame with columns 'original', 'reference', 'destination'
            
        Returns:
            Dictionary containing BLEU score and detailed results
        """
        # Validate DataFrame columns
        required_columns = ['original', 'reference', 'destination']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Extract translations and references
        model_translations = df['destination'].tolist()
        references = self._prepare_references(df['reference'].tolist())
        
        # Calculate BLEU score
        bleu_score = sacrebleu.corpus_bleu(model_translations, references, tokenize=self.tokenizer)
        
        # Calculate individual sentence BLEU scores
        individual_scores = []
        for i, (translation, ref_list) in enumerate(zip(model_translations, references)):
            try:
                sentence_bleu = sacrebleu.sentence_bleu(translation, ref_list, tokenize=self.tokenizer)
                individual_scores.append({
                    'index': i,
                    'original': df.iloc[i]['original'],
                    'translation': translation,
                    'references': ref_list,
                    'bleu_score': sentence_bleu.score
                })
            except Exception as e:
                individual_scores.append({
                    'index': i,
                    'original': df.iloc[i]['original'],
                    'translation': translation,
                    'references': ref_list,
                    'bleu_score': 0.0,
                    'error': str(e)
                })
        
        return {
            'corpus_bleu_score': bleu_score.score,
            # 'bleu_signature': bleu_score.signature(),
            'individual_scores': individual_scores,
            'statistics': self._calculate_statistics(individual_scores)
        }
    
    def _prepare_references(self, references: List) -> List[List[str]]:
        """
        Prepare references for BLEU calculation.
        
        Args:
            references: List of references (can be strings or lists of strings)
            
        Returns:
            List of lists, where each inner list contains reference translations
        """
        prepared_refs = []
        for ref in references:
            if isinstance(ref, str):
                # Single reference
                prepared_refs.append([ref])
            elif isinstance(ref, list):
                # Multiple references
                prepared_refs.append(ref)
            else:
                # Convert to string if not already
                prepared_refs.append([str(ref)])
        return prepared_refs
    
    def _calculate_statistics(self, individual_scores: List[Dict]) -> Dict[str, float]:
        """Calculate statistics from individual BLEU scores."""
        valid_scores = [item['bleu_score'] for item in individual_scores if 'error' not in item]
        
        if not valid_scores:
            return {'mean': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0}
        
        import statistics
        return {
            'mean': statistics.mean(valid_scores),
            'median': statistics.median(valid_scores),
            'min': min(valid_scores),
            'max': max(valid_scores),
            'std': statistics.stdev(valid_scores) if len(valid_scores) > 1 else 0.0,
            'valid_sentences': len(valid_scores),
            'total_sentences': len(individual_scores)
        }
    
    def print_detailed_report(self, results: Dict[str, Any]):
        """Print a detailed evaluation report."""
        print("=" * 60)
        print("BLEU EVALUATION REPORT")
        print("=" * 60)
        print(f"Corpus BLEU Score: {results['corpus_bleu_score']:.2f}")
        # print(f"Tokenizer: {results['bleu_signature']}")
        print()
        
        stats = results['statistics']
        print("INDIVIDUAL SENTENCE STATISTICS:")
        print(f"  Mean BLEU: {stats['mean']:.2f}")
        print(f"  Median BLEU: {stats['median']:.2f}")
        print(f"  Min BLEU: {stats['min']:.2f}")
        print(f"  Max BLEU: {stats['max']:.2f}")
        print(f"  Std Dev: {stats['std']:.2f}")
        print(f"  Valid sentences: {stats['valid_sentences']}/{stats['total_sentences']}")
        print()
        
        # Show top and bottom performing sentences
        individual_scores = results['individual_scores']
        valid_scores = [item for item in individual_scores if 'error' not in item]
        
        if valid_scores:
            # Sort by BLEU score
            sorted_scores = sorted(valid_scores, key=lambda x: x['bleu_score'], reverse=True)
            
            print("TOP 3 BEST TRANSLATIONS:")
            for i, item in enumerate(sorted_scores[:3], 1):
                print(f"  {i}. BLEU: {item['bleu_score']:.2f}")
                print(f"     Original: {item['original']}")
                print(f"     Translation: {item['translation']}")
                print(f"     Reference: {item['references'][0]}")
                print()
            
            print("TOP 3 WORST TRANSLATIONS:")
            for i, item in enumerate(sorted_scores[-3:], 1):
                print(f"  {i}. BLEU: {item['bleu_score']:.2f}")
                print(f"     Original: {item['original']}")
                print(f"     Translation: {item['translation']}")
                print(f"     Reference: {item['references'][0]}")
                print()