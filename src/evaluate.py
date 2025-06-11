"""
Translation Quality Evaluation Tool
"""
import csv
import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass, field
import logging
import statistics
from datetime import datetime

try:
    from litellm import completion
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False

__version__ = "0.1.0"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


DEFAULT_EVALUATION_PROMPT = """You are an expert translation evaluator. Please analyze the following English-Chinese translation pairs and provide a detailed evaluation for each.

Translation Pairs:
{translation_pairs}

Please evaluate each translation and respond with a JSON array containing evaluation objects in the same order as the input pairs. Each evaluation object should have this structure:
{{
    "overall_score": <float 0-10>,
    "missed_elements": ["element1", "element2", ...],
    "added_elements": ["element1", "element2", ...], 
    "wrong_translations": ["wrong_word/phrase -> should_be", ...],
    "grammar_errors": ["error1", "error2", ...],
    "style_issues": ["issue1", "issue2", ...],
    "cultural_adaptation": <float 0-10>,
    "fluency": <float 0-10>,
    "accuracy": <float 0-10>,
    "reasoning": "Brief explanation of the evaluation"
}}

Focus on:
- Missed elements: Important content from English that's missing in Chinese
- Added elements: Content in Chinese that wasn't in the English original
- Wrong translations: Incorrect word choices, especially with multiple meanings
- Grammar and syntax errors in the Chinese translation
- Style appropriateness and naturalness
- Cultural adaptation and localization quality
- Overall fluency and readability

Respond with a JSON array: [evaluation1, evaluation2, ...]"""


@dataclass
class EvaluationConfig:
    """Configuration for translation evaluation."""
    
    model: str = "mistral/mistral-medium"
    temperature: float = 0.3
    max_tokens: int = 2000 # may need to adjust according to evaluation batch size
    custom_prompt: Optional[str] = None
    batch_size: int = 5  # For rate limiting/processing
    evaluation_batch_size: int = 3  # Number of sentence pairs per LLM call
    timeout: int = 30
    retry_attempts: int = 3
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class TranslationEvaluation:
    """Results of a single translation evaluation."""
    
    english_sentence: str
    chinese_sentence: str
    original_alignment_score: float
    overall_score: float
    missed_elements: List[str] = field(default_factory=list)
    added_elements: List[str] = field(default_factory=list)
    wrong_translations: List[str] = field(default_factory=list)
    grammar_errors: List[str] = field(default_factory=list)
    style_issues: List[str] = field(default_factory=list)
    cultural_adaptation: float = 0.0
    fluency: float = 0.0
    accuracy: float = 0.0
    reasoning: str = ""
    evaluation_error: Optional[str] = None


@dataclass
class EvaluationSummary:
    """Summary statistics for the entire evaluation."""
    
    total_sentences: int = 0
    average_overall_score: float = 0.0
    average_cultural_adaptation: float = 0.0
    average_fluency: float = 0.0
    average_accuracy: float = 0.0
    score_distribution: Dict[str, int] = field(default_factory=dict)
    most_common_issues: Dict[str, int] = field(default_factory=dict)
    evaluation_time: str = ""
    model_used: str = ""
    failed_evaluations: int = 0


class TranslationEvaluator:
    """Main translation evaluation class."""
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self._validate_setup()
    
    def _validate_setup(self):
        """Validate that required dependencies are available."""
        if not HAS_LITELLM:
            raise ImportError(
                "litellm is required for translation evaluation. "
                "Install it with: pip install litellm"
            )
    
    def evaluate_translations(
        self,
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        report_file: Optional[Union[str, Path]] = None
    ) -> bool:
        """
        Evaluate translations from an aligned TSV file.
        
        Args:
            input_file: Path to aligned sentences TSV file
            output_file: Path for evaluation results TSV
            report_file: Path for summary report JSON (optional)
        
        Returns:
            bool: True if evaluation completed successfully
        """
        try:
            # Read input data
            aligned_sentences = self._read_aligned_sentences(input_file)
            logger.info(f"Loaded {len(aligned_sentences)} sentence pairs for evaluation")
            
            # Perform evaluation
            evaluations = self._evaluate_batch(aligned_sentences)
            logger.info(f"Completed evaluation of {len(evaluations)} sentences")
            
            # Generate summary
            summary = self._generate_summary(evaluations)
            
            # Write outputs
            self._write_evaluation_results(evaluations, output_file)
            
            if report_file:
                self._write_summary_report(summary, report_file)
            
            # Print summary to console
            self._print_summary(summary)
            
            return True
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return False
    
    def _read_aligned_sentences(self, input_file: Union[str, Path]) -> List[Dict]:
        """Read aligned sentences from TSV file."""
        sentences = []
        input_path = Path(input_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        with input_path.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                sentences.append({
                    'english': row.get('English Sentence', '').strip('"'),
                    'chinese': row.get('Chinese Sentence', '').strip('"'),
                    'alignment_score': float(row.get('Alignment Score', 0.0))
                })
        
        return sentences
    
    def _evaluate_batch(self, sentences: List[Dict]) -> List[TranslationEvaluation]:
        """Evaluate a batch of sentence pairs using batch LLM calls."""
        evaluations = []
        
        # Process sentences in evaluation batches
        for i in range(0, len(sentences), self.config.evaluation_batch_size):
            batch_end = min(i + self.config.evaluation_batch_size, len(sentences))
            sentence_batch = sentences[i:batch_end]
            
            logger.info(f"Evaluating batch {i//self.config.evaluation_batch_size + 1}: sentences {i+1}-{batch_end}")
            
            # Always use batch evaluation (works for both single and multiple sentences)
            batch_evaluations = self._evaluate_multiple_pairs(sentence_batch)
            evaluations.extend(batch_evaluations)
            
            # Rate limiting delay
            if i > 0 and (i // self.config.evaluation_batch_size) % self.config.batch_size == 0:
                import time
                time.sleep(1)
        
        return evaluations
    
    def _evaluate_multiple_pairs(self, sentence_batch: List[Dict]) -> List[TranslationEvaluation]:
        """Evaluate translation pairs in a single LLM call (handles both single and multiple pairs)."""
        evaluations = []
        
        # Initialize evaluations with basic info
        for sentence_pair in sentence_batch:
            evaluation = TranslationEvaluation(
                english_sentence=sentence_pair['english'],
                chinese_sentence=sentence_pair['chinese'],
                original_alignment_score=sentence_pair['alignment_score'],
                overall_score=0.0
            )
            evaluations.append(evaluation)
        
        try:
            # Prepare batch prompt
            translation_pairs_text = ""
            for idx, sentence_pair in enumerate(sentence_batch, 1):
                translation_pairs_text += f"{idx}. English: {sentence_pair['english']}\n"
                translation_pairs_text += f"   Chinese: {sentence_pair['chinese']}\n\n"
            
            prompt = self.config.custom_prompt or DEFAULT_EVALUATION_PROMPT
            formatted_prompt = prompt.format(translation_pairs=translation_pairs_text.strip())
            
            # Make API call with retry logic
            response = self._call_llm_with_retry(formatted_prompt)
            
            # Parse batch response
            self._parse_batch_evaluation_response(response, evaluations)
            
        except Exception as e:
            logger.warning(f"Failed to evaluate sentence batch: {e}")
            # Mark all evaluations in batch as failed
            for evaluation in evaluations:
                evaluation.evaluation_error = str(e)
                evaluation.overall_score = 0.0
        
        return evaluations
    

    
    def _call_llm_with_retry(self, prompt: str) -> str:
        """Call LLM with retry logic."""
        for attempt in range(self.config.retry_attempts):
            try:
                response = completion(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                if attempt == self.config.retry_attempts - 1:
                    raise e
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                import time
                time.sleep(2 ** attempt)  # Exponential backoff
    

    
    def _parse_batch_evaluation_response(self, response: str, evaluations: List[TranslationEvaluation]):
        """Parse LLM response and populate batch evaluation objects."""
        try:
            # Try to extract JSON from response
            response_clean = response.strip()
            if response_clean.startswith('```json'):
                response_clean = response_clean[7:-3]
            elif response_clean.startswith('```'):
                response_clean = response_clean[3:-3]
            
            data = json.loads(response_clean)
            
            # Ensure data is a list
            if not isinstance(data, list):
                raise ValueError("Expected JSON array but got a different format")
            
            # Check if we have the expected number of evaluations
            if len(data) != len(evaluations):
                logger.warning(f"Expected {len(evaluations)} evaluations but got {len(data)}. Adjusting...")
            
            # Populate evaluation fields for available data
            for idx in range(min(len(data), len(evaluations))):
                try:
                    eval_data = data[idx]
                    evaluation = evaluations[idx]
                    
                    evaluation.overall_score = float(eval_data.get('overall_score', 0.0))
                    evaluation.missed_elements = eval_data.get('missed_elements', [])
                    evaluation.added_elements = eval_data.get('added_elements', [])
                    evaluation.wrong_translations = eval_data.get('wrong_translations', [])
                    evaluation.grammar_errors = eval_data.get('grammar_errors', [])
                    evaluation.style_issues = eval_data.get('style_issues', [])
                    evaluation.cultural_adaptation = float(eval_data.get('cultural_adaptation', 0.0))
                    evaluation.fluency = float(eval_data.get('fluency', 0.0))
                    evaluation.accuracy = float(eval_data.get('accuracy', 0.0))
                    evaluation.reasoning = eval_data.get('reasoning', '')
                    
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"Error parsing evaluation {idx}: {e}")
                    evaluations[idx].evaluation_error = f"Parse error for item {idx}: {e}"
                    evaluations[idx].overall_score = 0.0
            
            # Mark remaining evaluations as failed if we got fewer than expected
            for idx in range(len(data), len(evaluations)):
                evaluations[idx].evaluation_error = "Missing evaluation in LLM response"
                evaluations[idx].overall_score = 0.0
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM batch response: {e}")
            logger.debug(f"Raw response: {response}")
            # Mark all evaluations in batch as failed
            for evaluation in evaluations:
                evaluation.evaluation_error = f"Batch parse error: {e}"
                evaluation.overall_score = 0.0
                evaluation.reasoning = response[:200] + "..." if len(response) > 200 else response
    
    def _generate_summary(self, evaluations: List[TranslationEvaluation]) -> EvaluationSummary:
        """Generate summary statistics from evaluations."""
        summary = EvaluationSummary()
        summary.total_sentences = len(evaluations)
        summary.evaluation_time = datetime.now().isoformat()
        summary.model_used = self.config.model
        
        # Filter successful evaluations
        successful_evals = [e for e in evaluations if e.evaluation_error is None]
        summary.failed_evaluations = len(evaluations) - len(successful_evals)
        
        if not successful_evals:
            return summary
        
        # Calculate averages
        scores = [e.overall_score for e in successful_evals]
        cultural_scores = [e.cultural_adaptation for e in successful_evals if e.cultural_adaptation > 0]
        fluency_scores = [e.fluency for e in successful_evals if e.fluency > 0]
        accuracy_scores = [e.accuracy for e in successful_evals if e.accuracy > 0]
        
        summary.average_overall_score = statistics.mean(scores)
        if cultural_scores:
            summary.average_cultural_adaptation = statistics.mean(cultural_scores)
        if fluency_scores:
            summary.average_fluency = statistics.mean(fluency_scores)
        if accuracy_scores:
            summary.average_accuracy = statistics.mean(accuracy_scores)
        
        # Score distribution
        score_ranges = {
            '9-10 (Excellent)': 0,
            '7-8.9 (Good)': 0,
            '5-6.9 (Fair)': 0,
            '3-4.9 (Poor)': 0,
            '0-2.9 (Very Poor)': 0
        }
        
        for score in scores:
            if score >= 9:
                score_ranges['9-10 (Excellent)'] += 1
            elif score >= 7:
                score_ranges['7-8.9 (Good)'] += 1
            elif score >= 5:
                score_ranges['5-6.9 (Fair)'] += 1
            elif score >= 3:
                score_ranges['3-4.9 (Poor)'] += 1
            else:
                score_ranges['0-2.9 (Very Poor)'] += 1
        
        summary.score_distribution = score_ranges
        
        # Most common issues
        all_issues = {}
        for evaluation in successful_evals:
            for issue_type, issues in [
                ('Missed Elements', evaluation.missed_elements),
                ('Added Elements', evaluation.added_elements),
                ('Wrong Translations', evaluation.wrong_translations),
                ('Grammar Errors', evaluation.grammar_errors),
                ('Style Issues', evaluation.style_issues)
            ]:
                if issues:
                    all_issues[issue_type] = all_issues.get(issue_type, 0) + len(issues)
        
        summary.most_common_issues = dict(sorted(all_issues.items(), key=lambda x: x[1], reverse=True))
        
        return summary
    
    def _write_evaluation_results(self, evaluations: List[TranslationEvaluation], output_file: Union[str, Path]):
        """Write detailed evaluation results to TSV file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            
            # Header
            writer.writerow([
                'English Sentence',
                'Chinese Sentence',
                'Original Alignment Score',
                'Overall Score',
                'Missed Elements',
                'Added Elements',
                'Wrong Translations',
                'Grammar Errors',
                'Style Issues',
                'Cultural Adaptation',
                'Fluency',
                'Accuracy',
                'Reasoning',
                'Evaluation Error'
            ])
            
            # Data rows
            for eval_result in evaluations:
                writer.writerow([
                    eval_result.english_sentence,
                    eval_result.chinese_sentence,
                    eval_result.original_alignment_score,
                    eval_result.overall_score,
                    '; '.join(eval_result.missed_elements),
                    '; '.join(eval_result.added_elements),
                    '; '.join(eval_result.wrong_translations),
                    '; '.join(eval_result.grammar_errors),
                    '; '.join(eval_result.style_issues),
                    eval_result.cultural_adaptation,
                    eval_result.fluency,
                    eval_result.accuracy,
                    eval_result.reasoning,
                    eval_result.evaluation_error or ''
                ])
        
        logger.info(f"Evaluation results written to: {output_path}")
    
    def _write_summary_report(self, summary: EvaluationSummary, report_file: Union[str, Path]):
        """Write summary report to JSON file."""
        report_path = Path(report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        report_data = {
            'evaluation_summary': {
                'total_sentences': summary.total_sentences,
                'successful_evaluations': summary.total_sentences - summary.failed_evaluations,
                'failed_evaluations': summary.failed_evaluations,
                'average_scores': {
                    'overall': round(summary.average_overall_score, 2),
                    'cultural_adaptation': round(summary.average_cultural_adaptation, 2),
                    'fluency': round(summary.average_fluency, 2),
                    'accuracy': round(summary.average_accuracy, 2)
                },
                'score_distribution': summary.score_distribution,
                'most_common_issues': summary.most_common_issues
            },
            'metadata': {
                'evaluation_time': summary.evaluation_time,
                'model_used': summary.model_used,
                'tool_version': __version__
            }
        }
        
        with report_path.open('w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Summary report written to: {report_path}")
    
    def _print_summary(self, summary: EvaluationSummary):
        """Print summary to console."""
        print("\n" + "="*60)
        print("TRANSLATION QUALITY EVALUATION SUMMARY")
        print("="*60)
        print(f"Model Used: {summary.model_used}")
        print(f"Total Sentences: {summary.total_sentences}")
        print(f"Successful Evaluations: {summary.total_sentences - summary.failed_evaluations}")
        print(f"Failed Evaluations: {summary.failed_evaluations}")
        print("\nAverage Scores:")
        print(f"  Overall Quality: {summary.average_overall_score:.2f}/10")
        print(f"  Cultural Adaptation: {summary.average_cultural_adaptation:.2f}/10")
        print(f"  Fluency: {summary.average_fluency:.2f}/10")
        print(f"  Accuracy: {summary.average_accuracy:.2f}/10")
        
        print("\nScore Distribution:")
        for range_name, count in summary.score_distribution.items():
            percentage = (count / summary.total_sentences) * 100 if summary.total_sentences > 0 else 0
            print(f"  {range_name}: {count} ({percentage:.1f}%)")
        
        print("\nMost Common Issues:")
        for issue_type, count in list(summary.most_common_issues.items())[:5]:
            print(f"  {issue_type}: {count} occurrences")
        
        print("\n" + "="*60)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate translation quality using LLMs with batch processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation (processes 3 sentence pairs per LLM call by default)
  python -m src.evaluate input.tsv output.tsv
  
  # Use different model with custom settings
  python -m src.evaluate input.tsv output.tsv --model gpt-4 --temperature 0.2
  
  # Process 5 sentence pairs per LLM call to reduce token usage
  python -m src.evaluate input.tsv output.tsv --eval-batch-size 5
  
  # Process single sentence per LLM call for more detailed evaluation
  python -m src.evaluate input.tsv output.tsv --eval-batch-size 1
  
  # Generate summary report
  python -m src.evaluate input.tsv output.tsv --report summary.json
  
  # Use custom prompt
  python -m src.evaluate input.tsv output.tsv --prompt custom_prompt.txt
        """
    )
    
    parser.add_argument('input_file', help='Input TSV file with aligned sentences')
    parser.add_argument('output_file', help='Output TSV file for evaluation results')
    parser.add_argument('--report', '-r', help='Output JSON file for summary report')
    parser.add_argument('--model', '-m', default='mistral/mistral-medium', 
                       help='LLM model to use (default: mistral/mistral-medium)')
    parser.add_argument('--temperature', '-t', type=float, default=0.3,
                       help='Temperature for LLM (default: 0.3)')
    parser.add_argument('--prompt', '-p', help='Custom prompt file')
    parser.add_argument('--batch-size', '-b', type=int, default=5,
                       help='Batch size for rate limiting (default: 5)')
    parser.add_argument('--eval-batch-size', '-e', type=int, default=3,
                       help='Number of sentence pairs per LLM call (default: 3)')
    parser.add_argument('--timeout', type=int, default=30,
                       help='API timeout in seconds (default: 30)')
    parser.add_argument('--max-tokens', type=int, default=2000,
                       help='Maximum tokens for LLM response (default: 2000)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load custom prompt if provided
    custom_prompt = None
    if args.prompt:
        prompt_path = Path(args.prompt)
        if prompt_path.exists():
            custom_prompt = prompt_path.read_text(encoding='utf-8')
        else:
            logger.error(f"Prompt file not found: {prompt_path}")
            sys.exit(1)
    
    # Create configuration
    config = EvaluationConfig(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        custom_prompt=custom_prompt,
        batch_size=args.batch_size,
        evaluation_batch_size=args.eval_batch_size,
        timeout=args.timeout
    )
    
    # Create evaluator and run
    evaluator = TranslationEvaluator(config)
    
    success = evaluator.evaluate_translations(
        input_file=args.input_file,
        output_file=args.output_file,
        report_file=args.report
    )
    
    if not success:
        sys.exit(1)
    
    logger.info("Translation evaluation completed successfully!")


if __name__ == "__main__":
    main() 