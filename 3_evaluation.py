"""
Aviation Question Generation System - Evaluation Module
Calculates various metrics to assess question generation quality.
"""

import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
import numpy as np
from collections import Counter
import re


class QuestionGenerationEvaluator:
    """Evaluator for question generation system."""
    
    def __init__(self, model_path, device=None):
        """
        Initialize evaluator with trained model.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run evaluation on
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model from {model_path}...")
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load sentence transformer for semantic similarity
        print("Loading sentence transformer for similarity calculation...")
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Aviation-specific keywords
        self.aviation_keywords = {
            'airspeed', 'altitude', 'heading', 'pitch', 'roll', 'yaw', 'thrust',
            'drag', 'lift', 'weight', 'stall', 'aircraft', 'flight', 'pilot',
            'navigation', 'instrument', 'altimeter', 'compass', 'transponder',
            'runway', 'takeoff', 'landing', 'approach', 'clearance', 'atc',
            'vor', 'ils', 'gps', 'autopilot', 'flaps', 'throttle', 'engine',
            'propeller', 'cockpit', 'fuselage', 'wing', 'tail', 'rudder',
            'ailerons', 'elevator', 'airspace', 'weather', 'visibility', 'cloud',
            'turbulence', 'icing', 'pressure', 'temperature', 'wind', 'speed',
            'knots', 'mach', 'vertical', 'horizontal', 'descent', 'climb'
        }
    
    def generate_questions(self, context, num_questions=5):
        """Generate multiple questions from a context."""
        input_text = f"generate question: {context}"
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=64,
                num_return_sequences=num_questions,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                early_stopping=True
            )
        
        questions = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return questions
    
    def calculate_bleu(self, reference, candidate):
        """
        Calculate BLEU score between reference and candidate.
        
        Args:
            reference: Reference question
            candidate: Generated question
        
        Returns:
            BLEU score (0-1)
        """
        reference_tokens = word_tokenize(reference.lower())
        candidate_tokens = word_tokenize(candidate.lower())
        
        smoothing = SmoothingFunction().method1
        score = sentence_bleu(
            [reference_tokens],
            candidate_tokens,
            smoothing_function=smoothing
        )
        
        return score
    
    def calculate_semantic_similarity(self, text1, text2):
        """
        Calculate semantic similarity using sentence transformers.
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Cosine similarity score (0-1)
        """
        embeddings = self.similarity_model.encode([text1, text2])
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        return similarity
    
    def check_aviation_terminology(self, text):
        """
        Check if text contains aviation-specific terminology.
        
        Args:
            text: Text to check
        
        Returns:
            Tuple of (has_terminology, found_keywords, percentage)
        """
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        
        found_keywords = words.intersection(self.aviation_keywords)
        percentage = len(found_keywords) / len(words) if words else 0
        
        return len(found_keywords) > 0, found_keywords, percentage
    
    def calculate_diversity(self, questions):
        """
        Calculate diversity of generated questions.
        
        Args:
            questions: List of generated questions
        
        Returns:
            Dictionary with diversity metrics
        """
        # Calculate unique n-grams
        all_unigrams = []
        all_bigrams = []
        
        for question in questions:
            tokens = word_tokenize(question.lower())
            all_unigrams.extend(tokens)
            all_bigrams.extend(zip(tokens[:-1], tokens[1:]))
        
        unique_unigrams = len(set(all_unigrams))
        unique_bigrams = len(set(all_bigrams))
        total_unigrams = len(all_unigrams)
        total_bigrams = len(all_bigrams)
        
        return {
            'unique_unigram_ratio': unique_unigrams / total_unigrams if total_unigrams > 0 else 0,
            'unique_bigram_ratio': unique_bigrams / total_bigrams if total_bigrams > 0 else 0,
            'total_questions': len(questions),
            'unique_questions': len(set(questions))
        }
    
    def evaluate_dataset(self, test_data_path, num_samples=None):
        """
        Evaluate model on test dataset.
        
        Args:
            test_data_path: Path to test data JSON
            num_samples: Number of samples to evaluate (None for all)
        
        Returns:
            Dictionary with evaluation metrics
        """
        print("\n" + "="*60)
        print("Evaluating Question Generation Model")
        print("="*60)
        
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        if num_samples:
            test_data = test_data[:num_samples]
        
        results = {
            'bleu_scores': [],
            'similarity_scores': [],
            'has_aviation_terms': [],
            'aviation_term_percentage': [],
            'generated_questions': []
        }
        
        print(f"\nEvaluating {len(test_data)} examples...\n")
        
        for i, item in enumerate(test_data):
            context = item['context']
            reference_question = item['target_text']
            
            # Generate question
            generated_questions = self.generate_questions(context, num_questions=1)
            generated_question = generated_questions[0]
            
            # Calculate metrics
            bleu = self.calculate_bleu(reference_question, generated_question)
            similarity = self.calculate_semantic_similarity(reference_question, generated_question)
            has_terms, found_terms, term_percentage = self.check_aviation_terminology(generated_question)
            
            results['bleu_scores'].append(bleu)
            results['similarity_scores'].append(similarity)
            results['has_aviation_terms'].append(has_terms)
            results['aviation_term_percentage'].append(term_percentage)
            results['generated_questions'].append({
                'context': context[:100] + '...',
                'reference': reference_question,
                'generated': generated_question,
                'bleu': bleu,
                'similarity': similarity,
                'has_aviation_terms': has_terms
            })
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(test_data)} examples")
        
        # Calculate aggregate statistics
        avg_bleu = np.mean(results['bleu_scores'])
        avg_similarity = np.mean(results['similarity_scores'])
        aviation_term_coverage = np.mean(results['has_aviation_terms']) * 100
        avg_term_percentage = np.mean(results['aviation_term_percentage']) * 100
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Average BLEU Score: {avg_bleu:.4f}")
        print(f"Average Semantic Similarity: {avg_similarity:.4f}")
        print(f"Aviation Terminology Coverage: {aviation_term_coverage:.1f}%")
        print(f"Average Aviation Terms per Question: {avg_term_percentage:.1f}%")
        print("="*60)
        
        # Save detailed results
        output_path = 'evaluation_results.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'avg_bleu': avg_bleu,
                    'avg_similarity': avg_similarity,
                    'aviation_term_coverage': aviation_term_coverage,
                    'avg_term_percentage': avg_term_percentage,
                    'num_samples': len(test_data)
                },
                'detailed_results': results['generated_questions']
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Detailed results saved to {output_path}")
        
        return results
    
    def generate_sample_report(self, context_samples):
        """
        Generate sample questions and create a report.
        
        Args:
            context_samples: List of context strings to generate questions from
        
        Returns:
            Dictionary with generated samples and analysis
        """
        print("\n" + "="*60)
        print("Generating Sample Questions")
        print("="*60)
        
        all_generated = []
        
        for i, context in enumerate(context_samples, 1):
            print(f"\nSample {i}:")
            print(f"Context: {context[:100]}...")
            
            questions = self.generate_questions(context, num_questions=3)
            
            print("Generated Questions:")
            for j, question in enumerate(questions, 1):
                has_terms, found_terms, _ = self.check_aviation_terminology(question)
                print(f"  {j}. {question}")
                if found_terms:
                    print(f"     Aviation terms: {', '.join(list(found_terms)[:5])}")
            
            all_generated.extend(questions)
        
        # Calculate diversity
        diversity_metrics = self.calculate_diversity(all_generated)
        
        print("\n" + "="*60)
        print("GENERATION DIVERSITY METRICS")
        print("="*60)
        print(f"Unique Unigram Ratio: {diversity_metrics['unique_unigram_ratio']:.3f}")
        print(f"Unique Bigram Ratio: {diversity_metrics['unique_bigram_ratio']:.3f}")
        print(f"Unique Questions: {diversity_metrics['unique_questions']}/{diversity_metrics['total_questions']}")
        print("="*60)
        
        return {
            'generated_questions': all_generated,
            'diversity': diversity_metrics
        }


def main():
    """Main evaluation function."""
    
    # Initialize evaluator
    evaluator = QuestionGenerationEvaluator(
        model_path='./aviation_qg_model/checkpoint-epoch-4'
    )
    
    # Evaluate on validation set
    evaluator.evaluate_dataset('aviation_val.json')
    
    # Generate samples
    sample_contexts = [
        "The four forces acting on an aircraft are lift, weight, thrust, and drag. These forces must be balanced for stable flight.",
        "VOR stands for VHF Omnidirectional Range and is a radio navigation system used by aircraft.",
        "During a stall, the wing exceeds the critical angle of attack and loses lift suddenly."
    ]
    
    evaluator.generate_sample_report(sample_contexts)


if __name__ == "__main__":
    # Download NLTK data if needed
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    main()
