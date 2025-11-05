"""
데이터셋 기반 평가 메트릭
RAG-Evaluation-Dataset-KO 데이터셋을 사용한 평가
"""

from typing import List, Dict, Any, Optional
import numpy as np
from .metrics import RAGEvaluator

# LLM 평가 모듈 (조건부 import)
try:
    from .llm_evaluator import LLMEvaluator
    LLM_EVAL_AVAILABLE = True
except ImportError:
    LLM_EVAL_AVAILABLE = False


class DatasetEvaluator:
    """데이터셋 기반 평가 클래스"""
    
    def __init__(self):
        self.base_evaluator = RAGEvaluator()
    
    def evaluate_answer_correctness(self,
                                   predicted_answer: str,
                                   target_answer: str,
                                   use_llm_eval: bool = True) -> Dict[str, Any]:
        """
        답변 정확도 평가 (Auto Evaluate 방식)
        
        Args:
            predicted_answer: 예측된 답변
            target_answer: 정답
            use_llm_eval: LLM 기반 평가 사용 여부
        
        Returns:
            평가 결과 (O/X 및 스코어)
        """
        # LLM 기반 평가 사용 (가능한 경우)
        if use_llm_eval and LLM_EVAL_AVAILABLE:
            try:
                llm_evaluator = LLMEvaluator()
                auto_eval_result = llm_evaluator.auto_evaluate(
                    predicted_answer, target_answer, voting_threshold=3
                )
                
                # ROUGE 스코어도 추가
                rouge_scores = self.base_evaluator.evaluate_generation(
                    predicted_answer, target_answer
                )
                
                return {
                    'correct': auto_eval_result['final_correct'],
                    'ox': auto_eval_result['final_ox'],
                    'voting_o_count': auto_eval_result['o_count'],
                    'voting_total': auto_eval_result['total_evaluators'],
                    'rouge1': rouge_scores['rouge1'],
                    'rouge2': rouge_scores['rouge2'],
                    'rougeL': rouge_scores['rougeL'],
                    'avg_rouge': np.mean([
                        rouge_scores['rouge1'],
                        rouge_scores['rouge2'],
                        rouge_scores['rougeL']
                    ]),
                    'evaluation_method': 'LLM Auto Evaluate'
                }
            except Exception as e:
                # LLM 평가 실패 시 fallback
                pass
        
        # Fallback: ROUGE 스코어 기반 평가
        rouge_scores = self.base_evaluator.evaluate_generation(
            predicted_answer, target_answer
        )
        
        # 평균 ROUGE 스코어
        avg_rouge = np.mean([
            rouge_scores['rouge1'],
            rouge_scores['rouge2'],
            rouge_scores['rougeL']
        ])
        
        # 임계값 기반 O/X 결정 (threshold=0.4, rouge 기준)
        is_correct = avg_rouge >= 0.4
        
        return {
            'correct': is_correct,
            'ox': 'O' if is_correct else 'X',
            'rouge1': rouge_scores['rouge1'],
            'rouge2': rouge_scores['rouge2'],
            'rougeL': rouge_scores['rougeL'],
            'avg_rouge': avg_rouge,
            'evaluation_method': 'ROUGE-based'
        }
    
    def evaluate_by_context_type(self,
                                results: List[Dict[str, Any]],
                                context_types: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Context type별 성능 평가
        
        Args:
            results: 평가 결과 리스트
            context_types: 각 결과의 context_type 리스트 (paragraph, table, image)
        
        Returns:
            Context type별 메트릭
        """
        type_metrics = {}
        
        for context_type in ['paragraph', 'table', 'image']:
            type_results = [
                r for r, ct in zip(results, context_types) 
                if ct == context_type
            ]
            
            if not type_results:
                continue
            
            correct_count = sum(1 for r in type_results if r.get('correct', False))
            total_count = len(type_results)
            
            type_metrics[context_type] = {
                'accuracy': correct_count / total_count if total_count > 0 else 0.0,
                'correct': correct_count,
                'total': total_count
            }
        
        return type_metrics
    
    def evaluate_by_domain(self,
                          results: List[Dict[str, Any]],
                          domains: List[str]) -> Dict[str, Dict[str, float]]:
        """
        도메인별 성능 평가
        
        Args:
            results: 평가 결과 리스트
            domains: 각 결과의 도메인 리스트
        
        Returns:
            도메인별 메트릭
        """
        domain_metrics = {}
        
        for domain in ['finance', 'public', 'medical', 'law', 'commerce']:
            domain_results = [
                r for r, d in zip(results, domains) 
                if d == domain
            ]
            
            if not domain_results:
                continue
            
            correct_count = sum(1 for r in domain_results if r.get('correct', False))
            total_count = len(domain_results)
            
            domain_metrics[domain] = {
                'accuracy': correct_count / total_count if total_count > 0 else 0.0,
                'correct': correct_count,
                'total': total_count
            }
        
        return domain_metrics
    
    def calculate_overall_metrics(self,
                                  results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        전체 평균 메트릭 계산
        
        Args:
            results: 평가 결과 리스트
        
        Returns:
            전체 평균 메트릭
        """
        if not results:
            return {}
        
        correct_count = sum(1 for r in results if r.get('correct', False))
        total_count = len(results)
        
        avg_rouge1 = np.mean([r.get('rouge1', 0.0) for r in results])
        avg_rouge2 = np.mean([r.get('rouge2', 0.0) for r in results])
        avg_rougeL = np.mean([r.get('rougeL', 0.0) for r in results])
        avg_rouge = np.mean([r.get('avg_rouge', 0.0) for r in results])
        
        return {
            'accuracy': correct_count / total_count if total_count > 0 else 0.0,
            'correct': correct_count,
            'total': total_count,
            'avg_rouge1': avg_rouge1,
            'avg_rouge2': avg_rouge2,
            'avg_rougeL': avg_rougeL,
            'avg_rouge': avg_rouge
        }

