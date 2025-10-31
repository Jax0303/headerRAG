"""
RAG 시스템 평가 메트릭
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from rouge_score import rouge_scorer
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class RAGEvaluator:
    """RAG 시스템 평가 클래스"""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], 
                                                      use_stemmer=True)
    
    def evaluate_retrieval(self, 
                          retrieved_results: List[Dict[str, Any]],
                          ground_truth_ids: List[str]) -> Dict[str, float]:
        """
        검색 성능 평가
        
        Args:
            retrieved_results: 검색된 결과 리스트 (table_id 포함)
            ground_truth_ids: 정답 테이블 ID 리스트
        
        Returns:
            평가 메트릭 딕셔너리
        """
        retrieved_ids = [r['table_id'] for r in retrieved_results]
        
        # Precision@K
        precision = len(set(retrieved_ids) & set(ground_truth_ids)) / len(retrieved_ids) if retrieved_ids else 0.0
        
        # Recall@K
        recall = len(set(retrieved_ids) & set(ground_truth_ids)) / len(ground_truth_ids) if ground_truth_ids else 0.0
        
        # F1 Score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Mean Reciprocal Rank (MRR)
        mrr = 0.0
        for gt_id in ground_truth_ids:
            if gt_id in retrieved_ids:
                rank = retrieved_ids.index(gt_id) + 1
                mrr += 1.0 / rank
        mrr = mrr / len(ground_truth_ids) if ground_truth_ids else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mrr': mrr
        }
    
    def evaluate_generation(self,
                           generated_text: str,
                           reference_text: str) -> Dict[str, float]:
        """
        생성된 텍스트 평가 (ROUGE 스코어)
        
        Args:
            generated_text: 생성된 텍스트
            reference_text: 참조 텍스트
        
        Returns:
            ROUGE 스코어 딕셔너리
        """
        scores = self.rouge_scorer.score(reference_text, generated_text)
        
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def evaluate_answer_accuracy(self,
                                predicted_answers: List[Dict[str, Any]],
                                ground_truth_answers: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        답변 정확도 평가
        
        Args:
            predicted_answers: 예측된 답변 리스트
            ground_truth_answers: 정답 리스트
        
        Returns:
            정확도 메트릭
        """
        if not predicted_answers or not ground_truth_answers:
            return {'accuracy': 0.0, 'exact_match': 0.0}
        
        # Exact Match
        exact_matches = 0
        for pred in predicted_answers:
            for gt in ground_truth_answers:
                if pred.get('value') == gt.get('value'):
                    exact_matches += 1
                    break
        
        exact_match_rate = exact_matches / len(ground_truth_answers) if ground_truth_answers else 0.0
        
        # 부분 일치 (값 포함 여부)
        partial_matches = 0
        for pred in predicted_answers:
            pred_value = str(pred.get('value', '')).lower()
            for gt in ground_truth_answers:
                gt_value = str(gt.get('value', '')).lower()
                if gt_value in pred_value or pred_value in gt_value:
                    partial_matches += 1
                    break
        
        partial_match_rate = partial_matches / len(ground_truth_answers) if ground_truth_answers else 0.0
        
        return {
            'exact_match': exact_match_rate,
            'partial_match': partial_match_rate,
            'accuracy': exact_match_rate  # 기본 정확도는 exact match
        }
    
    def evaluate_parsing_quality(self,
                                parsed_structure: Dict[str, Any],
                                ground_truth_structure: Dict[str, Any]) -> Dict[str, float]:
        """
        파싱 품질 평가
        
        Args:
            parsed_structure: 파싱된 구조
            ground_truth_structure: 정답 구조
        
        Returns:
            파싱 품질 메트릭
        """
        metrics = {}
        
        # 헤더 감지 정확도
        if 'headers' in parsed_structure and 'headers' in ground_truth_structure:
            pred_headers = set(str(h) for h in parsed_structure['headers'])
            gt_headers = set(str(h) for h in ground_truth_structure['headers'])
            header_precision = len(pred_headers & gt_headers) / len(pred_headers) if pred_headers else 0.0
            header_recall = len(pred_headers & gt_headers) / len(gt_headers) if gt_headers else 0.0
            metrics['header_precision'] = header_precision
            metrics['header_recall'] = header_recall
            metrics['header_f1'] = 2 * header_precision * header_recall / (header_precision + header_recall) if (header_precision + header_recall) > 0 else 0.0
        
        # 데이터 셀 감지 정확도
        if 'data_cells' in parsed_structure and 'data_cells' in ground_truth_structure:
            pred_cells = len(parsed_structure['data_cells'])
            gt_cells = len(ground_truth_structure['data_cells'])
            cell_ratio = pred_cells / gt_cells if gt_cells > 0 else 0.0
            metrics['cell_detection_ratio'] = min(cell_ratio, 1.0) if cell_ratio <= 1.0 else 1.0 / cell_ratio
        
        return metrics
    
    def compare_systems(self,
                       kg_rag_results: Dict[str, Any],
                       naive_rag_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        두 RAG 시스템 결과 비교
        
        Args:
            kg_rag_results: KG 기반 RAG 결과
            naive_rag_results: Naive RAG 결과
        
        Returns:
            비교 결과 딕셔너리
        """
        comparison = {
            'kg_rag': kg_rag_results,
            'naive_rag': naive_rag_results,
            'improvement': {}
        }
        
        # 각 메트릭에 대해 개선율 계산
        for metric in ['precision', 'recall', 'f1', 'mrr']:
            if metric in kg_rag_results and metric in naive_rag_results:
                kg_score = kg_rag_results[metric]
                naive_score = naive_rag_results[metric]
                if naive_score > 0:
                    improvement = ((kg_score - naive_score) / naive_score) * 100
                    comparison['improvement'][metric] = improvement
                else:
                    comparison['improvement'][metric] = float('inf') if kg_score > 0 else 0.0
        
        return comparison

