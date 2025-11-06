"""
RAGAS 프레임워크 기반 RAG 평가 메트릭
- Faithfulness: 답변이 검색된 컨텍스트에 충실한지
- Answer Relevancy: 답변과 질문의 관련성
- Context Precision: 관련 문서가 상위 순위에 있는지
- Context Recall: Ground truth 대비 검색된 문서 비율
- Answer Correctness: Ground truth 대비 정답 정확도
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import warnings

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    warnings.warn("ragas 라이브러리가 설치되지 않았습니다. 설치: pip install ragas")


class RAGASMetrics:
    """RAGAS 프레임워크 기반 RAG 평가 메트릭"""
    
    def __init__(self, embedding_model=None):
        """
        Args:
            embedding_model: 임베딩 모델 (문장 유사도 계산용)
                None이면 기본 모델 사용
        """
        self.ragas_available = RAGAS_AVAILABLE
        self.embedding_model = embedding_model
        
        if embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            except ImportError:
                warnings.warn("sentence-transformers가 설치되지 않았습니다. 일부 메트릭이 제한될 수 있습니다.")
                self.embedding_model = None
    
    def calculate_faithfulness(self,
                              answer: str,
                              contexts: List[str]) -> float:
        """
        Faithfulness: 답변이 검색된 컨텍스트에 충실한지 평가
        
        Args:
            answer: 생성된 답변
            contexts: 검색된 컨텍스트 리스트
        
        Returns:
            Faithfulness 점수 (0-1)
        """
        if not answer or not contexts:
            return 0.0
        
        # 답변을 개별 주장(claims)으로 분해
        claims = self._extract_claims(answer)
        
        if not claims:
            return 1.0  # 주장이 없으면 완벽하게 충실하다고 간주
        
        # 각 주장이 컨텍스트에서 지원되는지 검증
        supported_claims = 0
        for claim in claims:
            if self._is_claim_supported(claim, contexts):
                supported_claims += 1
        
        # Faithfulness = |지원된 주장| / |전체 주장|
        faithfulness_score = supported_claims / len(claims) if claims else 1.0
        
        return faithfulness_score
    
    def _extract_claims(self, text: str) -> List[str]:
        """텍스트에서 개별 주장(claims) 추출"""
        # 문장 단위로 분리
        sentences = re.split(r'[.!?]\s+', text)
        claims = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        # 너무 짧은 문장 제거
        claims = [c for c in claims if len(c) > 10]
        
        return claims
    
    def _is_claim_supported(self, claim: str, contexts: List[str]) -> bool:
        """주장이 컨텍스트에서 지원되는지 확인"""
        if not self.embedding_model:
            # 단순 키워드 매칭
            claim_lower = claim.lower()
            for context in contexts:
                context_lower = context.lower()
                # 주요 키워드가 모두 포함되어 있는지 확인
                claim_words = set(claim_lower.split())
                context_words = set(context_lower.split())
                
                # 공통 키워드 비율이 30% 이상이면 지원된다고 간주
                if len(claim_words) > 0:
                    overlap = len(claim_words & context_words) / len(claim_words)
                    if overlap >= 0.3:
                        return True
            return False
        
        # 임베딩 기반 유사도 계산
        try:
            claim_embedding = self.embedding_model.encode([claim])
            context_embeddings = self.embedding_model.encode(contexts)
            
            similarities = cosine_similarity(claim_embedding, context_embeddings)[0]
            
            # 최대 유사도가 0.7 이상이면 지원된다고 간주
            return np.max(similarities) >= 0.7
        except Exception:
            # 실패 시 키워드 매칭으로 폴백
            return self._is_claim_supported(claim, contexts)
    
    def calculate_answer_relevancy(self,
                                  question: str,
                                  answer: str) -> float:
        """
        Answer Relevancy: 답변과 질문의 관련성 평가
        
        답변으로부터 역으로 질문을 생성하고 원래 질문과의 유사도를 계산
        
        Args:
            question: 원래 질문
            answer: 생성된 답변
        
        Returns:
            Answer Relevancy 점수 (0-1)
        """
        if not question or not answer:
            return 0.0
        
        if not self.embedding_model:
            # 단순 키워드 기반 유사도
            question_words = set(question.lower().split())
            answer_words = set(answer.lower().split())
            
            if not question_words:
                return 0.0
            
            overlap = len(question_words & answer_words) / len(question_words)
            return min(1.0, overlap * 2)  # 정규화
        
        # 임베딩 기반 유사도
        try:
            question_embedding = self.embedding_model.encode([question])
            answer_embedding = self.embedding_model.encode([answer])
            
            similarity = cosine_similarity(question_embedding, answer_embedding)[0][0]
            return max(0.0, min(1.0, similarity))
        except Exception:
            return 0.0
    
    def calculate_context_precision(self,
                                   retrieved_contexts: List[str],
                                   ground_truth_contexts: List[str]) -> float:
        """
        Context Precision: 관련 문서가 상위 순위에 있는지 평가
        
        Args:
            retrieved_contexts: 검색된 컨텍스트 리스트 (순서 중요)
            ground_truth_contexts: 정답 컨텍스트 리스트
        
        Returns:
            Context Precision 점수 (0-1)
        """
        if not retrieved_contexts:
            return 0.0
        
        if not ground_truth_contexts:
            return 0.0
        
        # 각 검색된 컨텍스트가 정답에 포함되는지 확인
        relevant_count = 0
        precision_sum = 0.0
        
        for i, context in enumerate(retrieved_contexts):
            if self._is_relevant(context, ground_truth_contexts):
                relevant_count += 1
                # Precision@i = 현재까지의 관련 문서 수 / 현재까지의 총 문서 수
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        # 평균 정밀도
        if relevant_count == 0:
            return 0.0
        
        return precision_sum / relevant_count
    
    def calculate_context_recall(self,
                                retrieved_contexts: List[str],
                                ground_truth_contexts: List[str]) -> float:
        """
        Context Recall: Ground truth 대비 검색된 문서 비율
        
        Args:
            retrieved_contexts: 검색된 컨텍스트 리스트
            ground_truth_contexts: 정답 컨텍스트 리스트
        
        Returns:
            Context Recall 점수 (0-1)
        """
        if not ground_truth_contexts:
            return 1.0 if not retrieved_contexts else 0.0
        
        # 정답 컨텍스트 중 검색된 것의 비율
        found_count = 0
        for gt_context in ground_truth_contexts:
            if self._is_relevant(gt_context, retrieved_contexts):
                found_count += 1
        
        return found_count / len(ground_truth_contexts)
    
    def calculate_answer_correctness(self,
                                    predicted_answer: str,
                                    ground_truth_answer: str,
                                    contexts: List[str]) -> float:
        """
        Answer Correctness: Ground truth 대비 정답 정확도
        
        Factual Correctness + Answer Similarity의 가중 합
        
        Args:
            predicted_answer: 예측된 답변
            ground_truth_answer: 정답 답변
            contexts: 검색된 컨텍스트
        
        Returns:
            Answer Correctness 점수 (0-1)
        """
        # Factual Correctness (Faithfulness와 유사)
        factual_correctness = self.calculate_faithfulness(predicted_answer, contexts)
        
        # Answer Similarity (예측 답변과 정답의 유사도)
        answer_similarity = self._calculate_answer_similarity(
            predicted_answer, 
            ground_truth_answer
        )
        
        # 가중 합 (Factual Correctness 60%, Answer Similarity 40%)
        correctness = 0.6 * factual_correctness + 0.4 * answer_similarity
        
        return correctness
    
    def _calculate_answer_similarity(self, answer1: str, answer2: str) -> float:
        """두 답변의 유사도 계산"""
        if not answer1 or not answer2:
            return 0.0
        
        if not self.embedding_model:
            # 단순 문자열 매칭
            answer1_lower = answer1.lower()
            answer2_lower = answer2.lower()
            
            if answer1_lower == answer2_lower:
                return 1.0
            
            # 부분 일치
            if answer1_lower in answer2_lower or answer2_lower in answer1_lower:
                return 0.8
            
            # 단어 오버랩
            words1 = set(answer1_lower.split())
            words2 = set(answer2_lower.split())
            
            if not words1 or not words2:
                return 0.0
            
            overlap = len(words1 & words2) / len(words1 | words2)
            return overlap
        
        # 임베딩 기반 유사도
        try:
            emb1 = self.embedding_model.encode([answer1])
            emb2 = self.embedding_model.encode([answer2])
            
            similarity = cosine_similarity(emb1, emb2)[0][0]
            return max(0.0, min(1.0, similarity))
        except Exception:
            return 0.0
    
    def _is_relevant(self, context: str, relevant_list: List[str]) -> bool:
        """컨텍스트가 관련 리스트에 포함되는지 확인"""
        if not self.embedding_model:
            # 단순 문자열 포함 여부
            context_lower = context.lower()
            for relevant in relevant_list:
                if context_lower in relevant.lower() or relevant.lower() in context_lower:
                    return True
            
            # 키워드 오버랩
            context_words = set(context_lower.split())
            for relevant in relevant_list:
                relevant_words = set(relevant.lower().split())
                if len(context_words & relevant_words) / max(len(context_words), 1) >= 0.5:
                    return True
            
            return False
        
        # 임베딩 기반 유사도
        try:
            context_emb = self.embedding_model.encode([context])
            relevant_embs = self.embedding_model.encode(relevant_list)
            
            similarities = cosine_similarity(context_emb, relevant_embs)[0]
            return np.max(similarities) >= 0.7
        except Exception:
            return False
    
    def evaluate_rag(self,
                     question: str,
                     answer: str,
                     contexts: List[str],
                     ground_truth_answer: Optional[str] = None,
                     ground_truth_contexts: Optional[List[str]] = None) -> Dict[str, float]:
        """
        종합 RAG 평가
        
        Args:
            question: 질문
            answer: 생성된 답변
            contexts: 검색된 컨텍스트
            ground_truth_answer: 정답 답변 (선택적)
            ground_truth_contexts: 정답 컨텍스트 (선택적)
        
        Returns:
            모든 RAGAS 메트릭을 포함한 딕셔너리
        """
        metrics = {}
        
        # Faithfulness
        metrics['faithfulness'] = self.calculate_faithfulness(answer, contexts)
        
        # Answer Relevancy
        metrics['answer_relevancy'] = self.calculate_answer_relevancy(question, answer)
        
        # Context Precision & Recall
        if ground_truth_contexts:
            metrics['context_precision'] = self.calculate_context_precision(
                contexts, ground_truth_contexts
            )
            metrics['context_recall'] = self.calculate_context_recall(
                contexts, ground_truth_contexts
            )
        
        # Answer Correctness
        if ground_truth_answer:
            metrics['answer_correctness'] = self.calculate_answer_correctness(
                answer, ground_truth_answer, contexts
            )
        
        return metrics
    
    def evaluate_with_ragas(self,
                           dataset: List[Dict[str, Any]],
                           metrics_list: Optional[List[str]] = None) -> Dict[str, float]:
        """
        RAGAS 라이브러리를 사용한 평가 (가능한 경우)
        
        Args:
            dataset: 평가 데이터셋
                각 항목은 다음 키를 포함해야 함:
                - question: 질문
                - answer: 답변
                - contexts: 컨텍스트 리스트
                - ground_truth: 정답 (선택적)
            metrics_list: 평가할 메트릭 리스트
        
        Returns:
            메트릭 점수 딕셔너리
        """
        if not self.ragas_available:
            warnings.warn("RAGAS 라이브러리를 사용할 수 없습니다. 기본 구현을 사용합니다.")
            # 기본 구현으로 폴백
            all_metrics = {}
            for item in dataset:
                item_metrics = self.evaluate_rag(
                    question=item.get('question', ''),
                    answer=item.get('answer', ''),
                    contexts=item.get('contexts', []),
                    ground_truth_answer=item.get('ground_truth', None),
                    ground_truth_contexts=item.get('ground_truth_contexts', None)
                )
                for key, value in item_metrics.items():
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
            
            # 평균 계산
            return {k: np.mean(v) for k, v in all_metrics.items()}
        
        # RAGAS 라이브러리 사용
        try:
            from datasets import Dataset
            
            # 데이터셋 형식 변환
            ragas_dataset = Dataset.from_list(dataset)
            
            # 메트릭 선택
            if metrics_list is None:
                metrics_list = [
                    faithfulness,
                    answer_relevancy,
                    context_precision
                ]
            
            # 평가 실행
            result = evaluate(ragas_dataset, metrics=metrics_list)
            
            return dict(result)
        except Exception as e:
            warnings.warn(f"RAGAS 평가 중 오류 발생: {e}. 기본 구현을 사용합니다.")
            return self.evaluate_with_ragas(dataset, metrics_list)



