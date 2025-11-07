"""
LLM 기반 평가 메트릭
Auto Evaluate 방식: 여러 LLM 평가를 사용하여 Voting
"""

import os
from typing import List, Dict, Any, Optional
import warnings
import numpy as np

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class LLMEvaluator:
    """
    LLM 기반 답변 평가 클래스
    
    RAG-Evaluation-Dataset-KO의 Auto Evaluate 방식을 구현:
    - TonicAI: answer_similarity (threshold=4)
    - MLflow: answer_similarity/v1/score (threshold=4)
    - MLflow: answer_correctness/v1/score (threshold=4)
    - Allganize Eval: answer_correctness/claude3-opus
    """
    
    def __init__(self, 
                 use_openai: bool = True,
                 use_anthropic: bool = True,
                 openai_api_key: Optional[str] = None,
                 anthropic_api_key: Optional[str] = None):
        """
        Args:
            use_openai: OpenAI API 사용 여부
            use_anthropic: Anthropic API 사용 여부
            openai_api_key: OpenAI API 키 (None이면 환경변수에서 읽음)
            anthropic_api_key: Anthropic API 키 (None이면 환경변수에서 읽음)
        """
        self.use_openai = use_openai and OPENAI_AVAILABLE
        self.use_anthropic = use_anthropic and ANTHROPIC_AVAILABLE
        
        if use_openai and not OPENAI_AVAILABLE:
            warnings.warn("OpenAI 패키지가 설치되지 않았습니다. pip install openai")
        
        if use_anthropic and not ANTHROPIC_AVAILABLE:
            warnings.warn("Anthropic 패키지가 설치되지 않았습니다. pip install anthropic")
        
        # API 키 설정
        if self.use_openai:
            if openai_api_key:
                openai.api_key = openai_api_key
            elif os.getenv("OPENAI_API_KEY"):
                openai.api_key = os.getenv("OPENAI_API_KEY")
            else:
                self.use_openai = False
                warnings.warn("OpenAI API 키가 설정되지 않았습니다.")
        
        if self.use_anthropic:
            if anthropic_api_key:
                self.anthropic_client = Anthropic(api_key=anthropic_api_key)
            elif os.getenv("ANTHROPIC_API_KEY"):
                self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            else:
                self.use_anthropic = False
                warnings.warn("Anthropic API 키가 설정되지 않았습니다.")
    
    def evaluate_answer_similarity(self,
                                  predicted_answer: str,
                                  target_answer: str,
                                  threshold: float = 4.0) -> Dict[str, Any]:
        """
        답변 유사도 평가 (TonicAI/MLflow 방식)
        
        Args:
            predicted_answer: 예측된 답변
            target_answer: 정답
            threshold: 통과 기준 점수 (1-5 스케일, threshold=4)
        
        Returns:
            평가 결과 딕셔너리
        """
        if not self.use_openai:
            # Fallback: 간단한 유사도 계산
            return self._fallback_similarity(predicted_answer, target_answer, threshold)
        
        try:
            # GPT-4를 사용한 유사도 평가 (1-5 스케일)
            prompt = f"""다음 두 답변의 유사도를 1-5 스케일로 평가하세요.

정답: {target_answer}
예측 답변: {predicted_answer}

1점: 완전히 다름
2점: 많이 다름
3점: 약간 다름
4점: 매우 유사함
5점: 거의 동일함

점수만 숫자로 답변하세요:"""
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert evaluator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            try:
                score = float(score_text)
            except ValueError:
                score = self._fallback_similarity(predicted_answer, target_answer, threshold)['score']
            
            is_correct = score >= threshold
            
            return {
                'score': score,
                'threshold': threshold,
                'correct': is_correct,
                'ox': 'O' if is_correct else 'X',
                'method': 'OpenAI GPT-4'
            }
            
        except Exception as e:
            warnings.warn(f"OpenAI 평가 실패: {e}. Fallback 사용.")
            return self._fallback_similarity(predicted_answer, target_answer, threshold)
    
    def evaluate_answer_correctness(self,
                                   predicted_answer: str,
                                   target_answer: str,
                                   use_claude: bool = True) -> Dict[str, Any]:
        """
        답변 정확도 평가 (Claude 방식)
        
        Args:
            predicted_answer: 예측된 답변
            target_answer: 정답
            use_claude: Claude 사용 여부
        
        Returns:
            평가 결과 딕셔너리
        """
        if use_claude and self.use_anthropic:
            try:
                prompt = f"""다음 예측 답변이 정답과 일치하는지 평가하세요.

정답: {target_answer}
예측 답변: {predicted_answer}

정답과 일치하면 "O", 일치하지 않으면 "X"로만 답변하세요."""
                
                message = self.anthropic_client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=10,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                result_text = message.content[0].text.strip().upper()
                is_correct = result_text == 'O'
                
                return {
                    'correct': is_correct,
                    'ox': 'O' if is_correct else 'X',
                    'method': 'Claude 3 Opus'
                }
                
            except Exception as e:
                warnings.warn(f"Claude 평가 실패: {e}. Fallback 사용.")
        
        # Fallback
        return self._fallback_correctness(predicted_answer, target_answer)
    
    def auto_evaluate(self,
                     predicted_answer: str,
                     target_answer: str,
                     voting_threshold: int = 3) -> Dict[str, Any]:
        """
        Auto Evaluate: 여러 평가 방법을 사용하여 Voting
        
        Args:
            predicted_answer: 예측된 답변
            target_answer: 정답
            voting_threshold: 통과를 위한 최소 O 개수 (기본값: 3/4)
        
        Returns:
            최종 평가 결과
        """
        results = []
        
        # 1. TonicAI 방식 (유사도 평가)
        result1 = self.evaluate_answer_similarity(predicted_answer, target_answer, threshold=4.0)
        result1['evaluator'] = 'TonicAI'
        results.append(result1)
        
        # 2. MLflow answer_similarity
        result2 = self.evaluate_answer_similarity(predicted_answer, target_answer, threshold=4.0)
        result2['evaluator'] = 'MLflow_similarity'
        results.append(result2)
        
        # 3. MLflow answer_correctness
        result3 = self.evaluate_answer_correctness(predicted_answer, target_answer, use_claude=False)
        result3['evaluator'] = 'MLflow_correctness'
        results.append(result3)
        
        # 4. Allganize Eval (Claude)
        result4 = self.evaluate_answer_correctness(predicted_answer, target_answer, use_claude=True)
        result4['evaluator'] = 'Allganize_Claude'
        results.append(result4)
        
        # Voting
        o_count = sum(1 for r in results if r.get('correct', False) or r.get('ox') == 'O')
        final_correct = o_count >= voting_threshold
        
        return {
            'final_ox': 'O' if final_correct else 'X',
            'final_correct': final_correct,
            'o_count': o_count,
            'total_evaluators': len(results),
            'voting_threshold': voting_threshold,
            'individual_results': results,
            'consensus': o_count >= voting_threshold
        }
    
    def _fallback_similarity(self,
                            predicted_answer: str,
                            target_answer: str,
                            threshold: float) -> Dict[str, Any]:
        """Fallback 유사도 평가 (간단한 텍스트 유사도)"""
        # 간단한 단어 겹침 기반 유사도
        pred_words = set(predicted_answer.lower().split())
        target_words = set(target_answer.lower().split())
        
        if not target_words:
            score = 3.0
        else:
            overlap = len(pred_words & target_words) / len(target_words)
            score = 1.0 + overlap * 4.0  # 1-5 스케일로 변환
        
        is_correct = score >= threshold
        
        return {
            'score': score,
            'threshold': threshold,
            'correct': is_correct,
            'ox': 'O' if is_correct else 'X',
            'method': 'Fallback (Word Overlap)'
        }
    
    def _fallback_correctness(self,
                             predicted_answer: str,
                             target_answer: str) -> Dict[str, Any]:
        """Fallback 정확도 평가"""
        # 간단한 정확도 체크
        pred_lower = predicted_answer.lower().strip()
        target_lower = target_answer.lower().strip()
        
        # 완전 일치 또는 주요 키워드 포함
        if pred_lower == target_lower:
            is_correct = True
        else:
            # 주요 단어가 포함되어 있는지 확인
            target_keywords = set(target_lower.split())
            pred_keywords = set(pred_lower.split())
            overlap_ratio = len(target_keywords & pred_keywords) / len(target_keywords) if target_keywords else 0
            is_correct = overlap_ratio >= 0.7
        
        return {
            'correct': is_correct,
            'ox': 'O' if is_correct else 'X',
            'method': 'Fallback (Keyword Overlap)'
        }


def main():
    """테스트 코드"""
    evaluator = LLMEvaluator()
    
    # 테스트 케이스
    predicted = "2023년 매출액은 1800억원입니다."
    target = "2023년 매출액은 1,800억원이다."
    
    result = evaluator.auto_evaluate(predicted, target)
    print("Auto Evaluate 결과:")
    print(f"  최종 판정: {result['final_ox']}")
    print(f"  O 개수: {result['o_count']}/{result['total_evaluators']}")
    print(f"  개별 결과:")
    for r in result['individual_results']:
        print(f"    - {r['evaluator']}: {r.get('ox', r.get('correct'))}")


if __name__ == "__main__":
    main()




