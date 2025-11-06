"""
통계 분석 도구
- K-Fold Cross Validation
- Paired t-test
- Wilcoxon signed-rank test
- Bonferroni correction
- 효과 크기 계산 (Cohen's d)
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
import warnings


class StatisticalAnalyzer:
    """통계 분석 클래스"""
    
    def __init__(self):
        pass
    
    def k_fold_cross_validation(self,
                                data: List[Any],
                                k: int = 5,
                                random_state: int = 42) -> List[Tuple[List[int], List[int]]]:
        """
        K-Fold Cross Validation 분할
        
        Args:
            data: 데이터 리스트
            k: Fold 수
            random_state: 랜덤 시드
        
        Returns:
            (train_indices, test_indices) 튜플 리스트
        """
        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
        splits = list(kf.split(data))
        return splits
    
    def paired_t_test(self,
                     method1_scores: List[float],
                     method2_scores: List[float]) -> Dict[str, float]:
        """
        Paired t-test 수행
        
        Args:
            method1_scores: 방법 1의 점수 리스트
            method2_scores: 방법 2의 점수 리스트
        
        Returns:
            통계 결과 딕셔너리:
                - t_statistic: t 통계량
                - p_value: p-value
                - is_significant: 유의수준 0.05에서 유의미한지
                - mean_diff: 평균 차이
        """
        if len(method1_scores) != len(method2_scores):
            raise ValueError("두 방법의 점수 개수가 일치해야 합니다.")
        
        # 차이 계산
        differences = np.array(method1_scores) - np.array(method2_scores)
        
        # Paired t-test
        t_statistic, p_value = stats.ttest_rel(method1_scores, method2_scores)
        
        mean_diff = np.mean(differences)
        
        return {
            't_statistic': float(t_statistic),
            'p_value': float(p_value),
            'is_significant': p_value < 0.05,
            'mean_diff': float(mean_diff),
            'std_diff': float(np.std(differences))
        }
    
    def wilcoxon_test(self,
                     method1_scores: List[float],
                     method2_scores: List[float]) -> Dict[str, float]:
        """
        Wilcoxon signed-rank test 수행 (비정규분포 시)
        
        Args:
            method1_scores: 방법 1의 점수 리스트
            method2_scores: 방법 2의 점수 리스트
        
        Returns:
            통계 결과 딕셔너리
        """
        if len(method1_scores) != len(method2_scores):
            raise ValueError("두 방법의 점수 개수가 일치해야 합니다.")
        
        # 차이 계산
        differences = np.array(method1_scores) - np.array(method2_scores)
        
        # Wilcoxon test
        statistic, p_value = stats.wilcoxon(method1_scores, method2_scores)
        
        mean_diff = np.mean(differences)
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'is_significant': p_value < 0.05,
            'mean_diff': float(mean_diff),
            'median_diff': float(np.median(differences))
        }
    
    def cohens_d(self,
                method1_scores: List[float],
                method2_scores: List[float]) -> float:
        """
        Cohen's d 효과 크기 계산
        
        Args:
            method1_scores: 방법 1의 점수 리스트
            method2_scores: 방법 2의 점수 리스트
        
        Returns:
            Cohen's d 값
            - d < 0.2: 작은 효과
            - 0.2 <= d < 0.5: 중간 효과
            - 0.5 <= d < 0.8: 큰 효과
            - d >= 0.8: 매우 큰 효과
        """
        if len(method1_scores) != len(method2_scores):
            raise ValueError("두 방법의 점수 개수가 일치해야 합니다.")
        
        mean1 = np.mean(method1_scores)
        mean2 = np.mean(method2_scores)
        
        # 합동 표준편차 (pooled standard deviation)
        std1 = np.std(method1_scores, ddof=1)
        std2 = np.std(method2_scores, ddof=1)
        
        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
        
        if pooled_std == 0:
            return 0.0
        
        d = (mean1 - mean2) / pooled_std
        return float(d)
    
    def bonferroni_correction(self,
                            p_values: List[float],
                            alpha: float = 0.05) -> List[bool]:
        """
        Bonferroni correction (다중 비교 보정)
        
        Args:
            p_values: p-value 리스트
            alpha: 유의수준
        
        Returns:
            보정 후 유의미한지 여부 리스트
        """
        n_comparisons = len(p_values)
        corrected_alpha = alpha / n_comparisons
        
        return [p < corrected_alpha for p in p_values]
    
    def compare_methods(self,
                      method1_scores: List[float],
                      method2_scores: List[float],
                      method1_name: str = "Method 1",
                      method2_name: str = "Method 2",
                      use_wilcoxon: bool = False) -> Dict[str, Any]:
        """
        두 방법의 종합 비교
        
        Args:
            method1_scores: 방법 1의 점수 리스트
            method2_scores: 방법 2의 점수 리스트
            method1_name: 방법 1 이름
            method2_name: 방법 2 이름
            use_wilcoxon: Wilcoxon test 사용 여부 (정규분포가 아닐 때)
        
        Returns:
            종합 비교 결과 딕셔너리
        """
        # 기본 통계
        mean1 = np.mean(method1_scores)
        mean2 = np.mean(method2_scores)
        std1 = np.std(method1_scores)
        std2 = np.std(method2_scores)
        
        # 통계 검정
        if use_wilcoxon:
            test_result = self.wilcoxon_test(method1_scores, method2_scores)
            test_name = "Wilcoxon signed-rank test"
        else:
            test_result = self.paired_t_test(method1_scores, method2_scores)
            test_name = "Paired t-test"
        
        # 효과 크기
        cohens_d_value = self.cohens_d(method1_scores, method2_scores)
        
        # 효과 크기 해석
        if abs(cohens_d_value) < 0.2:
            effect_size_interp = "작은 효과"
        elif abs(cohens_d_value) < 0.5:
            effect_size_interp = "중간 효과"
        elif abs(cohens_d_value) < 0.8:
            effect_size_interp = "큰 효과"
        else:
            effect_size_interp = "매우 큰 효과"
        
        return {
            'method1_name': method1_name,
            'method2_name': method2_name,
            'method1_stats': {
                'mean': float(mean1),
                'std': float(std1),
                'n': len(method1_scores)
            },
            'method2_stats': {
                'mean': float(mean2),
                'std': float(std2),
                'n': len(method2_scores)
            },
            'test': {
                'name': test_name,
                **test_result
            },
            'effect_size': {
                'cohens_d': cohens_d_value,
                'interpretation': effect_size_interp
            },
            'improvement': {
                'absolute': float(mean1 - mean2),
                'relative_percent': float(((mean1 - mean2) / mean2 * 100) if mean2 > 0 else 0)
            }
        }
    
    def analyze_multiple_methods(self,
                                method_scores: Dict[str, List[float]],
                                baseline_method: str,
                                alpha: float = 0.05) -> Dict[str, Any]:
        """
        여러 방법과 베이스라인 비교
        
        Args:
            method_scores: {method_name: scores} 딕셔너리
            baseline_method: 베이스라인 방법 이름
            alpha: 유의수준
        
        Returns:
            비교 결과 딕셔너리
        """
        if baseline_method not in method_scores:
            raise ValueError(f"베이스라인 방법 '{baseline_method}'를 찾을 수 없습니다.")
        
        baseline_scores = method_scores[baseline_method]
        comparisons = {}
        p_values = []
        
        for method_name, scores in method_scores.items():
            if method_name == baseline_method:
                continue
            
            comparison = self.compare_methods(
                scores,
                baseline_scores,
                method_name,
                baseline_method
            )
            comparisons[method_name] = comparison
            p_values.append(comparison['test']['p_value'])
        
        # Bonferroni correction
        corrected_significant = self.bonferroni_correction(p_values, alpha)
        
        # 결과에 보정 정보 추가
        for i, method_name in enumerate(comparisons.keys()):
            comparisons[method_name]['test']['bonferroni_significant'] = corrected_significant[i]
        
        return {
            'baseline': baseline_method,
            'comparisons': comparisons,
            'alpha': alpha,
            'bonferroni_corrected_alpha': alpha / len(comparisons) if comparisons else alpha
        }



