"""
새 메트릭을 기존 실험 시스템에 통합
"""

import sys
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.run_experiments import ExperimentRunner
from src.evaluation.parsing_metrics import ParsingMetrics
from src.evaluation.ragas_metrics import RAGASMetrics
from src.evaluation.complexity_metrics import ComplexityMetrics


def enhance_parsing_results_with_new_metrics(
    parsing_results: Dict[str, Any],
    ground_truth: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    기존 파싱 결과에 새로운 메트릭 추가
    
    Args:
        parsing_results: 기존 실험 결과
        ground_truth: 정답 구조 (선택적)
    
    Returns:
        새로운 메트릭이 추가된 결과
    """
    parsing_metrics = ParsingMetrics()
    complexity_metrics = ComplexityMetrics()
    
    enhanced_results = parsing_results.copy()
    
    # 레이블링 파싱 결과에 메트릭 추가
    if 'labeled_parsing' in enhanced_results:
        for i, result in enumerate(enhanced_results['labeled_parsing']):
            if 'structure' in result:
                structure = result['structure']
                
                # 복잡도 계산
                complexity = complexity_metrics.calculate_complexity(structure)
                result['complexity'] = complexity
                
                # 정답이 있으면 GriTS 등 계산
                if ground_truth and i < len(ground_truth):
                    gt_structure = ground_truth[i]
                    
                    # 새 메트릭 계산
                    new_metrics = parsing_metrics.evaluate_parsing(
                        predicted_table=structure,
                        ground_truth_table=gt_structure
                    )
                    
                    # 기존 metrics가 있으면 업데이트, 없으면 생성
                    if 'metrics' in result:
                        result['metrics'].update(new_metrics)
                    else:
                        result['metrics'] = new_metrics
    
    return enhanced_results


def enhance_rag_results_with_new_metrics(
    rag_results: Dict[str, Any],
    questions: List[str] = None
) -> Dict[str, Any]:
    """
    기존 RAG 결과에 새로운 메트릭 추가
    
    Args:
        rag_results: 기존 실험 결과
        questions: 질문 리스트 (선택적)
    
    Returns:
        새로운 메트릭이 추가된 결과
    """
    ragas_metrics = RAGASMetrics()
    
    enhanced_results = rag_results.copy()
    
    # KG RAG 결과에 메트릭 추가
    if 'kg_rag' in enhanced_results and questions:
        for i, result in enumerate(enhanced_results['kg_rag']):
            if i < len(questions):
                question = questions[i]
                answer = result.get('answer', '')
                contexts = [r.get('context', '') for r in result.get('results', [])]
                
                # RAGAS 메트릭 계산
                ragas_results = ragas_metrics.evaluate_rag(
                    question=question,
                    answer=answer,
                    contexts=contexts
                )
                
                # 기존 metrics 업데이트
                if 'metrics' in result:
                    result['metrics'].update(ragas_results)
                else:
                    result['metrics'] = ragas_results
    
    # Naive RAG에도 동일하게 적용
    if 'naive_rag' in enhanced_results and questions:
        for i, result in enumerate(enhanced_results['naive_rag']):
            if i < len(questions):
                question = questions[i]
                answer = result.get('answer', '')
                contexts = [r.get('context', '') for r in result.get('results', [])]
                
                ragas_results = ragas_metrics.evaluate_rag(
                    question=question,
                    answer=answer,
                    contexts=contexts
                )
                
                if 'metrics' in result:
                    result['metrics'].update(ragas_results)
                else:
                    result['metrics'] = ragas_results
    
    return enhanced_results


def run_enhanced_experiment_1(
    tables: List[pd.DataFrame],
    include_baselines: bool = True
) -> Dict[str, Any]:
    """
    향상된 실험 1 실행 (새 메트릭 포함)
    
    Args:
        tables: 테이블 리스트
        include_baselines: 베이스라인 포함 여부
    
    Returns:
        향상된 실험 결과
    """
    print("향상된 실험 1 실행 중...")
    
    # 기존 실험 실행
    runner = ExperimentRunner()
    base_results = runner.experiment_1_parsing_comparison(
        tables=tables,
        include_baselines=include_baselines
    )
    
    # 새 메트릭 추가
    enhanced_results = enhance_parsing_results_with_new_metrics(base_results)
    
    return enhanced_results


def run_enhanced_experiment_2(
    tables: List[pd.DataFrame],
    queries: List[str],
    ground_truth: Dict[str, List[str]],
    include_baselines: bool = True
) -> Dict[str, Any]:
    """
    향상된 실험 2 실행 (새 메트릭 포함)
    
    Args:
        tables: 테이블 리스트
        queries: 질문 리스트
        ground_truth: 질문별 정답 테이블 ID
        include_baselines: 베이스라인 포함 여부
    
    Returns:
        향상된 실험 결과
    """
    print("향상된 실험 2 실행 중...")
    
    # 기존 실험 실행
    runner = ExperimentRunner()
    base_results = runner.experiment_2_rag_comparison(
        tables=tables,
        queries=queries,
        ground_truth=ground_truth,
        include_baselines=include_baselines
    )
    
    # 새 메트릭 추가
    enhanced_results = enhance_rag_results_with_new_metrics(base_results, queries)
    
    return enhanced_results


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="새 메트릭 통합 실험")
    parser.add_argument('--experiment', type=str, choices=['1', '2', 'all'], default='all')
    parser.add_argument('--dataset', type=str, help='데이터셋 이름')
    parser.add_argument('--max_tables', type=int, default=20, help='최대 테이블 수')
    
    args = parser.parse_args()
    
    # 데이터 로드
    runner = ExperimentRunner()
    if args.dataset:
        tables = runner.load_test_data(use_dataset=True, datasets=[args.dataset])
    else:
        tables = runner.load_test_data(use_dataset=True)
    
    if args.max_tables:
        tables = tables[:args.max_tables]
    
    print(f"로드된 테이블 수: {len(tables)}")
    
    # 실험 실행
    if args.experiment in ['1', 'all']:
        print("\n=== 향상된 실험 1 실행 ===")
        results_1 = run_enhanced_experiment_1(tables, include_baselines=True)
        runner.save_results(results_1, "experiment_1_enhanced")
        print("✓ 실험 1 완료")
    
    if args.experiment in ['2', 'all']:
        print("\n=== 향상된 실험 2 실행 ===")
        # 샘플 쿼리 (실제로는 데이터셋에서 로드)
        queries = ["매출액은 얼마인가요?", "직원 수는?"]
        ground_truth = {queries[0]: ["table_0"], queries[1]: ["table_1"]}
        
        results_2 = run_enhanced_experiment_2(
            tables, queries, ground_truth, include_baselines=True
        )
        runner.save_results(results_2, "experiment_2_enhanced")
        print("✓ 실험 2 완료")
    
    print("\n✓ 모든 실험 완료!")


if __name__ == '__main__':
    main()



