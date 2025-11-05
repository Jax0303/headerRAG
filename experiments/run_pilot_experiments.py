"""
파일럿 실험 실행 스크립트
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.pilot_experiments import PilotExperimentRunner
from experiments.statistical_analysis import StatisticalAnalyzer
from experiments.run_experiments import ExperimentRunner
import pandas as pd


def load_data(data_path: str = None, dataset_name: str = None):
    """데이터 로드"""
    runner = ExperimentRunner()
    
    if dataset_name:
        tables = runner.load_test_data(use_dataset=True, datasets=[dataset_name])
    elif data_path:
        tables = runner.load_test_data(data_path=data_path)
    else:
        # 기본 샘플 데이터 사용
        tables = runner.load_test_data()
    
    return tables


def main():
    parser = argparse.ArgumentParser(description="파일럿 실험 실행")
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['1a', '2a', '3a', 'ablation', 'all'],
        default='all',
        help='실행할 실험'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='데이터셋 이름 (pubtables1m, tabrecset, korwiki_tabular 등)'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        help='데이터 파일 경로'
    )
    parser.add_argument(
        '--samples_per_level',
        type=int,
        default=20,
        help='복잡도 레벨당 샘플 수'
    )
    parser.add_argument(
        '--max_tables',
        type=int,
        default=100,
        help='최대 테이블 수 (전체 실험용)'
    )
    
    args = parser.parse_args()
    
    # 데이터 로드
    print("데이터 로드 중...")
    tables = load_data(args.data_path, args.dataset)
    
    if args.max_tables and len(tables) > args.max_tables:
        import random
        random.seed(42)
        tables = random.sample(tables, args.max_tables)
        print(f"랜덤 샘플링: {len(tables)}개 테이블")
    
    # 실험 실행
    pilot_runner = PilotExperimentRunner()
    
    if args.experiment in ['1a', 'all']:
        print("\n=== 실험 1A: 파싱 성능 평가 ===")
        parsing_results = pilot_runner.experiment_1a_parsing(tables)
        pilot_runner.save_results(parsing_results, 'experiment_1a_parsing.json')
        
        # 요약 통계 출력
        if parsing_results.get('labeled_parsing'):
            metrics_list = []
            for result in parsing_results['labeled_parsing']:
                if 'metrics' in result:
                    metrics_list.append(result['metrics'])
            
            if metrics_list:
                import pandas as pd
                metrics_df = pd.DataFrame(metrics_list)
                print("\n파싱 메트릭 요약:")
                print(metrics_df.describe())
    
    if args.experiment in ['2a', 'all']:
        print("\n=== 실험 2A: RAG 성능 평가 ===")
        # 쿼리 로드 (예시)
        queries = [
            "매출액은 얼마인가요?",
            "직원 수는 몇 명인가요?",
            "연도별 데이터를 보여주세요."
        ]
        ground_truth = {
            queries[0]: ["table_0"],
            queries[1]: ["table_1"],
            queries[2]: ["table_2"]
        }
        
        rag_results = pilot_runner.experiment_2a_rag(tables, queries, ground_truth)
        pilot_runner.save_results(rag_results, 'experiment_2a_rag.json')
        
        # 요약 통계 출력
        if rag_results.get('comparison'):
            import pandas as pd
            comparison_list = []
            for comp in rag_results['comparison']:
                comparison_list.append(comp.get('improvement', {}))
            
            if comparison_list:
                comparison_df = pd.DataFrame(comparison_list)
                print("\nRAG 개선율 요약:")
                print(comparison_df.describe())
    
    if args.experiment in ['3a', 'all']:
        print("\n=== 실험 3A: 복잡도 분석 ===")
        sampled_tables = pilot_runner.stratified_sampling(
            tables,
            samples_per_level=args.samples_per_level
        )
        
        complexity_results = pilot_runner.experiment_3a_complexity_analysis(sampled_tables)
        pilot_runner.save_results(complexity_results, 'experiment_3a_complexity.json')
    
    if args.experiment in ['ablation', 'all']:
        print("\n=== Ablation Study: 파싱 ===")
        ablation_results = pilot_runner.ablation_study_parsing(tables)
        pilot_runner.save_results(ablation_results, 'ablation_study_parsing.json')
        
        # 통계 분석
        analyzer = StatisticalAnalyzer()
        
        # 각 ablation과 baseline 비교
        baseline_metrics = [
            r.get('metrics', {}).get('grits_overall', 0.0)
            for r in ablation_results.get('baseline_full', [])
            if 'metrics' in r
        ]
        
        for ablation_name in ['ablation_1_no_header', 'ablation_2_no_merged', 
                              'ablation_3_header_only', 'ablation_4_naive']:
            ablation_metrics = [
                r.get('metrics', {}).get('grits_overall', 0.0)
                for r in ablation_results.get(ablation_name, [])
                if 'metrics' in r
            ]
            
            if baseline_metrics and ablation_metrics:
                comparison = analyzer.compare_methods(
                    baseline_metrics,
                    ablation_metrics,
                    'Baseline (Full)',
                    ablation_name
                )
                print(f"\n{ablation_name} vs Baseline:")
                print(f"  평균 차이: {comparison['improvement']['absolute']:.4f}")
                print(f"  p-value: {comparison['test']['p_value']:.4f}")
                print(f"  유의미: {comparison['test']['is_significant']}")
                print(f"  Cohen's d: {comparison['effect_size']['cohens_d']:.4f}")
    
    print("\n파일럿 실험 완료!")


if __name__ == '__main__':
    main()

