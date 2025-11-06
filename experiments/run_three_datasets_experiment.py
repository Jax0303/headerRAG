#!/usr/bin/env python
"""
3개 데이터셋 실험 실행 스크립트
- TabRecSet (Figshare)
- WTW-Dataset (GitHub)
- PubTables-1M (Microsoft Research)
"""

import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_experiments import ExperimentRunner
from experiments.integrate_new_metrics import (
    run_enhanced_experiment_1,
    run_enhanced_experiment_2
)
from utils.multi_dataset_loader import MultiDatasetLoader
from utils.prepare_rag_queries import prepare_queries_from_dataset
from experiments.result_analyzer import ResultAnalyzer
import json
import pandas as pd
from datetime import datetime


def main():
    print("="*70)
    print("HeaderRAG 3개 데이터셋 실험 실행")
    print("="*70)
    print("데이터셋:")
    print("  1. TabRecSet (Figshare)")
    print("  2. WTW-Dataset (GitHub)")
    print("  3. PubTables-1M (Microsoft Research)")
    print("="*70)
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # 데이터셋 로드
    print("\n[1/5] 데이터셋 로드 중...")
    loader = MultiDatasetLoader()
    
    all_tables = []
    dataset_counts = {}
    
    datasets_to_load = [
        ('tabrecset', lambda: loader.load_tabrecset(max_tables=None)),
        ('wtw', lambda: loader.load_wtw(split='train', max_tables=100)),  # 샘플만
        ('pubtables1m', lambda: loader.load_pubtables1m(max_tables=100)),  # 샘플만
    ]
    
    for dataset_name, load_func in datasets_to_load:
        print(f"\n  [{dataset_name}] 로드 중...")
        try:
            tables = load_func()
            dataset_counts[dataset_name] = len(tables)
            all_tables.extend(tables)
            print(f"    ✓ {len(tables)}개 테이블 로드 완료")
        except Exception as e:
            print(f"    ✗ {dataset_name} 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            dataset_counts[dataset_name] = 0
    
    print("\n" + "="*70)
    print("데이터셋 로드 요약")
    print("="*70)
    for name, count in dataset_counts.items():
        print(f"  {name}: {count}개")
    print(f"\n총 테이블 수: {len(all_tables)}개")
    
    if len(all_tables) == 0:
        print("\n경고: 테이블이 로드되지 않았습니다.")
        print("데이터셋 다운로드 상태를 확인하세요:")
        print("  - TabRecSet: data/tabrecset/DOWNLOAD_GUIDE.md")
        print("  - WTW: data/wtw/DOWNLOAD_GUIDE.md")
        print("  - PubTables-1M: data/pubtables1m/DOWNLOAD_GUIDE.md")
        return
    
    # 실험 러너 초기화
    runner = ExperimentRunner(output_dir="results", cycle_runs=1)
    
    results_summary = {
        'total_tables': len(all_tables),
        'dataset_counts': dataset_counts,
        'experiment_timestamp': datetime.now().isoformat(),
        'experiment_1': None,
        'experiment_2': None
    }
    
    # 실험 1: 파싱 성능 비교
    print("\n" + "="*70)
    print("[2/5] 실험 1: 파싱 성능 비교")
    print("="*70)
    print(f"테이블 수: {len(all_tables)}개")
    
    try:
        results_1 = run_enhanced_experiment_1(
            tables=all_tables,
            include_baselines=True
        )
        
        runner.save_results(results_1, "experiment_1_three_datasets")
        
        # 요약 통계
        parsing_summary = calculate_parsing_summary(results_1)
        parsing_summary['dataset_counts'] = dataset_counts
        results_summary['experiment_1'] = parsing_summary
        
        print("\n✓ 실험 1 완료")
        print_parsing_summary(parsing_summary)
        
    except Exception as e:
        print(f"✗ 실험 1 실패: {e}")
        import traceback
        traceback.print_exc()
        results_1 = None
    
    # 쿼리 준비
    print("\n" + "="*70)
    print("[3/5] 쿼리 및 Ground Truth 준비")
    print("="*70)
    
    try:
        queries, ground_truth = prepare_queries_from_dataset()
        print(f"✓ 준비된 쿼리 수: {len(queries)}개")
    except Exception as e:
        print(f"✗ 쿼리 준비 실패: {e}")
        queries = [f"테이블 {i} 분석" for i in range(min(50, len(all_tables)))]
        ground_truth = {q: [f"table_{i}"] for i, q in enumerate(queries)}
    
    # 실험 2: RAG 성능 비교
    print("\n" + "="*70)
    print("[4/5] 실험 2: RAG 성능 비교")
    print("="*70)
    print(f"테이블 수: {len(all_tables)}개")
    print(f"쿼리 수: {len(queries)}개")
    
    try:
        results_2 = run_enhanced_experiment_2(
            tables=all_tables,
            queries=queries,
            ground_truth=ground_truth,
            include_baselines=True
        )
        
        runner.save_results(results_2, "experiment_2_three_datasets")
        
        # 요약 통계
        rag_summary = calculate_rag_summary(results_2)
        rag_summary['dataset_counts'] = dataset_counts
        results_summary['experiment_2'] = rag_summary
        
        print("\n✓ 실험 2 완료")
        print_rag_summary(rag_summary)
        
    except Exception as e:
        print(f"✗ 실험 2 실패: {e}")
        import traceback
        traceback.print_exc()
        results_2 = None
    
    # 리포트 생성
    print("\n" + "="*70)
    print("[5/5] 종합 리포트 생성")
    print("="*70)
    
    try:
        summary_path = Path("results") / "three_datasets_experiment_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, ensure_ascii=False, indent=2, default=str)
        print(f"✓ 요약 저장: {summary_path}")
        
        analyzer = ResultAnalyzer(output_dir="results/analysis")
        
        if results_1:
            try:
                parsing_performance = create_parsing_performance_table(results_1)
                if parsing_performance:
                    analyzer.create_performance_table(
                        parsing_performance,
                        metrics=['grits_overall', 'header_f1'],
                        output_path='results/analysis/three_datasets_parsing_performance.csv'
                    )
            except Exception as e:
                print(f"파싱 성능 테이블 생성 실패: {e}")
        
        if results_2:
            try:
                rag_performance = create_rag_performance_table(results_2)
                if rag_performance:
                    analyzer.create_performance_table(
                        rag_performance,
                        metrics=['precision', 'recall', 'f1', 'mrr'],
                        output_path='results/analysis/three_datasets_rag_performance.csv'
                    )
            except Exception as e:
                print(f"RAG 성능 테이블 생성 실패: {e}")
        
        print("✓ 종합 리포트 생성 완료")
        
    except Exception as e:
        print(f"✗ 리포트 생성 실패: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("3개 데이터셋 실험 완료!")
    print("="*70)
    print(f"\n결과 저장 위치:")
    print(f"  - 실험 1: results/experiment_1_three_datasets/")
    print(f"  - 실험 2: results/experiment_2_three_datasets/")
    print(f"  - 요약: results/three_datasets_experiment_summary.json")


def calculate_parsing_summary(results: dict) -> dict:
    """파싱 실험 결과 요약"""
    summary = {}
    
    if 'labeled_parsing' in results:
        metrics_list = []
        for result in results['labeled_parsing']:
            if 'metrics' in result:
                metrics_list.append(result['metrics'])
        
        if metrics_list:
            metrics_df = pd.DataFrame(metrics_list)
            summary['labeled_parsing'] = {
                'count': len(metrics_list),
                'mean': metrics_df.mean().to_dict(),
                'std': metrics_df.std().to_dict()
            }
    
    return summary


def calculate_rag_summary(results: dict) -> dict:
    """RAG 실험 결과 요약"""
    summary = {}
    
    if 'kg_rag' in results:
        metrics_list = []
        for result in results['kg_rag']:
            if 'metrics' in result:
                metrics_list.append(result['metrics'])
        
        if metrics_list:
            metrics_df = pd.DataFrame(metrics_list)
            summary['kg_rag'] = {
                'count': len(metrics_list),
                'mean': metrics_df.mean().to_dict(),
                'std': metrics_df.std().to_dict()
            }
    
    if 'naive_rag' in results:
        metrics_list = []
        for result in results['naive_rag']:
            if 'metrics' in result:
                metrics_list.append(result['metrics'])
        
        if metrics_list:
            metrics_df = pd.DataFrame(metrics_list)
            summary['naive_rag'] = {
                'count': len(metrics_list),
                'mean': metrics_df.mean().to_dict(),
                'std': metrics_df.std().to_dict()
            }
    
    return summary


def print_parsing_summary(summary: dict):
    """파싱 요약 출력"""
    if 'labeled_parsing' in summary:
        lp = summary['labeled_parsing']
        print(f"\n레이블링 파싱 결과 ({lp['count']}개 테이블):")
        if 'mean' in lp:
            for metric, value in lp['mean'].items():
                std = lp['std'].get(metric, 0)
                print(f"  {metric}: {value:.4f} ± {std:.4f}")


def print_rag_summary(summary: dict):
    """RAG 요약 출력"""
    if 'kg_rag' in summary:
        kg = summary['kg_rag']
        print(f"\nKG RAG 결과 ({kg['count']}개 쿼리):")
        if 'mean' in kg:
            for metric, value in kg['mean'].items():
                std = kg['std'].get(metric, 0)
                print(f"  {metric}: {value:.4f} ± {std:.4f}")


def create_parsing_performance_table(results: dict) -> dict:
    """파싱 성능 테이블 생성"""
    performance = {}
    
    if 'labeled_parsing' in results:
        metrics_list = []
        for result in results['labeled_parsing']:
            if 'metrics' in result:
                metrics_list.append(result['metrics'])
        
        if metrics_list:
            metrics_df = pd.DataFrame(metrics_list)
            performance['HeaderRAG'] = metrics_df.mean().to_dict()
    
    return performance if performance else None


def create_rag_performance_table(results: dict) -> dict:
    """RAG 성능 테이블 생성"""
    performance = {}
    
    if 'kg_rag' in results:
        metrics_list = []
        for result in results['kg_rag']:
            if 'metrics' in result:
                metrics_list.append(result['metrics'])
        
        if metrics_list:
            metrics_df = pd.DataFrame(metrics_list)
            performance['KG-RAG'] = metrics_df.mean().to_dict()
    
    if 'naive_rag' in results:
        metrics_list = []
        for result in results['naive_rag']:
            if 'metrics' in result:
                metrics_list.append(result['metrics'])
        
        if metrics_list:
            metrics_df = pd.DataFrame(metrics_list)
            performance['Naive RAG'] = metrics_df.mean().to_dict()
    
    return performance if performance else None


if __name__ == '__main__':
    main()



