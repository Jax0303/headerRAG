#!/usr/bin/env python
"""
모든 데이터셋을 사용한 종합 실험 실행 스크립트
PubTables-1M, TabRecSet, KorWikiTabular, RAG-Evaluation-Dataset-KO 전체 사용
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
    print("HeaderRAG 모든 데이터셋 종합 실험 실행")
    print("="*70)
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # 사용 가능한 모든 데이터셋
    available_datasets = ['pubtables1m', 'tabrecset', 'korwiki_tabular', 'rag_eval_ko']
    
    print(f"\n사용할 데이터셋: {', '.join(available_datasets)}")
    
    # 다중 데이터셋 로드
    print("\n[1/6] 모든 데이터셋 로드 중...")
    loader = MultiDatasetLoader()
    
    all_tables = []
    dataset_counts = {}
    
    for dataset_name in available_datasets:
        print(f"\n  [{dataset_name}] 로드 중...")
        try:
            if dataset_name == 'rag_eval_ko':
                # RAG-Evaluation-Dataset-KO는 별도 로더 사용
                runner = ExperimentRunner()
                tables = runner.load_test_data("", use_dataset=True)
            else:
                # None이면 전체 로드
                if dataset_name == 'pubtables1m':
                    tables = loader.load_pubtables1m(max_tables=None)
                elif dataset_name == 'tabrecset':
                    tables = loader.load_tabrecset(max_tables=None)
                elif dataset_name == 'korwiki_tabular':
                    tables = loader.load_korwiki_tabular(max_tables=None)
                else:
                    tables = []
            
            dataset_counts[dataset_name] = len(tables)
            all_tables.extend(tables)
            print(f"    ✓ {len(tables)}개 테이블 로드 완료")
        except Exception as e:
            print(f"    ✗ {dataset_name} 로드 실패: {e}")
            dataset_counts[dataset_name] = 0
    
    print("\n" + "="*70)
    print("데이터셋 로드 요약")
    print("="*70)
    for name, count in dataset_counts.items():
        print(f"  {name}: {count}개")
    print(f"\n총 테이블 수: {len(all_tables)}개")
    
    if len(all_tables) == 0:
        print("\n경고: 테이블이 로드되지 않았습니다. 실험을 종료합니다.")
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
    
    # 2. 실험 1: 파싱 성능 비교 (모든 데이터셋)
    print("\n" + "="*70)
    print("[2/6] 실험 1: 파싱 성능 비교 (모든 데이터셋)")
    print("="*70)
    print(f"테이블 수: {len(all_tables)}개")
    
    try:
        results_1 = run_enhanced_experiment_1(
            tables=all_tables,
            include_baselines=True
        )
        
        # 결과 저장
        runner.save_results(results_1, "experiment_1_all_datasets")
        
        # 요약 통계 계산
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
    
    # 3. 쿼리 및 Ground Truth 준비
    print("\n" + "="*70)
    print("[3/6] 쿼리 및 Ground Truth 준비")
    print("="*70)
    
    try:
        queries, ground_truth = prepare_queries_from_dataset()
        print(f"✓ 준비된 쿼리 수: {len(queries)}개")
        print(f"✓ Ground Truth 항목 수: {len(ground_truth)}개")
    except Exception as e:
        print(f"✗ 쿼리 준비 실패: {e}")
        print("기본 쿼리 사용...")
        # 기본 쿼리 생성 (데이터셋별)
        queries = []
        ground_truth = {}
        
        # 각 데이터셋에서 몇 개씩 쿼리 생성
        for i, table in enumerate(all_tables[:50]):  # 최대 50개
            queries.append(f"테이블 {i}의 데이터를 분석해주세요")
            ground_truth[queries[-1]] = [f"table_{i}"]
    
    # 4. 실험 2: RAG 성능 비교 (모든 데이터셋)
    print("\n" + "="*70)
    print("[4/6] 실험 2: RAG 성능 비교 (모든 데이터셋)")
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
        
        # 결과 저장
        runner.save_results(results_2, "experiment_2_all_datasets")
        
        # 요약 통계 계산
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
    
    # 5. 데이터셋별 성능 분석
    print("\n" + "="*70)
    print("[5/6] 데이터셋별 성능 분석")
    print("="*70)
    
    try:
        dataset_analysis = analyze_by_dataset(
            results_1, results_2, dataset_counts, all_tables
        )
        results_summary['dataset_analysis'] = dataset_analysis
        
        print("\n데이터셋별 성능 요약:")
        for dataset_name, analysis in dataset_analysis.items():
            if analysis:
                print(f"\n  [{dataset_name}]")
                if 'parsing' in analysis:
                    print(f"    파싱 테이블 수: {analysis['parsing'].get('count', 0)}")
                if 'rag' in analysis:
                    rag = analysis['rag']
                    if 'mean' in rag:
                        print(f"    RAG Precision: {rag['mean'].get('precision', 0):.4f}")
                        print(f"    RAG Recall: {rag['mean'].get('recall', 0):.4f}")
                        print(f"    RAG F1: {rag['mean'].get('f1', 0):.4f}")
    except Exception as e:
        print(f"✗ 데이터셋별 분석 실패: {e}")
    
    # 6. 종합 리포트 생성
    print("\n" + "="*70)
    print("[6/6] 종합 리포트 생성")
    print("="*70)
    
    try:
        # 요약 저장
        summary_path = Path("results") / "all_datasets_experiment_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, ensure_ascii=False, indent=2, default=str)
        print(f"✓ 요약 저장: {summary_path}")
        
        # 결과 분석
        analyzer = ResultAnalyzer(output_dir="results/analysis")
        
        if results_1:
            try:
                parsing_performance = create_parsing_performance_table(results_1)
                if parsing_performance:
                    analyzer.create_performance_table(
                        parsing_performance,
                        metrics=['grits_overall', 'header_f1', 'grits_content'],
                        output_path='results/analysis/all_datasets_parsing_performance.csv'
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
                        output_path='results/analysis/all_datasets_rag_performance.csv'
                    )
            except Exception as e:
                print(f"RAG 성능 테이블 생성 실패: {e}")
        
        print("✓ 종합 리포트 생성 완료")
        
    except Exception as e:
        print(f"✗ 리포트 생성 실패: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("모든 데이터셋 실험 완료!")
    print("="*70)
    print(f"\n결과 저장 위치:")
    print(f"  - 실험 1: results/experiment_1_all_datasets/")
    print(f"  - 실험 2: results/experiment_2_all_datasets/")
    print(f"  - 요약: results/all_datasets_experiment_summary.json")
    print(f"  - 분석: results/analysis/")


def calculate_parsing_summary(results: dict) -> dict:
    """파싱 실험 결과 요약 통계 계산"""
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
    
    if 'naive_parsing' in results:
        summary['naive_parsing'] = {'count': len(results['naive_parsing'])}
    
    return summary


def calculate_rag_summary(results: dict) -> dict:
    """RAG 실험 결과 요약 통계 계산"""
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
    
    if 'naive_rag' in summary:
        naive = summary['naive_rag']
        print(f"\nNaive RAG 결과 ({naive['count']}개 쿼리):")
        if 'mean' in naive:
            for metric, value in naive['mean'].items():
                std = naive['std'].get(metric, 0)
                print(f"  {metric}: {value:.4f} ± {std:.4f}")


def analyze_by_dataset(results_1, results_2, dataset_counts, all_tables):
    """데이터셋별 성능 분석"""
    analysis = {}
    
    # 데이터셋별 테이블 인덱스 추적
    current_idx = 0
    dataset_ranges = {}
    
    for dataset_name, count in dataset_counts.items():
        if count > 0:
            dataset_ranges[dataset_name] = (current_idx, current_idx + count)
            current_idx += count
    
    # 파싱 결과 분석
    if results_1 and 'labeled_parsing' in results_1:
        for dataset_name, (start_idx, end_idx) in dataset_ranges.items():
            if dataset_name not in analysis:
                analysis[dataset_name] = {}
            
            dataset_results = results_1['labeled_parsing'][start_idx:end_idx]
            metrics_list = [r.get('metrics', {}) for r in dataset_results if 'metrics' in r]
            
            if metrics_list:
                metrics_df = pd.DataFrame(metrics_list)
                analysis[dataset_name]['parsing'] = {
                    'count': len(metrics_list),
                    'mean': metrics_df.mean().to_dict(),
                    'std': metrics_df.std().to_dict()
                }
    
    # RAG 결과 분석 (전체 쿼리에 대해)
    if results_2 and 'kg_rag' in results_2:
        # RAG는 쿼리 기반이므로 데이터셋별로 나누기 어려움
        # 전체 성능만 기록
        pass
    
    return analysis


def create_parsing_performance_table(results: dict) -> dict:
    """파싱 성능 비교 테이블 생성"""
    performance = {}
    
    if 'labeled_parsing' in results:
        metrics_list = []
        for result in results['labeled_parsing']:
            if 'metrics' in result:
                metrics_list.append(result['metrics'])
        
        if metrics_list:
            metrics_df = pd.DataFrame(metrics_list)
            performance['HeaderRAG (Labeled)'] = metrics_df.mean().to_dict()
    
    return performance if performance else None


def create_rag_performance_table(results: dict) -> dict:
    """RAG 성능 비교 테이블 생성"""
    performance = {}
    
    if 'kg_rag' in results:
        metrics_list = []
        for result in results['kg_rag']:
            if 'metrics' in result:
                metrics_list.append(result['metrics'])
        
        if metrics_list:
            metrics_df = pd.DataFrame(metrics_list)
            performance['KG-RAG (HeaderRAG)'] = metrics_df.mean().to_dict()
    
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




