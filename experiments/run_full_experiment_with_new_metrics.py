#!/usr/bin/env python
"""
전체 데이터셋 실험 실행 스크립트 (새 메트릭 포함)
전체 데이터셋을 사용하여 실험 실행 및 결과 분석
"""

import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_experiments import ExperimentRunner
from experiments.integrate_new_metrics import (
    run_enhanced_experiment_1,
    run_enhanced_experiment_2,
    enhance_parsing_results_with_new_metrics,
    enhance_rag_results_with_new_metrics
)
from utils.prepare_rag_queries import prepare_queries_from_dataset
from experiments.result_analyzer import ResultAnalyzer
import json
import pandas as pd
import numpy as np
from datetime import datetime


def main():
    print("="*70)
    print("HeaderRAG 전체 데이터셋 실험 실행 (새 메트릭 포함)")
    print("="*70)
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # 실험 러너 초기화
    runner = ExperimentRunner(output_dir="results", cycle_runs=1)
    
    # 1. 전체 데이터셋 로드
    print("\n[1/5] 전체 데이터셋 로드 중...")
    tables = runner.load_test_data("", use_dataset=True)
    print(f"✓ 로드된 테이블 수: {len(tables)}개")
    
    if len(tables) == 0:
        print("에러: 테이블이 없습니다. 실험을 종료합니다.")
        sys.exit(1)
    
    results_summary = {
        'total_tables': len(tables),
        'experiment_timestamp': datetime.now().isoformat(),
        'experiment_1': None,
        'experiment_2': None
    }
    
    # 2. 실험 1: 파싱 성능 비교 (전체 데이터, 새 메트릭 포함)
    print("\n" + "="*70)
    print("[2/5] 실험 1: 파싱 성능 비교 (전체 데이터)")
    print("="*70)
    print(f"테이블 수: {len(tables)}개")
    
    try:
        results_1 = run_enhanced_experiment_1(
            tables=tables,
            include_baselines=True
        )
        
        # 결과 저장
        runner.save_results(results_1, "experiment_1_full_enhanced")
        
        # 요약 통계 계산
        parsing_summary = calculate_parsing_summary(results_1)
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
    print("[3/5] 쿼리 및 Ground Truth 준비")
    print("="*70)
    
    try:
        queries, ground_truth = prepare_queries_from_dataset()
        print(f"✓ 준비된 쿼리 수: {len(queries)}개")
        print(f"✓ Ground Truth 항목 수: {len(ground_truth)}개")
    except Exception as e:
        print(f"✗ 쿼리 준비 실패: {e}")
        print("기본 쿼리 사용...")
        queries = [
            "매출액은 얼마인가요?",
            "직원 수는 몇 명인가요?",
            "연도별 데이터를 보여주세요."
        ]
        ground_truth = {
            queries[0]: ["table_0"] if len(tables) > 0 else [],
            queries[1]: ["table_1"] if len(tables) > 1 else [],
            queries[2]: ["table_2"] if len(tables) > 2 else []
        }
    
    # 4. 실험 2: RAG 성능 비교 (전체 데이터, 새 메트릭 포함)
    print("\n" + "="*70)
    print("[4/5] 실험 2: RAG 성능 비교 (전체 데이터)")
    print("="*70)
    print(f"테이블 수: {len(tables)}개")
    print(f"쿼리 수: {len(queries)}개")
    
    try:
        results_2 = run_enhanced_experiment_2(
            tables=tables,
            queries=queries,
            ground_truth=ground_truth,
            include_baselines=True
        )
        
        # 결과 저장
        runner.save_results(results_2, "experiment_2_full_enhanced")
        
        # 요약 통계 계산
        rag_summary = calculate_rag_summary(results_2)
        results_summary['experiment_2'] = rag_summary
        
        print("\n✓ 실험 2 완료")
        print_rag_summary(rag_summary)
        
    except Exception as e:
        print(f"✗ 실험 2 실패: {e}")
        import traceback
        traceback.print_exc()
        results_2 = None
    
    # 5. 종합 리포트 생성
    print("\n" + "="*70)
    print("[5/5] 종합 리포트 생성")
    print("="*70)
    
    try:
        # 요약 저장
        summary_path = Path("results") / "full_experiment_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, ensure_ascii=False, indent=2, default=str)
        print(f"✓ 요약 저장: {summary_path}")
        
        # 결과 분석
        analyzer = ResultAnalyzer(output_dir="results/analysis")
        
        if results_1:
            # 파싱 성능 비교 테이블 생성
            try:
                parsing_performance = create_parsing_performance_table(results_1)
                if parsing_performance is not None:
                    analyzer.create_performance_table(
                        parsing_performance,
                        metrics=['grits_overall', 'header_f1', 'grits_content', 'grits_topology'],
                        output_path='results/analysis/parsing_performance_table.csv'
                    )
            except Exception as e:
                print(f"파싱 성능 테이블 생성 실패: {e}")
        
        if results_2:
            # RAG 성능 비교 테이블 생성
            try:
                rag_performance = create_rag_performance_table(results_2)
                if rag_performance is not None:
                    analyzer.create_performance_table(
                        rag_performance,
                        metrics=['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall'],
                        output_path='results/analysis/rag_performance_table.csv'
                    )
            except Exception as e:
                print(f"RAG 성능 테이블 생성 실패: {e}")
        
        print("✓ 종합 리포트 생성 완료")
        
    except Exception as e:
        print(f"✗ 리포트 생성 실패: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("전체 실험 완료!")
    print("="*70)
    print(f"\n결과 저장 위치:")
    print(f"  - 실험 1: results/experiment_1_full_enhanced/")
    print(f"  - 실험 2: results/experiment_2_full_enhanced/")
    print(f"  - 요약: results/full_experiment_summary.json")
    print(f"  - 분석: results/analysis/")


def calculate_parsing_summary(results: dict) -> dict:
    """파싱 실험 결과 요약 통계 계산"""
    summary = {}
    
    # 레이블링 파싱 메트릭 수집
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
                'std': metrics_df.std().to_dict(),
                'min': metrics_df.min().to_dict(),
                'max': metrics_df.max().to_dict()
            }
    
    # Naive 파싱 메트릭 수집
    if 'naive_parsing' in results:
        summary['naive_parsing'] = {
            'count': len(results['naive_parsing'])
        }
    
    return summary


def calculate_rag_summary(results: dict) -> dict:
    """RAG 실험 결과 요약 통계 계산"""
    summary = {}
    
    # KG RAG 메트릭 수집
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
    
    # Naive RAG 메트릭 수집
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


def create_parsing_performance_table(results: dict) -> dict:
    """파싱 성능 비교 테이블 생성"""
    performance = {}
    
    # 레이블링 파싱
    if 'labeled_parsing' in results:
        metrics_list = []
        for result in results['labeled_parsing']:
            if 'metrics' in result:
                metrics_list.append(result['metrics'])
        
        if metrics_list:
            metrics_df = pd.DataFrame(metrics_list)
            performance['HeaderRAG (Labeled)'] = metrics_df.mean().to_dict()
    
    # Naive 파싱 (간단한 통계만)
    if 'naive_parsing' in results:
        performance['Naive Parsing'] = {'count': len(results['naive_parsing'])}
    
    return performance if performance else None


def create_rag_performance_table(results: dict) -> dict:
    """RAG 성능 비교 테이블 생성"""
    performance = {}
    
    # KG RAG
    if 'kg_rag' in results:
        metrics_list = []
        for result in results['kg_rag']:
            if 'metrics' in result:
                metrics_list.append(result['metrics'])
        
        if metrics_list:
            metrics_df = pd.DataFrame(metrics_list)
            performance['KG-RAG (HeaderRAG)'] = metrics_df.mean().to_dict()
    
    # Naive RAG
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



