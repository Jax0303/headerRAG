#!/usr/bin/env python
"""
다중 데이터셋 실험 실행 스크립트
PubTables-1M, TabRecSet, KorWikiTabular 등 다양한 데이터셋으로 실험
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_experiments import ExperimentRunner
from utils.multi_dataset_loader import MultiDatasetLoader
import argparse


def main():
    parser = argparse.ArgumentParser(description='HeaderRAG 다중 데이터셋 실험')
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        choices=['pubtables1m', 'tabrecset', 'korwiki_tabular', 'rag_eval_ko'],
        default=['rag_eval_ko'],
        help='사용할 데이터셋 리스트'
    )
    parser.add_argument(
        '--max_tables_per_dataset',
        type=int,
        default=None,
        help='데이터셋당 최대 테이블 수 (None이면 전체)'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['1', '2', 'all'],
        default='all',
        help='실행할 실험'
    )
    parser.add_argument(
        '--include_baselines',
        action='store_true',
        default=True,
        help='베이스라인 포함'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("HeaderRAG 다중 데이터셋 실험")
    print("="*70)
    print(f"사용할 데이터셋: {args.datasets}")
    print(f"데이터셋당 최대 테이블 수: {args.max_tables_per_dataset or '전체'}")
    print("="*70)
    
    # 다중 데이터셋 로드
    loader = MultiDatasetLoader()
    tables, dataset_counts = loader.load_mixed_datasets(
        datasets=args.datasets,
        max_tables_per_dataset=args.max_tables_per_dataset
    )
    
    if len(tables) == 0:
        print("경고: 테이블이 로드되지 않았습니다.")
        return
    
    # 실험 러너 초기화
    runner = ExperimentRunner(output_dir="results", cycle_runs=1)
    
    # 실험 1: 파싱 성능 비교
    if args.experiment in ['1', 'all']:
        print("\n" + "="*70)
        print("[실험 1] 파싱 성능 비교")
        print("="*70)
        
        try:
            results_1 = runner.experiment_1_parsing_comparison(
                tables=tables,
                include_baselines=args.include_baselines
            )
            runner.save_results(results_1, "experiment_1_parsing")
            
            summary = results_1.get('summary', {})
            print("\n✓ 실험 1 완료")
            print(f"  평균 구조 풍부도: {summary.get('avg_structure_richness', 0):.3f}")
            print(f"  헤더 감지율: {summary.get('header_detection_rate', 0):.2%}")
        except Exception as e:
            print(f"✗ 실험 1 실패: {e}")
            import traceback
            traceback.print_exc()
    
    # 실험 2: RAG 성능 비교
    if args.experiment in ['2', 'all']:
        print("\n" + "="*70)
        print("[실험 2] RAG 성능 비교")
        print("="*70)
        
        # 쿼리 준비 (RAG-Evaluation-Dataset-KO 사용)
        from utils.prepare_rag_queries import prepare_queries_from_dataset
        
        try:
            queries, ground_truth = prepare_queries_from_dataset()
            valid_queries = [q for q in queries if ground_truth.get(q, [])][:50]
            valid_ground_truth = {q: ground_truth[q] for q in valid_queries}
            
            print(f"테스트 쿼리 수: {len(valid_queries)}")
            
            results_2 = runner.experiment_2_rag_comparison(
                tables=tables,
                queries=valid_queries,
                ground_truth=valid_ground_truth,
                include_baselines=args.include_baselines
            )
            runner.save_results(results_2, "experiment_2_rag")
            
            summary = results_2.get('summary', {})
            print("\n✓ 실험 2 완료")
            
            if 'kg_rag_avg' in summary:
                kg = summary['kg_rag_avg']
                print(f"  KG RAG - Precision: {kg.get('precision', 0):.3f}, Recall: {kg.get('recall', 0):.3f}")
            
            if 'naive_rag_avg' in summary:
                naive = summary['naive_rag_avg']
                print(f"  Naive RAG - Precision: {naive.get('precision', 0):.3f}, Recall: {naive.get('recall', 0):.3f}")
                
        except Exception as e:
            print(f"✗ 실험 2 실패: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("다중 데이터셋 실험 완료!")
    print("="*70)
    print(f"\n데이터셋별 테이블 수:")
    for name, count in dataset_counts.items():
        print(f"  {name}: {count}개")
    print(f"\n총 테이블 수: {len(tables)}개")


if __name__ == "__main__":
    main()



