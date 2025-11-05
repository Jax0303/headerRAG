#!/usr/bin/env python
"""
비교 실험 실행 스크립트
베이스라인 모델과의 비교 실험을 쉽게 실행할 수 있는 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_experiments import ExperimentRunner
import argparse


def main():
    parser = argparse.ArgumentParser(description='HeaderRAG 비교 실험 실행')
    parser.add_argument(
        '--data_path',
        type=str,
        default='',
        help='테이블 데이터 경로 (기본값: 빈 문자열, use_dataset=True 권장)'
    )
    parser.add_argument(
        '--use_dataset',
        action='store_true',
        help='실제 평가 데이터셋 사용 (RAG-Evaluation-Dataset-KO)'
    )
    parser.add_argument(
        '--include_baselines',
        action='store_true',
        default=True,
        help='베이스라인 모델 포함 (기본값: True)'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['1', '2', '3', 'all'],
        default='all',
        help='실행할 실험 선택 (1: 파싱, 2: RAG, 3: 복잡도, all: 전체)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='결과 저장 디렉토리 (기본값: results)'
    )
    parser.add_argument(
        '--cycle_runs',
        type=int,
        default=10,
        help='사이클당 실행 횟수 (기본값: 10)'
    )
    
    args = parser.parse_args()
    
    # 실험 러너 초기화
    print("="*60)
    print("HeaderRAG 비교 실험 시작")
    print("="*60)
    print(f"데이터 경로: {args.data_path}")
    print(f"베이스라인 포함: {args.include_baselines}")
    print(f"실험 선택: {args.experiment}")
    print("="*60)
    
    runner = ExperimentRunner(
        output_dir=args.output_dir,
        cycle_runs=args.cycle_runs
    )
    
    # 테이블 데이터 로드
    print("\n[1/4] 테이블 데이터 로드 중...")
    tables = runner.load_test_data(args.data_path, use_dataset=args.use_dataset)
    print(f"✓ 로드된 테이블 수: {len(tables)}")
    
    if len(tables) == 0:
        print("경고: 테이블이 로드되지 않았습니다. 데이터 경로를 확인하세요.")
        return
    
    # 실험 1: 파싱 성능 비교
    if args.experiment in ['1', 'all']:
        print("\n" + "="*60)
        print("[2/4] 실험 1: 파싱 성능 비교")
        print("="*60)
        print("비교 대상:")
        print("  - 레이블링 기반 파싱 (HeaderRAG)")
        print("  - Naive 파싱")
        if args.include_baselines:
            print("  - TATR (베이스라인)")
            print("  - Sato (베이스라인)")
        
        try:
            results_1 = runner.experiment_1_parsing_comparison(
                tables=tables,
                include_baselines=args.include_baselines
            )
            runner.save_results(results_1, "experiment_1_parsing")
            
            # 요약 출력
            summary = results_1.get('summary', {})
            print("\n✓ 실험 1 완료")
            print(f"  평균 구조 풍부도: {summary.get('avg_structure_richness', 0):.3f}")
            print(f"  헤더 감지율: {summary.get('header_detection_rate', 0):.2%}")
            
            if 'tatr_parsing' in results_1:
                print(f"  TATR 파싱 결과: {len(results_1['tatr_parsing'])}개 테이블")
            if 'sato_semantic' in results_1:
                print(f"  Sato 검출 결과: {len(results_1['sato_semantic'])}개 테이블")
                
        except Exception as e:
            print(f"✗ 실험 1 실패: {e}")
            import traceback
            traceback.print_exc()
    
    # 실험 2: RAG 성능 비교
    if args.experiment in ['2', 'all']:
        print("\n" + "="*60)
        print("[3/4] 실험 2: RAG 성능 비교")
        print("="*60)
        print("비교 대상:")
        print("  - KG 기반 RAG (HeaderRAG)")
        print("  - Naive RAG")
        if args.include_baselines:
            print("  - TableRAG (베이스라인)")
        
        # 실제 데이터셋에서 쿼리와 ground truth 추출
        from utils.prepare_rag_queries import prepare_queries_from_dataset
        
        try:
            print("\n실제 데이터셋에서 쿼리 및 ground truth 추출 중...")
            queries, ground_truth = prepare_queries_from_dataset()
            
            # Ground truth가 있는 쿼리만 필터링
            valid_queries = [q for q in queries if ground_truth.get(q, [])]
            valid_ground_truth = {q: ground_truth[q] for q in valid_queries}
            
            # 전체 쿼리 사용 (또는 처음 100개)
            if len(valid_queries) > 100:
                print(f"정보: 쿼리가 많습니다 ({len(valid_queries)}개). 처음 100개를 사용합니다.")
                valid_queries = valid_queries[:100]
                valid_ground_truth = {q: valid_ground_truth[q] for q in valid_queries}
            else:
                print(f"정보: 전체 {len(valid_queries)}개 쿼리를 사용합니다.")
            
            queries = valid_queries
            ground_truth = valid_ground_truth
            
            print(f"✓ 유효한 쿼리 수: {len(queries)}개")
            print(f"✓ Ground truth가 있는 쿼리: {len([q for q in queries if ground_truth.get(q, [])])}개")
            
            # 샘플 출력
            if queries:
                print(f"\n샘플 쿼리 (처음 3개):")
                for i, q in enumerate(queries[:3], 1):
                    gt = ground_truth.get(q, [])
                    print(f"  {i}. {q[:60]}...")
                    print(f"     Ground truth: {gt}")
            
        except Exception as e:
            print(f"경고: 데이터셋에서 쿼리 추출 실패: {e}")
            print("샘플 쿼리로 대체합니다.")
            import traceback
            traceback.print_exc()
            
            # Fallback: 샘플 쿼리
            queries = [
                "2023년 매출액은 얼마인가요?",
                "직원 수가 가장 많은 연도는?",
                "순이익이 가장 높은 연도는?"
            ]
            ground_truth = {
                queries[0]: ["table_0"] if len(tables) > 0 else [],
                queries[1]: ["table_0"] if len(tables) > 0 else [],
                queries[2]: ["table_0"] if len(tables) > 0 else []
            }
        
        print(f"\n테스트 쿼리 수: {len(queries)}")
        
        try:
            results_2 = runner.experiment_2_rag_comparison(
                tables=tables,
                queries=queries,
                ground_truth=ground_truth,
                include_baselines=args.include_baselines
            )
            runner.save_results(results_2, "experiment_2_rag")
            
            # 요약 출력
            summary = results_2.get('summary', {})
            print("\n✓ 실험 2 완료")
            
            kg_avg = summary.get('kg_rag_avg', {})
            print(f"\n  KG RAG:")
            print(f"    Precision: {kg_avg.get('precision', 0):.3f}")
            print(f"    Recall: {kg_avg.get('recall', 0):.3f}")
            print(f"    F1: {kg_avg.get('f1', 0):.3f}")
            
            naive_avg = summary.get('naive_rag_avg', {})
            print(f"\n  Naive RAG:")
            print(f"    Precision: {naive_avg.get('precision', 0):.3f}")
            print(f"    Recall: {naive_avg.get('recall', 0):.3f}")
            print(f"    F1: {naive_avg.get('f1', 0):.3f}")
            
            if 'tablerag_baseline' in results_2 and results_2['tablerag_baseline']:
                import numpy as np
                tablerag_metrics = [r['metrics'] for r in results_2['tablerag_baseline']]
                print(f"\n  TableRAG:")
                print(f"    Precision: {np.mean([m.get('precision', 0) for m in tablerag_metrics]):.3f}")
                print(f"    Recall: {np.mean([m.get('recall', 0) for m in tablerag_metrics]):.3f}")
                print(f"    F1: {np.mean([m.get('f1', 0) for m in tablerag_metrics]):.3f}")
                
        except Exception as e:
            print(f"✗ 실험 2 실패: {e}")
            import traceback
            traceback.print_exc()
    
    # 실험 3: 복잡도 분석
    if args.experiment in ['3', 'all']:
        print("\n" + "="*60)
        print("[4/4] 실험 3: 복잡도 분석")
        print("="*60)
        
        try:
            results_3 = runner.experiment_3_complexity_analysis(tables)
            runner.save_results(results_3, "experiment_3_complexity")
            
            # 요약 출력
            summary = results_3.get('summary', {})
            complexity_dist = results_3.get('complexity_distribution', {})
            
            print("\n✓ 실험 3 완료")
            print(f"  복잡도 분포:")
            for comp_type, count in complexity_dist.items():
                print(f"    {comp_type}: {count}개")
                
        except Exception as e:
            print(f"✗ 실험 3 실패: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("모든 비교 실험 완료!")
    print("="*60)
    print(f"결과 저장 위치: {args.output_dir}/")
    print("\n결과 확인 방법:")
    print("  1. JSON 파일: {}/experiment_X/cycle_XXX.json".format(args.output_dir))
    print("  2. 로그 파일: {}/experiment_X/logs/".format(args.output_dir))
    print("  3. 시각화: python experiments/visualize_results.py")
    print("="*60)


if __name__ == "__main__":
    main()

