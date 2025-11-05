#!/usr/bin/env python
"""
전체 실험 실행 스크립트
실제 데이터셋으로 모든 실험 실행 (베이스라인 포함)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.run_experiments import ExperimentRunner
from utils.prepare_rag_queries import prepare_queries_from_dataset
from experiments.generate_comprehensive_report import ComprehensiveReportGenerator
import json
from pathlib import Path


def main():
    print("="*70)
    print("HeaderRAG 전체 실험 실행")
    print("="*70)
    
    # 실험 러너 초기화
    runner = ExperimentRunner(output_dir="results", cycle_runs=1)
    
    # 1. 실제 데이터셋 로드
    print("\n[1/5] 실제 데이터셋 로드 중...")
    tables = runner.load_test_data("", use_dataset=True)
    print(f"✓ 로드된 테이블 수: {len(tables)}")
    
    if len(tables) == 0:
        print("에러: 테이블이 없습니다. 실험을 종료합니다.")
        sys.exit(1)
    
    # 2. 실험 1: 파싱 성능 비교 (베이스라인 포함)
    print("\n" + "="*70)
    print("[2/5] 실험 1: 파싱 성능 비교 (베이스라인 포함)")
    print("="*70)
    
    # 처음 20개 테이블만 사용 (전체는 시간이 오래 걸림)
    test_tables = tables[:20] if len(tables) > 20 else tables
    print(f"테스트 테이블 수: {len(test_tables)}")
    
    try:
        results_1 = runner.experiment_1_parsing_comparison(
            tables=test_tables,
            include_baselines=True
        )
        runner.save_results(results_1, "experiment_1_parsing")
        print("✓ 실험 1 완료")
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
        print(f"✓ 준비된 쿼리 수: {len(queries)}")
        
        # 처음 10개 쿼리만 사용 (테스트용)
        if len(queries) > 10:
            queries = queries[:10]
            ground_truth = {q: ground_truth.get(q, []) for q in queries}
            print(f"테스트 쿼리 수: {len(queries)}")
    except Exception as e:
        print(f"경고: 데이터셋에서 쿼리 준비 실패: {e}")
        print("샘플 쿼리 생성 중...")
        from utils.prepare_rag_queries import create_simple_queries
        queries, ground_truth = create_simple_queries(tables, num_queries=5)
        print(f"✓ 샘플 쿼리 생성: {len(queries)}개")
    
    # 4. 실험 2: RAG 성능 비교 (베이스라인 포함)
    print("\n" + "="*70)
    print("[4/5] 실험 2: RAG 성능 비교 (베이스라인 포함)")
    print("="*70)
    
    try:
        results_2 = runner.experiment_2_rag_comparison(
            tables=test_tables,
            queries=queries,
            ground_truth=ground_truth,
            include_baselines=True
        )
        runner.save_results(results_2, "experiment_2_rag")
        print("✓ 실험 2 완료")
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
        generator = ComprehensiveReportGenerator()
        report_path = generator.generate_report(
            experiment_1_results=results_1,
            experiment_2_results=results_2,
            experiment_3_results=None,
            output_format="html"
        )
        print(f"✓ HTML 리포트 생성: {report_path}")
        
        md_path = generator.generate_report(
            experiment_1_results=results_1,
            experiment_2_results=results_2,
            experiment_3_results=None,
            output_format="md"
        )
        print(f"✓ Markdown 리포트 생성: {md_path}")
    except Exception as e:
        print(f"경고: 리포트 생성 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # 결과 요약 출력
    print("\n" + "="*70)
    print("실험 결과 요약")
    print("="*70)
    
    if results_1:
        summary_1 = results_1.get('summary', {})
        print(f"\n[실험 1] 파싱 성능:")
        print(f"  - 총 테이블 수: {summary_1.get('total_tables', 0)}")
        print(f"  - 평균 구조 풍부도: {summary_1.get('avg_structure_richness', 0):.3f}")
        print(f"  - 헤더 감지율: {summary_1.get('header_detection_rate', 0):.2%}")
        
        if 'tatr_parsing' in results_1:
            print(f"  - TATR 파싱 결과: {len(results_1['tatr_parsing'])}개 테이블")
        if 'sato_semantic' in results_1:
            print(f"  - Sato 검출 결과: {len(results_1['sato_semantic'])}개 테이블")
    
    if results_2:
        summary_2 = results_2.get('summary', {})
        print(f"\n[실험 2] RAG 성능:")
        kg_avg = summary_2.get('kg_rag_avg', {})
        naive_avg = summary_2.get('naive_rag_avg', {})
        print(f"  - KG RAG: Precision={kg_avg.get('precision', 0):.3f}, Recall={kg_avg.get('recall', 0):.3f}, F1={kg_avg.get('f1', 0):.3f}")
        print(f"  - Naive RAG: Precision={naive_avg.get('precision', 0):.3f}, Recall={naive_avg.get('recall', 0):.3f}, F1={naive_avg.get('f1', 0):.3f}")
        
        if 'tablerag_baseline' in results_2 and results_2['tablerag_baseline']:
            import numpy as np
            tablerag_metrics = [r['metrics'] for r in results_2['tablerag_baseline']]
            print(f"  - TableRAG: Precision={np.mean([m.get('precision', 0) for m in tablerag_metrics]):.3f}, Recall={np.mean([m.get('recall', 0) for m in tablerag_metrics]):.3f}, F1={np.mean([m.get('f1', 0) for m in tablerag_metrics]):.3f}")
    
    print("\n" + "="*70)
    print("모든 실험 완료!")
    print("="*70)
    print(f"\n결과 위치:")
    print(f"  - 실험 결과: results/")
    print(f"  - 시각화: results/visualizations/")
    print(f"  - 리포트: reports/")


if __name__ == "__main__":
    main()

