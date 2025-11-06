#!/usr/bin/env python
"""
임베딩 모델 비교 실험
다양한 임베딩 모델의 성능 비교
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_experiments import ExperimentRunner
from utils.prepare_rag_queries import prepare_queries_from_dataset
import json
from pathlib import Path


def test_embedding_models():
    """다양한 임베딩 모델로 실험"""
    
    # 테스트할 임베딩 모델 리스트
    embedding_models = [
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 기본 (한국어 지원)
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",  # 더 큰 모델
        "jhgan/ko-sroberta-multitask",  # 한국어 특화
    ]
    
    print("="*70)
    print("임베딩 모델 비교 실험")
    print("="*70)
    
    # 테이블 데이터 로드
    runner = ExperimentRunner(output_dir="results", cycle_runs=1)
    tables = runner.load_test_data("", use_dataset=True)
    print(f"\n로드된 테이블 수: {len(tables)}")
    
    # 쿼리 및 ground truth 준비
    queries, ground_truth = prepare_queries_from_dataset()
    valid_queries = [q for q in queries if ground_truth.get(q, [])][:20]  # 빠른 테스트를 위해 20개만
    valid_ground_truth = {q: ground_truth[q] for q in valid_queries}
    
    print(f"테스트 쿼리 수: {len(valid_queries)}")
    
    results_summary = {}
    
    for model_name in embedding_models:
        print(f"\n{'='*70}")
        print(f"모델 테스트: {model_name}")
        print(f"{'='*70}")
        
        try:
            # 임베딩 모델 변경을 위해 RAG 시스템을 직접 생성
            from src.rag.kg_rag import KGRAGSystem
            from src.rag.naive_rag import NaiveRAGSystem
            
            # KG RAG 구축
            print("KG RAG 구축 중...")
            kg_rag = KGRAGSystem(
                embedding_model=model_name,
                use_labeled_parsing=True
            )
            for i, table in enumerate(tables):
                kg_rag.add_table(table, f"table_{i}")
            kg_rag.build_index()
            
            # Naive RAG 구축
            print("Naive RAG 구축 중...")
            naive_rag = NaiveRAGSystem(embedding_model=model_name)
            for i, table in enumerate(tables):
                naive_rag.add_table(table, f"table_{i}")
            naive_rag.build_index()
            
            # 평가
            kg_metrics_list = []
            naive_metrics_list = []
            
            for query in valid_queries:
                gt_ids = valid_ground_truth.get(query, [])
                top_k = max(5, len(gt_ids) * 2) if gt_ids else 5
                top_k = min(top_k, len(tables))
                
                # KG RAG 평가
                kg_results = kg_rag.retrieve(query, top_k=top_k)
                kg_metrics = runner.evaluator.evaluate_retrieval(kg_results, gt_ids)
                kg_metrics_list.append(kg_metrics)
                
                # Naive RAG 평가
                naive_results = naive_rag.retrieve(query, top_k=top_k)
                naive_metrics = runner.evaluator.evaluate_retrieval(naive_results, gt_ids)
                naive_metrics_list.append(naive_metrics)
            
            # 평균 계산
            import numpy as np
            kg_avg = {
                'precision': np.mean([m['precision'] for m in kg_metrics_list]),
                'recall': np.mean([m['recall'] for m in kg_metrics_list]),
                'f1': np.mean([m['f1'] for m in kg_metrics_list]),
                'mrr': np.mean([m['mrr'] for m in kg_metrics_list])
            }
            
            naive_avg = {
                'precision': np.mean([m['precision'] for m in naive_metrics_list]),
                'recall': np.mean([m['recall'] for m in naive_metrics_list]),
                'f1': np.mean([m['f1'] for m in naive_metrics_list]),
                'mrr': np.mean([m['mrr'] for m in naive_metrics_list])
            }
            
            results_summary[model_name] = {
                'kg_rag': kg_avg,
                'naive_rag': naive_avg
            }
            
            print(f"\n✓ {model_name} 테스트 완료")
            print(f"  KG RAG - Precision: {kg_avg['precision']*100:.2f}%, Recall: {kg_avg['recall']*100:.2f}%")
            print(f"  Naive RAG - Precision: {naive_avg['precision']*100:.2f}%, Recall: {naive_avg['recall']*100:.2f}%")
            
        except Exception as e:
            print(f"✗ {model_name} 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
    
    # 결과 요약 출력
    print("\n" + "="*70)
    print("임베딩 모델 비교 결과 요약")
    print("="*70)
    
    for model_name, results in results_summary.items():
        print(f"\n{model_name}:")
        print(f"  KG RAG:")
        print(f"    Precision: {results['kg_rag']['precision']*100:.2f}%")
        print(f"    Recall: {results['kg_rag']['recall']*100:.2f}%")
        print(f"    F1: {results['kg_rag']['f1']:.4f}")
        print(f"  Naive RAG:")
        print(f"    Precision: {results['naive_rag']['precision']*100:.2f}%")
        print(f"    Recall: {results['naive_rag']['recall']*100:.2f}%")
        print(f"    F1: {results['naive_rag']['f1']:.4f}")
    
    # 결과 저장
    output_path = Path("results/embedding_model_comparison.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {output_path}")
    
    return results_summary


if __name__ == "__main__":
    test_embedding_models()



