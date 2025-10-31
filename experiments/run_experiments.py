"""
RAG 실험 실행 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import json
from datetime import datetime
from tqdm import tqdm

from src.parsing.labeled_parser import LabeledTableParser
from src.parsing.naive_parser import NaiveTableParser
from src.rag.kg_rag import KGRAGSystem
from src.rag.naive_rag import NaiveRAGSystem
from src.evaluation.metrics import RAGEvaluator


class ExperimentRunner:
    """실험 실행 클래스"""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.evaluator = RAGEvaluator()
    
    def load_test_data(self, data_path: str) -> List[pd.DataFrame]:
        """
        테스트 데이터 로드
        
        Args:
            data_path: 데이터 파일 경로 또는 디렉토리
        
        Returns:
            테이블 리스트
        """
        tables = []
        
        if os.path.isdir(data_path):
            # 디렉토리인 경우 모든 Excel/CSV 파일 로드
            for file in os.listdir(data_path):
                if file.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(os.path.join(data_path, file))
                    tables.append(df)
                elif file.endswith('.csv'):
                    df = pd.read_csv(os.path.join(data_path, file))
                    tables.append(df)
        else:
            # 단일 파일인 경우
            if data_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(data_path)
                tables.append(df)
            elif data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
                tables.append(df)
        
        return tables
    
    def experiment_1_parsing_comparison(self, tables: List[pd.DataFrame]) -> Dict[str, Any]:
        """
        실험 1: 레이블링 기반 파싱 vs Naive 파싱 성능 비교
        """
        print("=== 실험 1: 파싱 성능 비교 ===")
        
        labeled_parser = LabeledTableParser()
        naive_parser = NaiveTableParser()
        
        results = {
            'labeled_parsing': [],
            'naive_parsing': [],
            'comparison': []
        }
        
        for i, table in enumerate(tqdm(tables, desc="파싱 중")):
            table_id = f"table_{i}"
            
            # 레이블링 기반 파싱
            labeled_cells = labeled_parser.parse(table)
            labeled_structure = labeled_parser.to_structured_format(labeled_cells)
            
            # Naive 파싱
            naive_structure = naive_parser.parse(table)
            
            # 파싱 품질 평가 (간단한 메트릭)
            labeled_stats = {
                'total_cells': len(labeled_cells),
                'header_cells': len(labeled_structure['column_headers']),
                'data_cells': len(labeled_structure['data_cells']),
                'semantic_labels': sum(1 for cell in labeled_cells if cell.semantic_label)
            }
            
            naive_stats = {
                'total_cells': len(naive_structure['data']),
                'columns': len(naive_structure['columns'])
            }
            
            results['labeled_parsing'].append({
                'table_id': table_id,
                'stats': labeled_stats,
                'structure': labeled_structure
            })
            
            results['naive_parsing'].append({
                'table_id': table_id,
                'stats': naive_stats,
                'structure': naive_structure
            })
            
            comparison = {
                'table_id': table_id,
                'structure_richness': labeled_stats['semantic_labels'] / max(labeled_stats['total_cells'], 1),
                'header_detection': labeled_stats['header_cells'] > 0
            }
            results['comparison'].append(comparison)
        
        return results
    
    def experiment_2_rag_comparison(self, 
                                    tables: List[pd.DataFrame],
                                    queries: List[str],
                                    ground_truth: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        실험 2: KG 기반 RAG vs Naive 파싱 RAG 비교
        
        Args:
            tables: 테이블 리스트
            queries: 테스트 쿼리 리스트
            ground_truth: 각 쿼리에 대한 정답 테이블 ID 리스트
        """
        print("=== 실험 2: RAG 성능 비교 ===")
        
        # KG 기반 RAG 시스템 구축
        print("KG 기반 RAG 시스템 구축 중...")
        kg_rag = KGRAGSystem(use_labeled_parsing=True)
        for i, table in enumerate(tqdm(tables, desc="KG RAG 구축")):
            kg_rag.add_table(table, f"table_{i}")
        kg_rag.build_index()
        
        # Naive RAG 시스템 구축
        print("Naive RAG 시스템 구축 중...")
        naive_rag = NaiveRAGSystem()
        for i, table in enumerate(tqdm(tables, desc="Naive RAG 구축")):
            naive_rag.add_table(table, f"table_{i}")
        naive_rag.build_index()
        
        # 쿼리별 성능 평가
        results = {
            'kg_rag': [],
            'naive_rag': [],
            'comparison': []
        }
        
        for query in tqdm(queries, desc="쿼리 평가"):
            gt_ids = ground_truth.get(query, [])
            
            # KG RAG 검색
            kg_results = kg_rag.retrieve(query, top_k=5)
            kg_metrics = self.evaluator.evaluate_retrieval(kg_results, gt_ids)
            
            # Naive RAG 검색
            naive_results = naive_rag.retrieve(query, top_k=5)
            naive_metrics = self.evaluator.evaluate_retrieval(naive_results, gt_ids)
            
            results['kg_rag'].append({
                'query': query,
                'metrics': kg_metrics,
                'results': kg_results
            })
            
            results['naive_rag'].append({
                'query': query,
                'metrics': naive_metrics,
                'results': naive_results
            })
            
            # 비교
            comparison = self.evaluator.compare_systems(kg_metrics, naive_metrics)
            comparison['query'] = query
            results['comparison'].append(comparison)
        
        # 전체 평균 메트릭 계산
        kg_avg = self._calculate_average_metrics([r['metrics'] for r in results['kg_rag']])
        naive_avg = self._calculate_average_metrics([r['metrics'] for r in results['naive_rag']])
        
        results['summary'] = {
            'kg_rag_avg': kg_avg,
            'naive_rag_avg': naive_avg,
            'overall_improvement': self.evaluator.compare_systems(kg_avg, naive_avg)
        }
        
        return results
    
    def _calculate_average_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """메트릭 리스트의 평균 계산"""
        if not metrics_list:
            return {}
        
        avg_metrics = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m]
            avg_metrics[key] = np.mean(values) if values else 0.0
        
        return avg_metrics
    
    def save_results(self, results: Dict[str, Any], experiment_name: str):
        """결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # JSON 직렬화 가능하도록 변환
        serializable_results = self._make_serializable(results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"결과 저장: {filepath}")
        return filepath
    
    def _make_serializable(self, obj):
        """객체를 JSON 직렬화 가능하도록 변환"""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        else:
            return obj
    
    def run_all_experiments(self, 
                           data_path: str,
                           queries: List[str] = None,
                           ground_truth: Dict[str, List[str]] = None):
        """
        모든 실험 실행
        
        Args:
            data_path: 데이터 경로
            queries: 테스트 쿼리 리스트
            ground_truth: 정답 딕셔너리
        """
        print("데이터 로드 중...")
        tables = self.load_test_data(data_path)
        print(f"로드된 테이블 수: {len(tables)}")
        
        # 실험 1: 파싱 비교
        parsing_results = self.experiment_1_parsing_comparison(tables)
        self.save_results(parsing_results, "experiment_1_parsing")
        
        # 실험 2: RAG 비교 (쿼리가 있는 경우)
        if queries and ground_truth:
            rag_results = self.experiment_2_rag_comparison(tables, queries, ground_truth)
            self.save_results(rag_results, "experiment_2_rag")
        
        print("\n=== 모든 실험 완료 ===")


def main():
    """메인 실행 함수"""
    runner = ExperimentRunner(output_dir="results")
    
    # 예제 데이터 경로 (실제 데이터로 교체 필요)
    data_path = "data/sample_tables"
    
    # 예제 쿼리 및 정답 (실제 데이터에 맞게 수정 필요)
    queries = [
        "2023년 매출액은 얼마인가요?",
        "직원 수가 가장 많은 부서는?",
        "작년 대비 성장률이 높은 항목은?",
    ]
    
    ground_truth = {
        "2023년 매출액은 얼마인가요?": ["table_0"],
        "직원 수가 가장 많은 부서는?": ["table_1"],
        "작년 대비 성장률이 높은 항목은?": ["table_0", "table_1"],
    }
    
    # 실험 실행
    runner.run_all_experiments(
        data_path=data_path,
        queries=queries,
        ground_truth=ground_truth
    )


if __name__ == "__main__":
    main()

