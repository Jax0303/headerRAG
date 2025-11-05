"""
RAG 실험 실행 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import json
import time
from datetime import datetime
from tqdm import tqdm

from src.parsing.labeled_parser import LabeledTableParser
from src.parsing.naive_parser import NaiveTableParser
from src.evaluation.logger import EvaluationLogger

# 시각화 모듈 (조건부 import)
try:
    from experiments.visualize_results import ResultVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("경고: 시각화 모듈을 불러올 수 없습니다. 시각화는 생성되지 않습니다.")

# PDF 테이블 추출 모듈 (조건부 import)
try:
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from utils.pdf_table_extractor import PDFTableExtractor
    PDF_EXTRACTOR_AVAILABLE = True
except ImportError as e:
    PDF_EXTRACTOR_AVAILABLE = False
    print(f"경고: PDF 추출 모듈을 불러올 수 없습니다: {e}")

# 실험 2에서만 필요한 모듈 (조건부 import)
try:
    from src.rag.kg_rag import KGRAGSystem
    from src.rag.naive_rag import NaiveRAGSystem
    from src.evaluation.metrics import RAGEvaluator
    from src.evaluation.dataset_metrics import DatasetEvaluator
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("경고: RAG 모듈을 불러올 수 없습니다. 실험 2는 실행되지 않습니다.")

# 베이스라인 모델 (조건부 import)
try:
    from src.baselines import TATRParser, SatoSemanticTypeDetector, TableRAGBaseline, Tab2KGBaseline
    BASELINES_AVAILABLE = True
except ImportError as e:
    BASELINES_AVAILABLE = False
    print(f"경고: 베이스라인 모델을 불러올 수 없습니다: {e}")
    print("베이스라인 비교는 건너뜁니다.")


class ExperimentRunner:
    """실험 실행 클래스"""
    
    def __init__(self, output_dir: str = "results", cycle_runs: int = 10):
        """
        Args:
            output_dir: 결과 저장 디렉토리
            cycle_runs: 한 사이클당 실행 횟수 (기본값: 10)
        """
        self.output_dir = output_dir
        self.cycle_runs = cycle_runs
        os.makedirs(output_dir, exist_ok=True)
        
        # 실험별 디렉토리 생성
        self.experiment_dirs = {
            'experiment_1_parsing': os.path.join(output_dir, 'experiment_1'),
            'experiment_2_rag': os.path.join(output_dir, 'experiment_2'),
            'experiment_3_complexity': os.path.join(output_dir, 'experiment_3')
        }
        for exp_dir in self.experiment_dirs.values():
            os.makedirs(exp_dir, exist_ok=True)
        
        # 사이클 관리 변수
        self.current_cycle = {}  # 실험별 현재 사이클 번호
        self.cycle_results = {}  # 실험별 사이클 결과 누적
        self.cycle_run_count = {}  # 실험별 현재 사이클의 실행 횟수
        
        self.logger = EvaluationLogger(base_dir=output_dir)
        if RAG_AVAILABLE:
            self.evaluator = RAGEvaluator()
            self.dataset_evaluator = DatasetEvaluator()
        else:
            self.evaluator = None
            self.dataset_evaluator = None
    
    def load_test_data(self, 
                      data_path: str = "", 
                      use_dataset: bool = False,
                      datasets: Optional[List[str]] = None) -> List[pd.DataFrame]:
        """
        테스트 데이터 로드
        
        Args:
            data_path: 데이터 파일 경로 또는 디렉토리
            use_dataset: True이면 RAG-Evaluation-Dataset-KO의 전체 데이터셋 사용
            datasets: 다중 데이터셋 리스트 ('pubtables1m', 'tabrecset', 'korwiki_tabular', 'rag_eval_ko')
        
        Returns:
            테이블 리스트
        """
        # 다중 데이터셋 모드
        if datasets:
            from utils.multi_dataset_loader import MultiDatasetLoader
            loader = MultiDatasetLoader()
            tables, counts = loader.load_mixed_datasets(
                datasets=datasets,
                max_tables_per_dataset=None  # 전체 사용
            )
            return tables
        tables = []
        
        # 실제 평가 데이터셋 사용
        if use_dataset:
            print("실제 평가 데이터셋에서 테이블 추출 중...")
            
            # 이미 추출된 테이블이 있는지 확인
            extracted_dir = "data/extracted_tables"
            if os.path.exists(extracted_dir):
                print(f"추출된 테이블 디렉토리 발견: {extracted_dir}")
                # 모든 도메인 디렉토리에서 테이블 로드
                for domain_dir in os.listdir(extracted_dir):
                    domain_path = os.path.join(extracted_dir, domain_dir)
                    if os.path.isdir(domain_path):
                        for file in os.listdir(domain_path):
                            if file.endswith(('.xlsx', '.xls')):
                                try:
                                    df = pd.read_excel(os.path.join(domain_path, file))
                                    if not df.empty:
                                        tables.append(df)
                                except Exception as e:
                                    print(f"  경고: 파일 읽기 실패 ({file}): {e}")
            
            # 추출된 테이블이 없으면 PDF에서 추출 시도
            if len(tables) == 0 and PDF_EXTRACTOR_AVAILABLE:
                print("추출된 테이블이 없습니다. PDF에서 테이블 추출을 시도합니다...")
                try:
                    extractor = PDFTableExtractor(output_dir=extracted_dir)
                    tables_info = extractor.extract_all_from_dataset(
                        documents_csv="RAG-Evaluation-Dataset-KO/documents.csv",
                        pdf_base_dir="RAG-Evaluation-Dataset-KO",
                        save_to_excel=True
                    )
                    tables = [info['table_data'] for info in tables_info]
                    print(f"PDF에서 {len(tables)}개 테이블 추출 완료")
                except Exception as e:
                    print(f"  경고: PDF 테이블 추출 실패: {e}")
                    print("  에러: 테이블을 추출할 수 없습니다.")
                    return []
            
            if len(tables) == 0:
                print("경고: 추출된 테이블이 없습니다.")
                return []
            
            return tables
        
        # 기존 로직 (디렉토리 또는 파일에서 로드)
        if os.path.isdir(data_path):
            # 디렉토리인 경우 모든 Excel/CSV 파일 로드
            for file in os.listdir(data_path):
                if file.endswith(('.xlsx', '.xls')):
                    try:
                        df = pd.read_excel(os.path.join(data_path, file))
                        if not df.empty:
                            tables.append(df)
                    except Exception as e:
                        print(f"  경고: 파일 읽기 실패 ({file}): {e}")
                elif file.endswith('.csv'):
                    try:
                        df = pd.read_csv(os.path.join(data_path, file))
                        if not df.empty:
                            tables.append(df)
                    except Exception as e:
                        print(f"  경고: 파일 읽기 실패 ({file}): {e}")
        else:
            # 단일 파일인 경우
            if data_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(data_path)
                tables.append(df)
            elif data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
                tables.append(df)
        
        return tables
    
    def experiment_1_parsing_comparison(self, tables: List[pd.DataFrame], include_baselines: bool = True) -> Dict[str, Any]:
        """
        실험 1: 레이블링 기반 파싱 vs Naive 파싱 성능 비교
        
        Args:
            tables: 테스트할 테이블 리스트
            include_baselines: 베이스라인 모델 포함 여부
        """
        print("=== 실험 1: 파싱 성능 비교 ===")
        
        labeled_parser = LabeledTableParser()
        naive_parser = NaiveTableParser()
        
        # 베이스라인 모델 초기화 (선택적)
        tatr_parser = None
        sato_detector = None
        if include_baselines and BASELINES_AVAILABLE:
            try:
                tatr_parser = TATRParser(model_version="v1.1-pub")
                sato_detector = SatoSemanticTypeDetector()
                print("베이스라인 모델 초기화 완료: TATR, Sato")
            except Exception as e:
                print(f"경고: 베이스라인 모델 초기화 실패: {e}")
                include_baselines = False
        
        results = {
            'labeled_parsing': [],
            'naive_parsing': [],
            'comparison': []
        }
        
        # 베이스라인 결과 저장소
        if include_baselines:
            results['tatr_parsing'] = []
            results['sato_semantic'] = []
            results['baseline_comparison'] = []
        
        for i, table in enumerate(tqdm(tables, desc="파싱 중")):
            table_id = f"table_{i}"
            
            # 레이블링 기반 파싱 (시간 측정)
            start_time = time.time()
            labeled_cells = labeled_parser.parse(table)
            labeled_structure = labeled_parser.to_structured_format(labeled_cells)
            labeled_parsing_time = (time.time() - start_time) * 1000  # 밀리초
            
            # Naive 파싱 (시간 측정)
            start_time = time.time()
            naive_structure = naive_parser.parse(table)
            naive_parsing_time = (time.time() - start_time) * 1000  # 밀리초
            
            # 파싱 품질 평가 (간단한 메트릭)
            labeled_stats = {
                'total_cells': len(labeled_cells),
                'header_cells': len(labeled_structure['column_headers']),
                'data_cells': len(labeled_structure['data_cells']),
                'semantic_labels': sum(1 for cell in labeled_cells if cell.semantic_label),
                'parsing_time_ms': labeled_parsing_time,
                'parsing_speed_tables_per_sec': 1000.0 / labeled_parsing_time if labeled_parsing_time > 0 else 0.0
            }
            
            naive_stats = {
                'total_cells': len(naive_structure['data']),
                'columns': len(naive_structure['columns']),
                'parsing_time_ms': naive_parsing_time,
                'parsing_speed_tables_per_sec': 1000.0 / naive_parsing_time if naive_parsing_time > 0 else 0.0
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
                'header_detection': labeled_stats['header_cells'] > 0,
                'speed_ratio': naive_parsing_time / labeled_parsing_time if labeled_parsing_time > 0 else 0.0,
                'speed_improvement_percent': ((naive_parsing_time - labeled_parsing_time) / naive_parsing_time * 100) if naive_parsing_time > 0 else 0.0
            }
            results['comparison'].append(comparison)
            
            # 베이스라인 모델 평가 (선택적)
            if include_baselines and tatr_parser is not None:
                try:
                    # TATR 파싱
                    start_time = time.time()
                    tatr_result = tatr_parser.parse(table_data=table)
                    tatr_parsing_time = (time.time() - start_time) * 1000
                    
                    tatr_stats = {
                        'parsing_time_ms': tatr_parsing_time,
                        'cells_detected': len(tatr_result.get('cells', [])),
                        'structure_detected': tatr_result.get('structure', {})
                    }
                    results['tatr_parsing'].append({
                        'table_id': table_id,
                        'stats': tatr_stats,
                        'result': tatr_result
                    })
                except Exception as e:
                    print(f"  경고: TATR 파싱 실패 ({table_id}): {e}")
                
                # Sato 시맨틱 타입 검출
                try:
                    start_time = time.time()
                    column_types = sato_detector.detect_column_types(table)
                    sato_detection_time = (time.time() - start_time) * 1000
                    
                    sato_stats = {
                        'detection_time_ms': sato_detection_time,
                        'columns_detected': len(column_types),
                        'types': column_types
                    }
                    results['sato_semantic'].append({
                        'table_id': table_id,
                        'stats': sato_stats
                    })
                except Exception as e:
                    print(f"  경고: Sato 검출 실패 ({table_id}): {e}")
            
            # TXT 파일로 개별 평가 결과 저장
            table_result = {
                'labeled_parsing': {
                    'stats': labeled_stats,
                    'structure': labeled_structure
                },
                'naive_parsing': {
                    'stats': naive_stats,
                    'structure': naive_structure
                },
                'comparison': comparison
            }
            # 평가 결과를 파일에 추가 (첫 번째 테이블일 때만 파일 경로 출력)
            log_path = self.logger.log_parsing_evaluation(
                experiment_name="experiment_1_parsing",
                table_id=table_id,
                results=table_result
            )
            if i == 0:
                print(f"  파싱 평가 결과 저장: {log_path}")
        
        # 전체 요약 계산 및 저장
        total_tables = len(tables)
        avg_structure_richness = np.mean([c['structure_richness'] for c in results['comparison']])
        header_detection_count = sum(1 for c in results['comparison'] if c['header_detection'])
        
        # 속도 통계 계산
        labeled_times = [r['stats']['parsing_time_ms'] for r in results['labeled_parsing']]
        naive_times = [r['stats']['parsing_time_ms'] for r in results['naive_parsing']]
        
        summary = {
            'total_tables': total_tables,
            'avg_structure_richness': avg_structure_richness,
            'header_detection_rate': header_detection_count / total_tables if total_tables > 0 else 0.0,
            'header_detection_count': header_detection_count,
            'labeled_parsing_stats': {
                'avg_total_cells': np.mean([r['stats']['total_cells'] for r in results['labeled_parsing']]),
                'avg_header_cells': np.mean([r['stats']['header_cells'] for r in results['labeled_parsing']]),
                'avg_data_cells': np.mean([r['stats']['data_cells'] for r in results['labeled_parsing']]),
                'avg_semantic_labels': np.mean([r['stats']['semantic_labels'] for r in results['labeled_parsing']]),
                'avg_parsing_time_ms': np.mean(labeled_times),
                'min_parsing_time_ms': np.min(labeled_times),
                'max_parsing_time_ms': np.max(labeled_times),
                'total_parsing_time_ms': np.sum(labeled_times),
                'avg_parsing_speed_tables_per_sec': np.mean([r['stats']['parsing_speed_tables_per_sec'] for r in results['labeled_parsing']])
            },
            'naive_parsing_stats': {
                'avg_total_cells': np.mean([r['stats']['total_cells'] for r in results['naive_parsing']]),
                'avg_columns': np.mean([r['stats']['columns'] for r in results['naive_parsing']]),
                'avg_parsing_time_ms': np.mean(naive_times),
                'min_parsing_time_ms': np.min(naive_times),
                'max_parsing_time_ms': np.max(naive_times),
                'total_parsing_time_ms': np.sum(naive_times),
                'avg_parsing_speed_tables_per_sec': np.mean([r['stats']['parsing_speed_tables_per_sec'] for r in results['naive_parsing']])
            },
            'speed_comparison': {
                'avg_speed_ratio': np.mean([c['speed_ratio'] for c in results['comparison']]),
                'avg_speed_improvement_percent': np.mean([c['speed_improvement_percent'] for c in results['comparison']]),
                'total_time_saved_ms': np.sum(naive_times) - np.sum(labeled_times)
            }
        }
        
        # 요약 결과를 results에 추가
        results['summary'] = summary
        
        # 요약 결과 TXT 저장
        summary_path = self.logger.log_summary(
            experiment_name="experiment_1_parsing",
            evaluation_type="parsing",
            summary=summary
        )
        print(f"  파싱 실험 요약 저장: {summary_path}")
        
        # 시각화 생성
        if VISUALIZATION_AVAILABLE:
            try:
                print("  시각화 생성 중...")
                visualizer = ResultVisualizer(results_dir=self.output_dir)
                visualizer.visualize_parsing_results()
                
                # 베이스라인 비교 시각화 (있는 경우)
                if include_baselines and ('tatr_parsing' in results or 'sato_semantic' in results):
                    visualizer.visualize_baseline_comparison(results, "baseline_parsing")
                
                print("  시각화 완료")
            except Exception as e:
                print(f"  시각화 생성 중 오류 발생: {e}")
        
        return results
    
    def experiment_2_rag_comparison(self, 
                                    tables: List[pd.DataFrame],
                                    queries: List[str],
                                    ground_truth: Dict[str, List[str]],
                                    include_baselines: bool = True) -> Dict[str, Any]:
        """
        실험 2: KG 기반 RAG vs Naive 파싱 RAG 비교
        
        Args:
            tables: 테이블 리스트
            queries: 테스트 쿼리 리스트
            ground_truth: 각 쿼리에 대한 정답 테이블 ID 리스트
            include_baselines: 베이스라인 모델 포함 여부
        """
        print("=== 실험 2: RAG 성능 비교 ===")
        
        # KG 기반 RAG 시스템 구축 (시간 측정)
        print("KG 기반 RAG 시스템 구축 중...")
        kg_build_start = time.time()
        kg_rag = KGRAGSystem(use_labeled_parsing=True)
        for i, table in enumerate(tqdm(tables, desc="KG RAG 구축")):
            kg_rag.add_table(table, f"table_{i}")
        kg_rag.build_index()
        kg_build_time = (time.time() - kg_build_start) * 1000  # 밀리초
        
        # Naive RAG 시스템 구축 (시간 측정)
        print("Naive RAG 시스템 구축 중...")
        naive_build_start = time.time()
        naive_rag = NaiveRAGSystem()
        for i, table in enumerate(tqdm(tables, desc="Naive RAG 구축")):
            naive_rag.add_table(table, f"table_{i}")
        naive_rag.build_index()
        naive_build_time = (time.time() - naive_build_start) * 1000  # 밀리초
        
        # 베이스라인 RAG 시스템 구축 (선택적)
        tablerag_baseline = None
        tab2kg_baseline = None
        if include_baselines and BASELINES_AVAILABLE:
            try:
                print("TableRAG 베이스라인 구축 중...")
                tablerag_build_start = time.time()
                tablerag_baseline = TableRAGBaseline(use_colbert=False)
                for i, table in enumerate(tqdm(tables, desc="TableRAG 구축")):
                    tablerag_baseline.add_table(table, f"table_{i}", chunk_strategy="row")
                tablerag_baseline.build_index()
                tablerag_build_time = (time.time() - tablerag_build_start) * 1000
                print(f"TableRAG 구축 완료 ({tablerag_build_time:.2f}ms)")
            except Exception as e:
                print(f"경고: TableRAG 베이스라인 구축 실패: {e}")
                tablerag_baseline = None
        
        # 쿼리별 성능 평가
        results = {
            'kg_rag': [],
            'naive_rag': [],
            'comparison': []
        }
        
        # 베이스라인 결과 저장소
        if include_baselines:
            results['tablerag_baseline'] = []
            results['baseline_comparison'] = []
        
        for idx, query in enumerate(tqdm(queries, desc="쿼리 평가")):
            query_id = f"query_{idx}"
            gt_ids = ground_truth.get(query, [])
            
            # Ground truth가 많으면 top_k를 더 크게 설정하여 Recall 개선
            top_k = max(5, len(gt_ids) * 2) if gt_ids else 5
            top_k = min(top_k, len(tables))  # 테이블 수를 초과하지 않도록
            
            # KG RAG 검색 (시간 측정)
            kg_retrieve_start = time.time()
            kg_results = kg_rag.retrieve(query, top_k=top_k)
            kg_retrieve_time = (time.time() - kg_retrieve_start) * 1000  # 밀리초
            kg_metrics = self.evaluator.evaluate_retrieval(kg_results, gt_ids)
            
            # Naive RAG 검색 (시간 측정)
            naive_retrieve_start = time.time()
            naive_results = naive_rag.retrieve(query, top_k=top_k)
            naive_retrieve_time = (time.time() - naive_retrieve_start) * 1000  # 밀리초
            naive_metrics = self.evaluator.evaluate_retrieval(naive_results, gt_ids)
            
            kg_result_dict = {
                'query': query,
                'metrics': kg_metrics,
                'results': kg_results,
                'retrieve_time_ms': kg_retrieve_time
            }
            results['kg_rag'].append(kg_result_dict)
            
            naive_result_dict = {
                'query': query,
                'metrics': naive_metrics,
                'results': naive_results,
                'retrieve_time_ms': naive_retrieve_time
            }
            results['naive_rag'].append(naive_result_dict)
            
            # 비교
            comparison = self.evaluator.compare_systems(kg_metrics, naive_metrics)
            comparison['query'] = query
            results['comparison'].append(comparison)
            
            # 베이스라인 RAG 평가 (선택적)
            if include_baselines and tablerag_baseline is not None:
                try:
                    tablerag_retrieve_start = time.time()
                    tablerag_top_k = max(5, len(gt_ids) * 2) if gt_ids else 5
                    tablerag_top_k = min(tablerag_top_k, len(tables))
                    tablerag_results = tablerag_baseline.query(query, top_k=tablerag_top_k, return_context=True)
                    tablerag_retrieve_time = (time.time() - tablerag_retrieve_start) * 1000
                    
                    # TableRAG 결과를 기존 형식으로 변환
                    tablerag_formatted = [
                        {
                            'table_id': r['table_id'],
                            'score': r['score'],
                            'text': r['text']
                        }
                        for r in tablerag_results['results']
                    ]
                    tablerag_metrics = self.evaluator.evaluate_retrieval(tablerag_formatted, gt_ids)
                    
                    results['tablerag_baseline'].append({
                        'query': query,
                        'metrics': tablerag_metrics,
                        'results': tablerag_formatted,
                        'retrieve_time_ms': tablerag_retrieve_time
                    })
                except Exception as e:
                    print(f"  경고: TableRAG 평가 실패 ({query_id}): {e}")
            
            # TXT 파일로 개별 평가 결과 저장
            query_result = {
                'query': query,
                'kg_rag': kg_result_dict,
                'naive_rag': naive_result_dict,
                'comparison': comparison
            }
            # 평가 결과를 파일에 추가 (첫 번째 쿼리일 때만 파일 경로 출력)
            log_path = self.logger.log_rag_evaluation(
                experiment_name="experiment_2_rag",
                query_id=query_id,
                results=query_result
            )
            if idx == 0:
                print(f"  RAG 평가 결과 저장: {log_path}")
        
        # 전체 평균 메트릭 계산
        kg_avg = self._calculate_average_metrics([r['metrics'] for r in results['kg_rag']])
        naive_avg = self._calculate_average_metrics([r['metrics'] for r in results['naive_rag']])
        
        # 시간 통계 계산
        kg_retrieve_times = [r['retrieve_time_ms'] for r in results['kg_rag']]
        naive_retrieve_times = [r['retrieve_time_ms'] for r in results['naive_rag']]
        
        results['summary'] = {
            'kg_rag_avg': kg_avg,
            'naive_rag_avg': naive_avg,
            'overall_improvement': self.evaluator.compare_systems(kg_avg, naive_avg),
            'build_time': {
                'kg_rag_build_time_ms': kg_build_time,
                'naive_rag_build_time_ms': naive_build_time,
                'build_time_ratio': kg_build_time / naive_build_time if naive_build_time > 0 else 0.0
            },
            'retrieve_time': {
                'kg_rag_avg_retrieve_time_ms': np.mean(kg_retrieve_times),
                'kg_rag_min_retrieve_time_ms': np.min(kg_retrieve_times),
                'kg_rag_max_retrieve_time_ms': np.max(kg_retrieve_times),
                'naive_rag_avg_retrieve_time_ms': np.mean(naive_retrieve_times),
                'naive_rag_min_retrieve_time_ms': np.min(naive_retrieve_times),
                'naive_rag_max_retrieve_time_ms': np.max(naive_retrieve_times),
                'retrieve_time_ratio': np.mean(kg_retrieve_times) / np.mean(naive_retrieve_times) if np.mean(naive_retrieve_times) > 0 else 0.0
            }
        }
        
        # 요약 결과 TXT 저장
        summary_path = self.logger.log_summary(
            experiment_name="experiment_2_rag",
            evaluation_type="rag",
            summary=results['summary']
        )
        print(f"  RAG 실험 요약 저장: {summary_path}")
        
        # 시각화 생성
        if VISUALIZATION_AVAILABLE:
            try:
                print("  시각화 생성 중...")
                visualizer = ResultVisualizer(results_dir=self.output_dir)
                visualizer.visualize_rag_results()
                
                # 베이스라인 비교 시각화 (있는 경우)
                if include_baselines and 'tablerag_baseline' in results and results['tablerag_baseline']:
                    visualizer.visualize_baseline_comparison(results, "baseline_rag")
                
                print("  시각화 완료")
            except Exception as e:
                print(f"  시각화 생성 중 오류 발생: {e}")
        
        return results
    
    def _classify_table_complexity(self, table: pd.DataFrame) -> str:
        """
        테이블 구조 복잡도 분류
        
        Returns:
            'simple': 단순 표 (2차원, 헤더 1개)
            'nested_header': 중첩 헤더 표 (다중 헤더 행/열)
            'merged_cell': 병합 셀 표 (rowspan, colspan)
            'irregular': 비정형 표 (불규칙한 구조)
        """
        n_rows, n_cols = table.shape
        
        # 1. 중첩 헤더 확인 (MultiIndex 컬럼)
        if isinstance(table.columns, pd.MultiIndex):
            return 'nested_header'
        
        # 2. 병합 셀 확인 (동일한 값이 인접한 셀에 연속으로 있는 경우)
        has_merged = False
        for i in range(n_rows):
            for j in range(n_cols):
                val = table.iloc[i, j]
                # 같은 값이 여러 셀에 걸쳐있는지 확인
                if pd.notna(val):
                    # 행 방향 병합 확인
                    if j < n_cols - 1 and table.iloc[i, j+1] == val:
                        has_merged = True
                        break
                    # 열 방향 병합 확인
                    if i < n_rows - 1 and table.iloc[i+1, j] == val:
                        has_merged = True
                        break
            if has_merged:
                break
        
        if has_merged:
            return 'merged_cell'
        
        # 3. 비정형 구조 확인 (불규칙한 행/열 크기, 많은 빈 셀)
        empty_ratio = table.isna().sum().sum() / (n_rows * n_cols)
        if empty_ratio > 0.3 or n_rows < 2 or n_cols < 2:
            return 'irregular'
        
        # 4. 단순 표
        return 'simple'
    
    def _calculate_average_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """메트릭 리스트의 평균 계산"""
        if not metrics_list:
            return {}
        
        avg_metrics = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m]
            avg_metrics[key] = np.mean(values) if values else 0.0
        
        return avg_metrics
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """
        객체를 JSON 직렬화 가능한 형태로 변환
        
        Args:
            obj: 변환할 객체
        
        Returns:
            JSON 직렬화 가능한 객체
        """
        import networkx as nx
        import pandas as pd
        
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (nx.Graph, nx.DiGraph)):
            # NetworkX 그래프는 간단한 정보로 변환
            return {
                '_type': 'NetworkX_Graph',
                'num_nodes': obj.number_of_nodes(),
                'num_edges': obj.number_of_edges(),
                'nodes': list(obj.nodes())[:10],  # 처음 10개만
                'edges': list(obj.edges())[:10]   # 처음 10개만
            }
        elif isinstance(obj, pd.DataFrame):
            # DataFrame은 딕셔너리로 변환
            return {
                '_type': 'DataFrame',
                'shape': list(obj.shape),
                'columns': list(obj.columns),
                'data': obj.head(5).to_dict('records')  # 처음 5행만
            }
        elif isinstance(obj, (pd.Series, pd.Index)):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            # 일반 객체는 딕셔너리로 변환
            return {
                '_type': obj.__class__.__name__,
                **{k: self._make_json_serializable(v) for k, v in obj.__dict__.items()}
            }
        else:
            return obj
    
    def save_results(self, results: Dict[str, Any], experiment_name: str):
        """
        결과 저장 (사이클별로 묶어서 저장)
        
        Args:
            results: 실험 결과 딕셔너리
            experiment_name: 실험 이름
        """
        # 실험별 사이클 초기화 (없는 경우)
        if experiment_name not in self.current_cycle:
            self.current_cycle[experiment_name] = 1
            self.cycle_results[experiment_name] = []
            self.cycle_run_count[experiment_name] = 0
        
        # 현재 결과를 사이클 결과에 추가
        self.cycle_results[experiment_name].append(results)
        self.cycle_run_count[experiment_name] += 1
        
        # 사이클이 완료되었는지 확인
        if self.cycle_run_count[experiment_name] >= self.cycle_runs:
            # 사이클 완료 - 결과를 하나의 파일로 저장
            cycle_num = self.current_cycle[experiment_name]
            
            # 실험 디렉토리 가져오기
            exp_dir = self.experiment_dirs.get(experiment_name, self.output_dir)
            
            # 사이클 결과 통합
            cycle_summary = {
                'cycle_number': cycle_num,
                'cycle_runs': self.cycle_run_count[experiment_name],
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'individual_results': self.cycle_results[experiment_name],
                'aggregated_summary': self._aggregate_cycle_results(self.cycle_results[experiment_name], experiment_name)
            }
            
            # 파일 저장
            filename = f"cycle_{cycle_num:03d}.json"
            filepath = os.path.join(exp_dir, filename)
            
            # JSON 직렬화 가능하도록 변환
            serializable_results = self._make_json_serializable(cycle_summary)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            
            print(f"사이클 {cycle_num} 결과 저장: {filepath} ({self.cycle_run_count[experiment_name]}회 실행)")
            
            # 다음 사이클을 위해 초기화
            self.current_cycle[experiment_name] += 1
            self.cycle_results[experiment_name] = []
            self.cycle_run_count[experiment_name] = 0
            
            return filepath
        else:
            # 사이클이 아직 완료되지 않음
            print(f"사이클 {self.current_cycle[experiment_name]} 진행 중: {self.cycle_run_count[experiment_name]}/{self.cycle_runs}회 완료")
            return None
    
    def _aggregate_cycle_results(self, results_list: List[Dict[str, Any]], experiment_name: str) -> Dict[str, Any]:
        """
        사이클 내 여러 실행 결과를 집계
        
        Args:
            results_list: 사이클 내 모든 실행 결과 리스트
            experiment_name: 실험 이름
        
        Returns:
            집계된 요약 딕셔너리
        """
        if not results_list:
            return {}
        
        aggregated = {
            'total_runs': len(results_list),
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        if 'experiment_1_parsing' in experiment_name:
            # 실험 1 집계
            if results_list and 'summary' in results_list[0]:
                summaries = [r.get('summary', {}) for r in results_list if 'summary' in r]
                if summaries:
                    aggregated['avg_total_tables'] = np.mean([s.get('total_tables', 0) for s in summaries])
                    aggregated['avg_structure_richness'] = np.mean([s.get('avg_structure_richness', 0) for s in summaries])
                    aggregated['avg_header_detection_rate'] = np.mean([s.get('header_detection_rate', 0) for s in summaries])
                    
                    # 레이블링 파싱 통계 집계
                    labeled_stats_list = [s.get('labeled_parsing_stats', {}) for s in summaries]
                    if labeled_stats_list:
                        aggregated['labeled_parsing'] = {
                            'avg_parsing_time_ms': np.mean([s.get('avg_parsing_time_ms', 0) for s in labeled_stats_list]),
                            'avg_parsing_speed': np.mean([s.get('avg_parsing_speed_tables_per_sec', 0) for s in labeled_stats_list])
                        }
                    
                    # Naive 파싱 통계 집계
                    naive_stats_list = [s.get('naive_parsing_stats', {}) for s in summaries]
                    if naive_stats_list:
                        aggregated['naive_parsing'] = {
                            'avg_parsing_time_ms': np.mean([s.get('avg_parsing_time_ms', 0) for s in naive_stats_list]),
                            'avg_parsing_speed': np.mean([s.get('avg_parsing_speed_tables_per_sec', 0) for s in naive_stats_list])
                        }
        
        elif 'experiment_2_rag' in experiment_name:
            # 실험 2 집계
            if results_list and 'summary' in results_list[0]:
                summaries = [r.get('summary', {}) for r in results_list if 'summary' in r]
                if summaries:
                    # KG RAG 평균 메트릭
                    kg_avg_list = [s.get('kg_rag_avg', {}) for s in summaries]
                    if kg_avg_list:
                        aggregated['kg_rag_avg'] = {
                            'precision': np.mean([m.get('precision', 0) for m in kg_avg_list]),
                            'recall': np.mean([m.get('recall', 0) for m in kg_avg_list]),
                            'f1': np.mean([m.get('f1', 0) for m in kg_avg_list]),
                            'mrr': np.mean([m.get('mrr', 0) for m in kg_avg_list])
                        }
                    
                    # Naive RAG 평균 메트릭
                    naive_avg_list = [s.get('naive_rag_avg', {}) for s in summaries]
                    if naive_avg_list:
                        aggregated['naive_rag_avg'] = {
                            'precision': np.mean([m.get('precision', 0) for m in naive_avg_list]),
                            'recall': np.mean([m.get('recall', 0) for m in naive_avg_list]),
                            'f1': np.mean([m.get('f1', 0) for m in naive_avg_list]),
                            'mrr': np.mean([m.get('mrr', 0) for m in naive_avg_list])
                        }
                    
                    # 구축 시간 집계
                    build_times = [s.get('build_time', {}) for s in summaries]
                    if build_times:
                        aggregated['build_time'] = {
                            'avg_kg_rag_ms': np.mean([b.get('kg_rag_build_time_ms', 0) for b in build_times]),
                            'avg_naive_rag_ms': np.mean([b.get('naive_rag_build_time_ms', 0) for b in build_times])
                        }
                    
                    # 검색 시간 집계
                    retrieve_times = [s.get('retrieve_time', {}) for s in summaries]
                    if retrieve_times:
                        aggregated['retrieve_time'] = {
                            'avg_kg_rag_ms': np.mean([r.get('kg_rag_avg_retrieve_time_ms', 0) for r in retrieve_times]),
                            'avg_naive_rag_ms': np.mean([r.get('naive_rag_avg_retrieve_time_ms', 0) for r in retrieve_times])
                        }
        
        elif 'experiment_3_complexity' in experiment_name:
            # 실험 3 집계
            if results_list and 'summary' in results_list[0]:
                summaries = [r.get('summary', {}) for r in results_list if 'summary' in r]
                if summaries:
                    # 복잡도 분포는 첫 번째 결과에서 가져옴 (일반적으로 동일함)
                    aggregated['complexity_distribution'] = summaries[0].get('complexity_distribution', {})
                    
                    # 복잡도별 비교 결과 집계
                    complexity_comparisons = [s.get('complexity_comparison', {}) for s in summaries]
                    if complexity_comparisons:
                        aggregated['complexity_comparison'] = {}
                        # 각 복잡도 타입별로 집계
                        for comp_type in ['simple', 'nested_header', 'merged_cell', 'irregular']:
                            comp_data_list = [cc.get(comp_type, {}) for cc in complexity_comparisons if comp_type in cc]
                            if comp_data_list:
                                aggregated['complexity_comparison'][comp_type] = {
                                    'total_groups': len(comp_data_list)
                                }
        
        return aggregated
    
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
    
    def experiment_3_complexity_analysis(self, 
                                        tables: List[pd.DataFrame],
                                        queries: List[str] = None,
                                        ground_truth: Dict[str, List[str]] = None) -> Dict[str, Any]:
        """
        실험 3: 표 구조 복잡도에 따른 성능 비교
        
        Args:
            tables: 테이블 리스트
            queries: 테스트 쿼리 리스트 (RAG 실험용)
            ground_truth: 정답 딕셔너리 (RAG 실험용)
        
        Returns:
            복잡도별 실험 결과
        """
        print("=== 실험 3: 표 구조 복잡도에 따른 성능 비교 ===")
        
        # 테이블을 복잡도별로 분류
        complexity_groups = {
            'simple': [],
            'nested_header': [],
            'merged_cell': [],
            'irregular': []
        }
        
        table_complexity_map = {}
        for i, table in enumerate(tables):
            complexity = self._classify_table_complexity(table)
            complexity_groups[complexity].append((i, table))
            table_complexity_map[f"table_{i}"] = complexity
        
        print(f"\n복잡도별 테이블 분포:")
        for complexity, group in complexity_groups.items():
            print(f"  {complexity}: {len(group)}개")
        
        results = {
            'complexity_distribution': {k: len(v) for k, v in complexity_groups.items()},
            'table_complexity_map': table_complexity_map,
            'complexity_results': {}
        }
        
        # 각 복잡도별로 실험 1과 실험 2 실행
        for complexity, group in complexity_groups.items():
            if len(group) == 0:
                continue
            
            print(f"\n--- {complexity} 복잡도 그룹 분석 ({len(group)}개 테이블) ---")
            
            # 그룹의 테이블만 추출
            group_tables = [table for _, table in group]
            group_table_ids = [f"table_{i}" for i, _ in group]
            
            # 실험 1: 파싱 성능 비교
            print(f"  파싱 성능 비교 중...")
            parsing_results = self.experiment_1_parsing_comparison(group_tables)
            
            # 실험 2: RAG 성능 비교 (쿼리가 있는 경우)
            rag_results = None
            if queries and ground_truth and RAG_AVAILABLE:
                print(f"  RAG 성능 비교 중...")
                # 해당 복잡도 그룹의 테이블에 관련된 쿼리만 필터링
                filtered_queries = []
                filtered_ground_truth = {}
                for query_id, query in enumerate(queries):
                    # ground truth에 해당 복잡도 그룹의 테이블이 포함되어 있는지 확인
                    gt_tables = ground_truth.get(f"query_{query_id}", [])
                    if any(tid in group_table_ids for tid in gt_tables):
                        filtered_queries.append(query)
                        filtered_ground_truth[f"query_{len(filtered_queries)-1}"] = [
                            tid for tid in gt_tables if tid in group_table_ids
                        ]
                
                if len(filtered_queries) > 0:
                    try:
                        rag_results = self.experiment_2_rag_comparison(
                            group_tables, filtered_queries, filtered_ground_truth
                        )
                    except Exception as e:
                        print(f"  경고: RAG 실험 실패: {e}")
                        rag_results = None
            
            # 결과 저장
            complexity_result = {
                'parsing': parsing_results,
                'rag': rag_results
            }
            
            # 평균 메트릭 계산
            if parsing_results:
                labeled_avg = self._calculate_average_metrics(
                    [r for r in parsing_results.get('labeled_parsing', []) if isinstance(r, dict)]
                )
                naive_avg = self._calculate_average_metrics(
                    [r for r in parsing_results.get('naive_parsing', []) if isinstance(r, dict)]
                )
                
                complexity_result['parsing_summary'] = {
                    'labeled_avg': labeled_avg,
                    'naive_avg': naive_avg,
                    'improvement': {}
                }
                
                # 개선율 계산
                for key in labeled_avg.keys():
                    if key in naive_avg and naive_avg[key] > 0:
                        improvement = ((labeled_avg[key] - naive_avg[key]) / naive_avg[key]) * 100
                        complexity_result['parsing_summary']['improvement'][key] = improvement
            
            if rag_results and rag_results.get('summary'):
                complexity_result['rag_summary'] = rag_results['summary']
            
            results['complexity_results'][complexity] = complexity_result
        
        # 전체 요약 생성
        results['summary'] = self._generate_complexity_summary(results)
        
        return results
    
    def _generate_complexity_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """복잡도별 결과 요약 생성"""
        summary = {
            'complexity_comparison': {}
        }
        
        for complexity, comp_results in results['complexity_results'].items():
            comp_summary = {}
            
            # 파싱 요약
            if 'parsing_summary' in comp_results:
                comp_summary['parsing'] = comp_results['parsing_summary']
            
            # RAG 요약
            if 'rag_summary' in comp_results:
                comp_summary['rag'] = comp_results['rag_summary']
            
            summary['complexity_comparison'][complexity] = comp_summary
        
        return summary


def main():
    """메인 실행 함수"""
    # 사이클당 실행 횟수 설정 (기본값: 10)
    # 예: cycle_runs=10이면 10번 실행 후 결과를 하나의 파일로 저장
    runner = ExperimentRunner(output_dir="results", cycle_runs=10)
    
    # 실제 평가 데이터셋 사용 (전체)
    print("실험 1 실행: 레이블링 기반 파싱 vs Naive 파싱 성능 비교")
    print("실제 평가 데이터셋 전체 사용 중...")
    print("데이터 로드 중...")
    tables = runner.load_test_data("", use_dataset=True)
    print(f"로드된 테이블 수: {len(tables)}")
    
    # 실험 1: 파싱 비교
    parsing_results = runner.experiment_1_parsing_comparison(tables)
    runner.save_results(parsing_results, "experiment_1_parsing")
    
    print("\n=== 실험 1 완료 ===")
    
    # 실험 2: RAG 비교 (실제 평가 데이터셋의 쿼리 사용)
    if RAG_AVAILABLE and len(tables) > 0:
        print("\n=== 실험 2 시작: RAG 성능 비교 ===")
        
        # 쿼리와 ground truth 준비
        try:
            import sys
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from utils.prepare_rag_queries import prepare_queries_from_dataset, create_simple_queries
            
            print("쿼리 및 ground truth 준비 중...")
            # 실제 데이터셋에서 쿼리 추출 시도
            try:
                queries, ground_truth = prepare_queries_from_dataset()
                if len(queries) == 0:
                    raise ValueError("쿼리 추출 실패")
            except Exception as e:
                print(f"경고: 데이터셋에서 쿼리 추출 실패 ({e}). 간단한 쿼리 생성...")
                queries, ground_truth = create_simple_queries(tables, num_queries=min(20, len(tables)))
            
            print(f"사용할 쿼리 수: {len(queries)}")
            print(f"샘플 쿼리: {queries[0] if queries else 'None'}")
            
            # 실험 2 실행
            rag_results = runner.experiment_2_rag_comparison(tables, queries, ground_truth)
            runner.save_results(rag_results, "experiment_2_rag")
            print("\n=== 실험 2 완료 ===")
        except Exception as e:
            print(f"경고: 실험 2 실행 실패: {e}")
            print("RAG 모듈이 제대로 설치되지 않았거나 설정이 필요할 수 있습니다.")
    else:
        print("\n실험 2는 RAG 모듈이 필요합니다. 현재 사용할 수 없습니다.")


if __name__ == "__main__":
    main()

