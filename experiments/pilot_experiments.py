"""
파일럿 실험 프레임워크
- 소규모 데이터셋 실험
- 복잡도별 균등 샘플링
- Ablation Study
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm

from src.parsing.labeled_parser import LabeledTableParser
from src.parsing.naive_parser import NaiveTableParser
from src.rag.kg_rag import KGRAGSystem
from src.rag.naive_rag import NaiveRAGSystem
from src.evaluation.parsing_metrics import ParsingMetrics
from src.evaluation.ragas_metrics import RAGASMetrics
from src.evaluation.complexity_metrics import ComplexityMetrics


class PilotExperimentRunner:
    """파일럿 실험 실행 클래스"""
    
    def __init__(self, output_dir: str = "results/pilot"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.parsing_metrics = ParsingMetrics()
        self.ragas_metrics = RAGASMetrics()
        self.complexity_metrics = ComplexityMetrics()
        
        self.parsers = {
            'labeled': LabeledTableParser(),
            'naive': NaiveTableParser()
        }
    
    def stratified_sampling(self,
                           tables: List[pd.DataFrame],
                           samples_per_level: int = 20,
                           random_state: int = 42) -> Dict[str, List[pd.DataFrame]]:
        """
        복잡도별 계층적 샘플링
        
        Args:
            tables: 테이블 리스트
            samples_per_level: 복잡도 레벨당 샘플 수
            random_state: 랜덤 시드
        
        Returns:
            복잡도 레벨별 테이블 딕셔너리
        """
        # 각 테이블의 복잡도 계산
        complexity_scores = []
        table_structures = []
        
        for table in tables:
            # 테이블을 구조 딕셔너리로 변환
            structure = self._dataframe_to_structure(table)
            complexity = self.complexity_metrics.calculate_complexity(structure)
            complexity_scores.append(complexity['complexity_level'])
            table_structures.append(structure)
        
        # 복잡도 레벨별로 그룹화
        complexity_df = pd.DataFrame({
            'table_idx': range(len(tables)),
            'complexity_level': complexity_scores
        })
        
        # 각 레벨별 샘플링
        sampled_tables = {}
        for level in ['low', 'medium', 'high']:
            level_tables = complexity_df[complexity_df['complexity_level'] == level]
            
            if len(level_tables) == 0:
                sampled_tables[level] = []
                continue
            
            # 샘플링
            n_samples = min(samples_per_level, len(level_tables))
            sampled_indices = level_tables.sample(
                n=n_samples, 
                random_state=random_state
            )['table_idx'].tolist()
            
            sampled_tables[level] = [tables[i] for i in sampled_indices]
        
        return sampled_tables
    
    def _dataframe_to_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """DataFrame을 구조 딕셔너리로 변환"""
        cells = []
        for row_idx, row in df.iterrows():
            for col_idx, value in enumerate(row):
                cells.append({
                    'row': row_idx,
                    'col': col_idx,
                    'rowspan': 1,
                    'colspan': 1,
                    'text': str(value) if pd.notna(value) else ''
                })
        
        return {'cells': cells}
    
    def experiment_1a_parsing(self,
                            tables: List[pd.DataFrame],
                            ground_truth: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        실험 1A: 파싱 성능 평가 (파일럿)
        
        Args:
            tables: 테이블 리스트
            ground_truth: 정답 구조 리스트 (선택적)
        
        Returns:
            실험 결과 딕셔너리
        """
        results = {
            'labeled_parsing': [],
            'naive_parsing': [],
            'comparison': []
        }
        
        for i, table in enumerate(tqdm(tables, desc="파싱 평가")):
            table_id = f"table_{i}"
            
            # 레이블링 기반 파싱
            labeled_cells = self.parsers['labeled'].parse(table)
            labeled_structure = self.parsers['labeled'].to_structured_format(labeled_cells)
            
            # Naive 파싱
            naive_structure = self.parsers['naive'].parse(table)
            
            # 평가 메트릭 계산
            if ground_truth and i < len(ground_truth):
                gt_structure = ground_truth[i]
                
                # GriTS, 헤더 메트릭
                parsing_metrics = self.parsing_metrics.evaluate_parsing(
                    predicted_table=labeled_structure,
                    ground_truth_table=gt_structure
                )
                
                results['labeled_parsing'].append({
                    'table_id': table_id,
                    'metrics': parsing_metrics,
                    'structure': labeled_structure
                })
            else:
                # 정답이 없으면 구조 정보만 저장
                results['labeled_parsing'].append({
                    'table_id': table_id,
                    'structure': labeled_structure
                })
            
            results['naive_parsing'].append({
                'table_id': table_id,
                'structure': naive_structure
            })
        
        return results
    
    def experiment_2a_rag(self,
                          tables: List[pd.DataFrame],
                          queries: List[str],
                          ground_truth: Dict[str, List[str]],
                          include_baselines: bool = True) -> Dict[str, Any]:
        """
        실험 2A: RAG 성능 평가 (파일럿)
        
        Args:
            tables: 테이블 리스트
            queries: 질문 리스트
            ground_truth: 질문별 정답 테이블 ID 딕셔너리
            include_baselines: 베이스라인 포함 여부
        
        Returns:
            실험 결과 딕셔너리
        """
        results = {
            'kg_rag': [],
            'naive_rag': [],
            'comparison': []
        }
        
        # RAG 시스템 구축
        kg_rag = KGRAGSystem(use_labeled_parsing=True)
        naive_rag = NaiveRAGSystem()
        
        for i, table in enumerate(tables):
            kg_rag.add_table(table, f"table_{i}")
            naive_rag.add_table(table, f"table_{i}")
        
        kg_rag.build_index()
        naive_rag.build_index()
        
        # 각 쿼리 평가
        for idx, query in enumerate(tqdm(queries, desc="RAG 평가")):
            query_id = f"query_{idx}"
            gt_ids = ground_truth.get(query, [])
            
            top_k = max(5, len(gt_ids) * 2) if gt_ids else 5
            top_k = min(top_k, len(tables))
            
            # KG RAG 검색
            kg_results = kg_rag.retrieve(query, top_k=top_k)
            kg_contexts = [r.get('context', '') for r in kg_results]
            
            # Naive RAG 검색
            naive_results = naive_rag.retrieve(query, top_k=top_k)
            naive_contexts = [r.get('context', '') for r in naive_results]
            
            # RAGAS 메트릭 계산 (답변이 있는 경우)
            # 여기서는 컨텍스트만 평가
            kg_ragas = self.ragas_metrics.evaluate_rag(
                question=query,
                answer='',  # 답변이 없으면 빈 문자열
                contexts=kg_contexts,
                ground_truth_contexts=gt_ids
            )
            
            naive_ragas = self.ragas_metrics.evaluate_rag(
                question=query,
                answer='',
                contexts=naive_contexts,
                ground_truth_contexts=gt_ids
            )
            
            results['kg_rag'].append({
                'query_id': query_id,
                'query': query,
                'results': kg_results,
                'metrics': kg_ragas
            })
            
            results['naive_rag'].append({
                'query_id': query_id,
                'query': query,
                'results': naive_results,
                'metrics': naive_ragas
            })
            
            # 비교
            comparison = {
                'query': query,
                'kg_rag': kg_ragas,
                'naive_rag': naive_ragas,
                'improvement': {}
            }
            
            for metric in kg_ragas.keys():
                if metric in naive_ragas:
                    kg_score = kg_ragas[metric]
                    naive_score = naive_ragas[metric]
                    if naive_score > 0:
                        improvement = ((kg_score - naive_score) / naive_score) * 100
                        comparison['improvement'][metric] = improvement
            
            results['comparison'].append(comparison)
        
        return results
    
    def experiment_3a_complexity_analysis(self,
                                         sampled_tables: Dict[str, List[pd.DataFrame]],
                                         queries: Optional[List[str]] = None,
                                         ground_truth: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """
        실험 3A: 복잡도 분석 (파일럿)
        
        Args:
            sampled_tables: 복잡도 레벨별 테이블 딕셔너리
            queries: 질문 리스트 (선택적)
            ground_truth: 질문별 정답 (선택적)
        
        Returns:
            복잡도별 실험 결과 딕셔너리
        """
        results = {}
        
        for level in ['low', 'medium', 'high']:
            level_tables = sampled_tables.get(level, [])
            
            if not level_tables:
                results[level] = None
                continue
            
            print(f"\n복잡도 레벨: {level} ({len(level_tables)}개 테이블)")
            
            # 실험 1: 파싱 성능
            parsing_results = self.experiment_1a_parsing(level_tables)
            
            # 실험 2: RAG 성능 (쿼리가 있는 경우)
            rag_results = None
            if queries and ground_truth:
                rag_results = self.experiment_2a_rag(level_tables, queries, ground_truth)
            
            results[level] = {
                'parsing': parsing_results,
                'rag': rag_results
            }
        
        return results
    
    def ablation_study_parsing(self,
                               tables: List[pd.DataFrame],
                               ground_truth: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        파싱 Ablation Study
        
        Ablation 구성:
        - Baseline (Full): 레이블링 + 헤더 감지 + 병합 셀 처리
        - Ablation 1: 레이블링 - 헤더 감지
        - Ablation 2: 레이블링 - 병합 셀 처리
        - Ablation 3: 헤더 감지만 (레이블링 제거)
        - Ablation 4: Naive 파싱
        """
        results = {
            'baseline_full': [],
            'ablation_1_no_header': [],
            'ablation_2_no_merged': [],
            'ablation_3_header_only': [],
            'ablation_4_naive': []
        }
        
        for i, table in enumerate(tqdm(tables, desc="Ablation Study")):
            table_id = f"table_{i}"
            
            # Baseline (Full)
            labeled_cells = self.parsers['labeled'].parse(table)
            baseline_structure = self.parsers['labeled'].to_structured_format(labeled_cells)
            
            # Ablation 1: 헤더 감지 제거 (간단한 구현)
            ablation_1_structure = baseline_structure.copy()
            ablation_1_structure['headers'] = []
            
            # Ablation 2: 병합 셀 처리 제거 (간단한 구현)
            ablation_2_structure = baseline_structure.copy()
            for cell in ablation_2_structure.get('cells', []):
                cell['rowspan'] = 1
                cell['colspan'] = 1
            
            # Ablation 3: 헤더만 (레이블링 제거는 별도 파서 필요)
            # 여기서는 Naive 파서 사용
            ablation_3_structure = self.parsers['naive'].parse(table)
            
            # Ablation 4: Naive 파싱
            ablation_4_structure = self.parsers['naive'].parse(table)
            
            # 평가
            if ground_truth and i < len(ground_truth):
                gt_structure = ground_truth[i]
                
                baseline_metrics = self.parsing_metrics.evaluate_parsing(
                    baseline_structure, gt_structure
                )
                ablation_1_metrics = self.parsing_metrics.evaluate_parsing(
                    ablation_1_structure, gt_structure
                )
                ablation_2_metrics = self.parsing_metrics.evaluate_parsing(
                    ablation_2_structure, gt_structure
                )
                ablation_3_metrics = self.parsing_metrics.evaluate_parsing(
                    ablation_3_structure, gt_structure
                )
                ablation_4_metrics = self.parsing_metrics.evaluate_parsing(
                    ablation_4_structure, gt_structure
                )
                
                results['baseline_full'].append({
                    'table_id': table_id,
                    'metrics': baseline_metrics
                })
                results['ablation_1_no_header'].append({
                    'table_id': table_id,
                    'metrics': ablation_1_metrics
                })
                results['ablation_2_no_merged'].append({
                    'table_id': table_id,
                    'metrics': ablation_2_metrics
                })
                results['ablation_3_header_only'].append({
                    'table_id': table_id,
                    'metrics': ablation_3_metrics
                })
                results['ablation_4_naive'].append({
                    'table_id': table_id,
                    'metrics': ablation_4_metrics
                })
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """결과 저장"""
        output_path = self.output_dir / filename
        
        # JSON으로 저장 (구조 딕셔너리는 제외)
        save_results = {}
        for key, value in results.items():
            if isinstance(value, list):
                save_results[key] = [
                    {k: v for k, v in item.items() if k != 'structure'}
                    for item in value
                ]
            else:
                save_results[key] = value
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_results, f, ensure_ascii=False, indent=2)
        
        print(f"결과 저장: {output_path}")

