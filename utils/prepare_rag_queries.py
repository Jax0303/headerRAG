"""
RAG 실험을 위한 쿼리 및 ground truth 준비
RAG-Evaluation-Dataset-KO의 평가 결과에서 쿼리 추출
"""

import pandas as pd
import os
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


def prepare_queries_from_dataset(
    rag_result_csv: str = "RAG-Evaluation-Dataset-KO/rag_evaluation_result.csv",
    documents_csv: str = "RAG-Evaluation-Dataset-KO/documents.csv",
    table_mapping: Dict[str, str] = None
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    평가 데이터셋에서 쿼리와 ground truth 추출
    
    Args:
        rag_result_csv: RAG 평가 결과 CSV 파일
        documents_csv: 문서 메타데이터 CSV 파일
        table_mapping: PDF 파일명 -> 테이블 ID 매핑 (None이면 자동 생성)
    
    Returns:
        (queries, ground_truth) 튜플
    """
    # 평가 결과 로드
    df_results = pd.read_csv(rag_result_csv)
    df_docs = pd.read_csv(documents_csv)
    
    # 문서명 -> 인덱스 매핑 생성
    doc_name_to_idx = {}
    for idx, row in df_docs.iterrows():
        file_name = row['file_name']
        domain = row['domain']
        doc_name_to_idx[file_name] = idx
    
    # 테이블 ID 매핑 생성
    # - table_id_to_doc: 추출된 테이블 ID -> (domain, doc_idx) 매핑
    # - table_mapping: 테이블 ID -> 실험용 table_id (table_0, table_1, ...) 매핑
    table_id_to_doc = {}  # 추출된 테이블 ID -> (domain, doc_idx)
    if table_mapping is None:
        table_mapping = {}
        extracted_dir = Path("data/extracted_tables")
        if extracted_dir.exists():
            table_idx = 0
            for domain_dir in extracted_dir.iterdir():
                if domain_dir.is_dir():
                    domain = domain_dir.name
                    for excel_file in domain_dir.glob("*.xlsx"):
                        # 원본 테이블 ID (파일명에서 .xlsx 제거)
                        original_table_id = excel_file.stem
                        # 실험용 table_id 생성
                        mapped_id = f"table_{table_idx}"
                        table_mapping[original_table_id] = mapped_id
                        
                        # 테이블 ID에서 domain과 doc_idx 추출
                        # 형식: {domain}_{doc_idx}_...
                        parts = original_table_id.split('_', 2)
                        if len(parts) >= 2:
                            try:
                                doc_idx = int(parts[1])
                                table_id_to_doc[original_table_id] = (domain, doc_idx)
                            except ValueError:
                                # 숫자로 변환 실패 시, 도메인만 저장
                                table_id_to_doc[original_table_id] = (domain, None)
                        else:
                            table_id_to_doc[original_table_id] = (domain, None)
                        
                        table_idx += 1
    
    # 쿼리와 ground truth 추출
    queries = []
    ground_truth = {}
    
    # 각 질문에 대해
    for idx, row in df_results.iterrows():
        question = row['question']
        target_file = row['target_file_name']
        domain = row['domain']
        
        # 문서 인덱스 찾기
        doc_idx = None
        if target_file in doc_name_to_idx:
            doc_idx = doc_name_to_idx[target_file]
        
        # 질문이 이미 리스트에 있으면 스킵 (중복 제거)
        if question in queries:
            if question not in ground_truth:
                ground_truth[question] = []
        else:
            queries.append(question)
            ground_truth[question] = []
        
        # 테이블 ID 찾기
        # 방법 1: 문서 인덱스 기반 매핑 (가장 정확)
        found_tables = []
        if doc_idx is not None:
            for original_table_id, mapped_id in table_mapping.items():
                if original_table_id in table_id_to_doc:
                    t_domain, t_doc_idx = table_id_to_doc[original_table_id]
                    if t_domain == domain and t_doc_idx == doc_idx:
                        if mapped_id not in ground_truth[question]:
                            found_tables.append(mapped_id)
        
        # 방법 2: 파일명 기반 매핑 (백업)
        if not found_tables:
            target_file_no_ext = target_file.replace('.pdf', '')
            for original_table_id, mapped_id in table_mapping.items():
                # 파일명이 테이블 ID에 포함되어 있는지 확인
                if target_file_no_ext in original_table_id or target_file in original_table_id:
                    if mapped_id not in found_tables:
                        found_tables.append(mapped_id)
        
        # 방법 3: 도메인만 매칭 (마지막 백업)
        if not found_tables and doc_idx is not None:
            for original_table_id, mapped_id in table_mapping.items():
                if original_table_id in table_id_to_doc:
                    t_domain, t_doc_idx = table_id_to_doc[original_table_id]
                    if t_domain == domain:
                        if mapped_id not in found_tables:
                            found_tables.append(mapped_id)
        
        # Ground truth에 추가
        ground_truth[question].extend(found_tables)
    
    # 중복 제거
    for query in ground_truth:
        ground_truth[query] = list(set(ground_truth[query]))
    
    return queries, ground_truth


def create_simple_queries(tables: List, num_queries: int = 10) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    테이블에서 간단한 쿼리 생성 (실험용)
    
    Args:
        tables: 테이블 리스트
        num_queries: 생성할 쿼리 수
    
    Returns:
        (queries, ground_truth) 튜플
    """
    queries = []
    ground_truth = {}
    
    # 각 테이블에서 간단한 쿼리 생성
    for i, table in enumerate(tables[:num_queries]):
        table_id = f"table_{i}"
        
        # 컬럼명 기반 쿼리 생성
        if not table.empty and len(table.columns) > 0:
            col_name = str(table.columns[0])
            query = f"{col_name}에 대한 정보를 알려주세요"
            queries.append(query)
            ground_truth[query] = [table_id]
        
        # 데이터 기반 쿼리 생성
        if not table.empty and len(table) > 0:
            first_val = str(table.iloc[0, 0]) if len(table.columns) > 0 else ""
            if first_val and len(first_val) < 50:
                query = f"{first_val}에 대한 정보는?"
                if query not in queries:
                    queries.append(query)
                    ground_truth[query] = [table_id]
    
    return queries[:num_queries], ground_truth


if __name__ == "__main__":
    # 쿼리와 ground truth 준비
    print("쿼리 및 ground truth 준비 중...")
    queries, ground_truth = prepare_queries_from_dataset()
    
    print(f"\n추출된 쿼리 수: {len(queries)}")
    print(f"Ground truth 매핑 수: {len(ground_truth)}")
    
    # 샘플 출력
    print("\n샘플 쿼리 (처음 5개):")
    for i, query in enumerate(queries[:5], 1):
        print(f"{i}. {query}")
        print(f"   정답 테이블: {ground_truth.get(query, [])}")

