"""
Naive 파싱 기반 RAG 시스템
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
from ..parsing.naive_parser import NaiveTableParser


class NaiveRAGSystem:
    """Naive 파싱 기반 RAG 시스템"""
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Args:
            embedding_model: 임베딩 모델 이름
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.parser = NaiveTableParser()
        self.tables = []  # 저장된 테이블들
        self.table_texts = []  # 테이블 텍스트 표현
        self.embeddings = None  # FAISS 인덱스
        self.index = None
    
    def add_table(self, table_data: pd.DataFrame, table_id: str):
        """
        테이블을 RAG 시스템에 추가
        
        Args:
            table_data: 추가할 테이블
            table_id: 테이블 식별자
        """
        self.tables.append({
            'table_data': table_data,
            'table_id': table_id
        })
        
        # 테이블을 텍스트로 변환
        table_text = self.parser.to_text_format(table_data, include_headers=True)
        self.table_texts.append({
            'text': table_text,
            'table_id': table_id,
            'table_data': table_data
        })
    
    def build_index(self):
        """임베딩 인덱스 구축"""
        if not self.table_texts:
            raise ValueError("테이블이 없습니다. 먼저 테이블을 추가하세요.")
        
        # 모든 테이블 텍스트 임베딩
        texts = [item['text'] for item in self.table_texts]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # FAISS 인덱스 생성
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.embeddings = embeddings
        self.index.add(embeddings.astype('float32'))
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        쿼리에 대한 관련 테이블 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 k개 결과
        
        Returns:
            관련 테이블 정보 리스트
        """
        if self.index is None:
            self.build_index()
        
        # 쿼리 임베딩
        query_embedding = self.embedding_model.encode([query])
        
        # 유사도 검색 (top_k를 충분히 크게 하여 Recall 개선)
        search_k = min(top_k * 2, len(self.table_texts))  # 더 많은 후보 검색
        distances, indices = self.index.search(query_embedding.astype('float32'), search_k)
        
        results = []
        seen_table_ids = set()  # 중복 제거
        
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.table_texts):
                table_id = self.table_texts[idx]['table_id']
                
                # 중복 제거
                if table_id in seen_table_ids:
                    continue
                seen_table_ids.add(table_id)
                
                # 거리를 점수로 변환
                score = float(1 / (1 + dist))
                
                result = {
                    'table_id': table_id,
                    'table_text': self.table_texts[idx]['text'],
                    'table_data': self.table_texts[idx]['table_data'],
                    'score': score,
                    'distance': float(dist)
                }
                results.append(result)
                
                # 원하는 개수만큼 수집
                if len(results) >= top_k:
                    break
        
        return results
    
    def generate_context(self, query: str, top_k: int = 3) -> str:
        """
        쿼리에 대한 컨텍스트 생성 (RAG용)
        
        Args:
            query: 쿼리
            top_k: 사용할 상위 k개 테이블
        
        Returns:
            컨텍스트 문자열
        """
        results = self.retrieve(query, top_k)
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"=== Table {result['table_id']} (Score: {result['score']:.3f}) ===")
            context_parts.append(result['table_text'])
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def extract_answer(self, query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        쿼리에 대한 답변 추출 (간단한 키워드 매칭)
        
        Args:
            query: 쿼리
            top_k: 사용할 상위 k개 테이블
        
        Returns:
            추출된 답변 리스트
        """
        results = self.retrieve(query, top_k)
        answers = []
        
        query_lower = query.lower()
        query_keywords = query_lower.split()
        
        for result in results:
            table_data = result['table_data']
            matched_cells = []
            
            # 테이블에서 키워드가 포함된 셀 찾기
            for i in range(len(table_data)):
                for j, col in enumerate(table_data.columns):
                    value = str(table_data.iloc[i, j])
                    if any(keyword in value.lower() for keyword in query_keywords):
                        matched_cells.append({
                            'row': i,
                            'column': col,
                            'value': value
                        })
            
            if matched_cells:
                answers.append({
                    'table_id': result['table_id'],
                    'matched_cells': matched_cells,
                    'score': result['score']
                })
        
        return answers

