"""
TableRAG 베이스라인 구현
표 기반 RAG 시스템의 베이스라인 모델
"""

import os
import warnings
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# TableRAG 저장소 경로 (선택적)
TABLERAG_REPO_PATH = os.environ.get('TABLERAG_REPO_PATH', None)


class TableRAGBaseline:
    """
    TableRAG 베이스라인 RAG 시스템
    
    표 데이터를 처리하여 RAG 질의응답을 수행합니다.
    기존 NaiveRAGSystem과 호환되는 인터페이스를 제공합니다.
    """
    
    def __init__(self,
                 embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 use_colbert: bool = False,
                 repo_path: Optional[str] = None):
        """
        Args:
            embedding_model: 임베딩 모델 이름
            use_colbert: ColBERT 모델 사용 여부 (True면 토큰 수준 임베딩)
            repo_path: TableRAG 저장소 경로 (선택적)
        """
        self.embedding_model_name = embedding_model
        self.use_colbert = use_colbert
        self.repo_path = repo_path or TABLERAG_REPO_PATH
        
        # 임베딩 모델 로드
        if use_colbert:
            self._load_colbert_model()
        else:
            self.embedding_model = SentenceTransformer(embedding_model)
        
        # 테이블 저장소
        self.tables = []
        self.table_chunks = []
        self.embeddings = None
        self.index = None
        
        self._check_installation()
    
    def _check_installation(self):
        """TableRAG 설치 확인 및 안내"""
        if self.use_colbert and self.repo_path is None:
            warnings.warn(
                "ColBERT를 사용하려면 TableRAG 저장소가 필요합니다.\n"
                "설치 방법:\n"
                "1. git clone https://github.com/YuhangWuAI/tablerag.git\n"
                "또는\n"
                "2. git clone https://github.com/yxh-y/TableRAG.git\n"
                "3. 환경변수 설정: export TABLERAG_REPO_PATH=/path/to/tablerag",
                UserWarning
            )
    
    def _load_colbert_model(self):
        """ColBERT 모델 로드 (실제 구현은 TableRAG 저장소 참조)"""
        try:
            # ColBERT 모델 로드 시도
            # 실제 구현은 TableRAG 저장소의 코드 사용
            warnings.warn(
                "ColBERT 모델 로드는 TableRAG 저장소의 코드를 사용해야 합니다. "
                "현재는 일반 SentenceTransformer를 사용합니다.",
                UserWarning
            )
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        except Exception as e:
            warnings.warn(
                f"ColBERT 모델 로드 실패: {e}. "
                "일반 SentenceTransformer를 사용합니다.",
                UserWarning
            )
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
    
    def add_table(self, 
                  table_data: pd.DataFrame, 
                  table_id: str,
                  chunk_strategy: str = "row"):
        """
        테이블을 RAG 시스템에 추가
        
        Args:
            table_data: 추가할 테이블
            table_id: 테이블 식별자
            chunk_strategy: 청킹 전략 ('row', 'cell', 'hybrid')
        """
        self.tables.append({
            'table_data': table_data,
            'table_id': table_id
        })
        
        # 테이블을 청크로 분할
        chunks = self._chunk_table(table_data, table_id, chunk_strategy)
        self.table_chunks.extend(chunks)
    
    def _chunk_table(self,
                     table_data: pd.DataFrame,
                     table_id: str,
                     strategy: str) -> List[Dict[str, Any]]:
        """
        테이블을 청크로 분할
        
        Args:
            table_data: 분할할 테이블
            table_id: 테이블 식별자
            strategy: 청킹 전략
        
        Returns:
            청크 리스트
        """
        chunks = []
        n_rows, n_cols = table_data.shape
        
        if strategy == "row":
            # 행 단위로 청킹
            for i in range(n_rows):
                row_data = table_data.iloc[i]
                chunk_text = self._row_to_text(row_data, table_data.columns)
                chunks.append({
                    'text': chunk_text,
                    'table_id': table_id,
                    'chunk_type': 'row',
                    'row_index': i
                    # table_data는 JSON 직렬화 문제로 제외
                })
        
        elif strategy == "cell":
            # 셀 단위로 청킹
            for i in range(n_rows):
                for j in range(n_cols):
                    cell_value = table_data.iloc[i, j]
                    col_name = table_data.columns[j]
                    chunk_text = f"{col_name}: {cell_value}"
                    chunks.append({
                        'text': chunk_text,
                        'table_id': table_id,
                        'chunk_type': 'cell',
                        'row_index': i,
                        'col_index': j
                        # table_data는 JSON 직렬화 문제로 제외
                    })
        
        elif strategy == "hybrid":
            # 하이브리드: 행 + 셀 조합
            # 먼저 행 단위 청크 추가
            row_chunks = self._chunk_table(table_data, table_id, "row")
            chunks.extend(row_chunks)
            
            # 중요한 셀들도 개별 청크로 추가
            # (예: 숫자 값이 큰 셀, 헤더 셀 등)
            for i in range(n_rows):
                for j in range(n_cols):
                    cell_value = table_data.iloc[i, j]
                    if pd.notna(cell_value):
                        col_name = table_data.columns[j]
                        chunk_text = f"{col_name}: {cell_value}"
                        chunks.append({
                            'text': chunk_text,
                            'table_id': table_id,
                            'chunk_type': 'cell',
                            'row_index': i,
                            'col_index': j
                            # table_data는 JSON 직렬화 문제로 제외
                        })
        
        return chunks
    
    def _row_to_text(self, row_data: pd.Series, columns: pd.Index) -> str:
        """행 데이터를 텍스트로 변환"""
        parts = []
        for col, val in zip(columns, row_data):
            if pd.notna(val):
                parts.append(f"{col}: {val}")
        return " | ".join(parts)
    
    def build_index(self):
        """임베딩 인덱스 구축"""
        if len(self.table_chunks) == 0:
            raise ValueError("인덱스를 구축할 테이블 청크가 없습니다. add_table()을 먼저 호출하세요.")
        
        # 텍스트 추출
        texts = [chunk['text'] for chunk in self.table_chunks]
        
        # 임베딩 생성
        print(f"임베딩 생성 중... ({len(texts)}개 청크)")
        self.embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # FAISS 인덱스 구축
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"인덱스 구축 완료: {len(texts)}개 벡터")
    
    def query(self, 
              query_text: str,
              top_k: int = 5,
              return_context: bool = True) -> Dict[str, Any]:
        """
        질의 수행
        
        Args:
            query_text: 질의 텍스트
            top_k: 반환할 상위 k개 결과
            return_context: 컨텍스트 정보 반환 여부
        
        Returns:
            검색 결과 및 답변
        """
        if self.index is None:
            raise ValueError("인덱스가 구축되지 않았습니다. build_index()를 먼저 호출하세요.")
        
        # 쿼리 임베딩
        query_embedding = self.embedding_model.encode(
            [query_text],
            convert_to_numpy=True
        )[0]
        
        # 검색 수행
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            top_k
        )
        
        # 결과 수집
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            chunk = self.table_chunks[idx]
            result = {
                'rank': i + 1,
                'score': float(1 / (1 + dist)),  # 거리를 점수로 변환
                'text': chunk['text'],
                'table_id': chunk['table_id'],
                'chunk_type': chunk['chunk_type']
            }
            
            if return_context:
                result['context'] = {
                    'row_index': chunk.get('row_index'),
                    'col_index': chunk.get('col_index')
                    # table_data는 JSON 직렬화 문제로 제외
                }
            
            results.append(result)
        
        return {
            'query': query_text,
            'results': results,
            'top_k': top_k
        }
    
    def answer(self,
               query_text: str,
               top_k: int = 5,
               use_llm: bool = False,
               llm_model: Optional[str] = None) -> str:
        """
        질의에 대한 답변 생성
        
        Args:
            query_text: 질의 텍스트
            top_k: 검색할 상위 k개 결과
            use_llm: LLM을 사용하여 답변 생성 여부
            llm_model: 사용할 LLM 모델 (None이면 검색 결과만 반환)
        
        Returns:
            생성된 답변 텍스트
        """
        # 검색 수행
        search_results = self.query(query_text, top_k=top_k, return_context=True)
        
        if not use_llm or llm_model is None:
            # LLM 없이 검색 결과만 반환
            top_result = search_results['results'][0] if search_results['results'] else None
            if top_result:
                return top_result['text']
            return "관련 정보를 찾을 수 없습니다."
        
        # LLM을 사용한 답변 생성
        # 실제 구현은 OpenAI API 또는 로컬 LLM 사용
        context_texts = [r['text'] for r in search_results['results']]
        context = "\n".join(context_texts)
        
        # 간단한 프롬프트 구성 (실제로는 더 정교한 프롬프트 필요)
        prompt = f"""다음 표 데이터를 바탕으로 질문에 답변하세요.

표 데이터:
{context}

질문: {query_text}

답변:"""
        
        # 실제 LLM 호출은 구현 필요
        warnings.warn(
            "LLM 기반 답변 생성은 실제 LLM API를 호출해야 합니다. "
            "현재는 검색 결과만 반환합니다.",
            UserWarning
        )
        
        return context


def main():
    """테스트 코드"""
    import pandas as pd
    
    # 샘플 테이블 생성
    sample_table = pd.DataFrame({
        '연도': [2020, 2021, 2022, 2023],
        '매출액(억원)': [1000, 1200, 1500, 1800],
        '순이익(억원)': [100, 150, 200, 250]
    })
    
    # TableRAG 초기화
    rag = TableRAGBaseline(use_colbert=False)
    
    # 테이블 추가
    rag.add_table(sample_table, "table_1", chunk_strategy="row")
    
    # 인덱스 구축
    rag.build_index()
    
    # 질의 수행
    query = "2023년 매출액은 얼마인가요?"
    results = rag.query(query, top_k=3)
    
    print(f"질의: {query}")
    print("\n검색 결과:")
    for result in results['results']:
        print(f"  [{result['rank']}] {result['text']} (점수: {result['score']:.4f})")
    
    # 답변 생성
    answer = rag.answer(query, use_llm=False)
    print(f"\n답변: {answer}")


if __name__ == "__main__":
    main()

