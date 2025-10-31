"""
Knowledge Graph 기반 RAG 시스템
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import networkx as nx
from ..kg.table_to_kg import TableToKGConverter
import pandas as pd


class KGRAGSystem:
    """Knowledge Graph 기반 RAG 시스템"""
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 use_labeled_parsing: bool = True):
        """
        Args:
            embedding_model: 임베딩 모델 이름
            use_labeled_parsing: 레이블링 기반 파싱 사용 여부
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.kg_converter = TableToKGConverter(use_labeled_parsing=use_labeled_parsing)
        self.graphs = []  # 저장된 그래프들
        self.graph_texts = []  # 그래프 텍스트 표현
        self.embeddings = None  # FAISS 인덱스
        self.index = None
    
    def add_table(self, table_data: pd.DataFrame, table_id: str):
        """
        테이블을 KG로 변환하여 RAG 시스템에 추가
        
        Args:
            table_data: 추가할 테이블
            table_id: 테이블 식별자
        """
        # 테이블을 KG로 변환
        graph = self.kg_converter.convert(table_data, table_id)
        self.graphs.append({
            'graph': graph,
            'table_id': table_id,
            'table_data': table_data
        })
        
        # 그래프를 텍스트로 변환
        graph_text = self.kg_converter.graph_to_text(graph, format="triples")
        self.graph_texts.append({
            'text': graph_text,
            'table_id': table_id,
            'graph': graph
        })
    
    def build_index(self):
        """임베딩 인덱스 구축"""
        if not self.graph_texts:
            raise ValueError("그래프가 없습니다. 먼저 테이블을 추가하세요.")
        
        # 모든 그래프 텍스트 임베딩
        texts = [item['text'] for item in self.graph_texts]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # FAISS 인덱스 생성
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.embeddings = embeddings
        self.index.add(embeddings.astype('float32'))
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        쿼리에 대한 관련 그래프 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 k개 결과
        
        Returns:
            관련 그래프 정보 리스트
        """
        if self.index is None:
            self.build_index()
        
        # 쿼리 임베딩
        query_embedding = self.embedding_model.encode([query])
        
        # 유사도 검색
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.graph_texts):
                result = {
                    'table_id': self.graph_texts[idx]['table_id'],
                    'graph_text': self.graph_texts[idx]['text'],
                    'graph': self.graph_texts[idx]['graph'],
                    'score': float(1 / (1 + dist)),  # 거리를 점수로 변환
                    'distance': float(dist)
                }
                results.append(result)
        
        return results
    
    def retrieve_subgraph(self, query: str, top_k: int = 5, 
                         max_nodes: int = 50) -> List[nx.DiGraph]:
        """
        쿼리에 대한 관련 서브그래프 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 k개 그래프
            max_nodes: 서브그래프 최대 노드 수
        
        Returns:
            관련 서브그래프 리스트
        """
        results = self.retrieve(query, top_k)
        subgraphs = []
        
        for result in results:
            graph = result['graph']
            # 간단한 필터링: 쿼리 키워드가 포함된 노드 중심으로 서브그래프 추출
            query_keywords = query.lower().split()
            
            relevant_nodes = []
            for node, data in graph.nodes(data=True):
                node_text = str(data.get('value', '')).lower()
                if any(keyword in node_text for keyword in query_keywords):
                    relevant_nodes.append(node)
            
            if relevant_nodes:
                # 관련 노드와 인접 노드들을 포함한 서브그래프 생성
                subgraph_nodes = set(relevant_nodes)
                for node in relevant_nodes:
                    subgraph_nodes.update(graph.neighbors(node))
                    subgraph_nodes.update(graph.predecessors(node))
                
                # 노드 수 제한
                if len(subgraph_nodes) > max_nodes:
                    # 점수 기반으로 정렬하여 상위 노드만 선택
                    node_scores = {}
                    for node in subgraph_nodes:
                        data = graph.nodes[node]
                        score = sum(1 for keyword in query_keywords 
                                  if keyword in str(data.get('value', '')).lower())
                        node_scores[node] = score
                    
                    top_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
                    subgraph_nodes = set(node for node, _ in top_nodes)
                
                subgraph = graph.subgraph(subgraph_nodes).copy()
                subgraphs.append(subgraph)
            else:
                # 관련 노드가 없으면 전체 그래프 반환 (크기 제한)
                if len(graph.nodes) <= max_nodes:
                    subgraphs.append(graph.copy())
                else:
                    # 무작위 샘플링
                    sample_nodes = list(graph.nodes())[:max_nodes]
                    subgraphs.append(graph.subgraph(sample_nodes).copy())
        
        return subgraphs
    
    def generate_context(self, query: str, top_k: int = 3) -> str:
        """
        쿼리에 대한 컨텍스트 생성 (RAG용)
        
        Args:
            query: 쿼리
            top_k: 사용할 상위 k개 그래프
        
        Returns:
            컨텍스트 문자열
        """
        results = self.retrieve(query, top_k)
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"=== Table {result['table_id']} (Score: {result['score']:.3f}) ===")
            context_parts.append(result['graph_text'])
            context_parts.append("")
        
        return "\n".join(context_parts)

