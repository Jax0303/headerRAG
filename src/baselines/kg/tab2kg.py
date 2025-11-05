"""
Tab2KG 베이스라인 구현
표를 Knowledge Graph로 변환하는 베이스라인 모델
"""

import os
import warnings
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import networkx as nx
from rdflib import Graph, Namespace, Literal, URIRef, BNode
from rdflib.namespace import RDF, RDFS, XSD

# Tab2KG 저장소 경로
TAB2KG_REPO_PATH = os.environ.get('TAB2KG_REPO_PATH', None)


class Tab2KGBaseline:
    """
    Tab2KG 베이스라인 Knowledge Graph 변환기
    
    표를 RDF 그래프로 자동 변환합니다.
    기존 TableToKGConverter와 호환되는 인터페이스를 제공합니다.
    """
    
    def __init__(self,
                 repo_path: Optional[str] = None,
                 use_semantic_profiles: bool = True):
        """
        Args:
            repo_path: Tab2KG 저장소 경로 (선택적)
            use_semantic_profiles: 시맨틱 프로필 기반 접근 사용 여부
        """
        self.repo_path = repo_path or TAB2KG_REPO_PATH
        self.use_semantic_profiles = use_semantic_profiles
        
        # RDF 네임스페이스 정의
        self.ns = Namespace("http://example.org/table/")
        self.ns_prop = Namespace("http://example.org/table/property/")
        self.ns_entity = Namespace("http://example.org/entity/")
        
        self._check_installation()
    
    def _check_installation(self):
        """Tab2KG 설치 확인 및 안내"""
        if self.repo_path is None:
            warnings.warn(
                "Tab2KG 저장소 경로가 설정되지 않았습니다.\n"
                "설치 방법:\n"
                "1. git clone https://github.com/sgottsch/Tab2KG.git\n"
                "2. 환경변수 설정: export TAB2KG_REPO_PATH=/path/to/Tab2KG\n"
                "또는 직접 사용: Tab2KGBaseline(repo_path='/path/to/Tab2KG')",
                UserWarning
            )
    
    def convert(self,
                table_data: pd.DataFrame,
                table_id: str = "table_1",
                output_format: str = "networkx") -> Any:
        """
        테이블을 Knowledge Graph로 변환
        
        Args:
            table_data: 변환할 테이블
            table_id: 테이블 식별자
            output_format: 출력 형식 ('networkx', 'rdf', 'dict')
        
        Returns:
            변환된 Knowledge Graph (형식에 따라 다름)
        """
        if output_format == "networkx":
            return self.convert_to_networkx(table_data, table_id)
        elif output_format == "rdf":
            return self.convert_to_rdf(table_data, table_id)
        elif output_format == "dict":
            return self.convert_to_dict(table_data, table_id)
        else:
            raise ValueError(f"지원하지 않는 출력 형식: {output_format}")
    
    def convert_to_networkx(self,
                           table_data: pd.DataFrame,
                           table_id: str = "table_1") -> nx.DiGraph:
        """
        테이블을 NetworkX 그래프로 변환
        
        Args:
            table_data: 변환할 테이블
            table_id: 테이블 식별자
        
        Returns:
            NetworkX DiGraph
        """
        G = nx.DiGraph()
        
        n_rows, n_cols = table_data.shape
        
        # 테이블 노드 추가
        table_node = f"table:{table_id}"
        G.add_node(table_node, type="Table", table_id=table_id)
        
        # 컬럼 노드 및 엣지 추가
        for j, col_name in enumerate(table_data.columns):
            col_node = f"column:{table_id}:{j}"
            G.add_node(col_node, type="Column", name=str(col_name), index=j)
            G.add_edge(table_node, col_node, relation="hasColumn")
        
        # 행 데이터를 엔티티로 변환
        for i in range(n_rows):
            row_data = table_data.iloc[i]
            
            # 첫 번째 컬럼을 엔티티 식별자로 사용 (있는 경우)
            entity_id = None
            if len(table_data.columns) > 0:
                first_val = row_data.iloc[0]
                if pd.notna(first_val):
                    entity_id = f"entity:{table_id}:{i}:{str(first_val)}"
                else:
                    entity_id = f"entity:{table_id}:{i}"
            else:
                entity_id = f"entity:{table_id}:{i}"
            
            # 엔티티 노드 추가
            G.add_node(entity_id, type="Entity", row_index=i)
            G.add_edge(table_node, entity_id, relation="hasRow")
            
            # 속성 엣지 추가
            for j, col_name in enumerate(table_data.columns):
                cell_value = row_data.iloc[j]
                if pd.notna(cell_value):
                    col_node = f"column:{table_id}:{j}"
                    
                    # 속성 값 노드 생성
                    value_node = f"value:{table_id}:{i}:{j}"
                    G.add_node(value_node, type="Value", value=str(cell_value))
                    
                    # 엔티티 -> 컬럼 -> 값 관계
                    G.add_edge(entity_id, col_node, relation="hasAttribute")
                    G.add_edge(col_node, value_node, relation="hasValue")
        
        return G
    
    def convert_to_rdf(self,
                      table_data: pd.DataFrame,
                      table_id: str = "table_1") -> Graph:
        """
        테이블을 RDF 그래프로 변환
        
        Args:
            table_data: 변환할 테이블
            table_id: 테이블 식별자
        
        Returns:
            RDF Graph
        """
        g = Graph()
        
        # 네임스페이스 바인딩
        g.bind("table", self.ns)
        g.bind("prop", self.ns_prop)
        g.bind("entity", self.ns_entity)
        
        n_rows, n_cols = table_data.shape
        
        # 테이블 리소스
        table_uri = self.ns[table_id]
        g.add((table_uri, RDF.type, self.ns["Table"]))
        g.add((table_uri, RDFS.label, Literal(f"Table {table_id}")))
        
        # 컬럼 리소스
        for j, col_name in enumerate(table_data.columns):
            col_uri = self.ns[f"{table_id}/column/{j}"]
            g.add((col_uri, RDF.type, self.ns["Column"]))
            g.add((col_uri, RDFS.label, Literal(str(col_name))))
            g.add((table_uri, self.ns_prop["hasColumn"], col_uri))
        
        # 행 데이터를 엔티티로 변환
        for i in range(n_rows):
            row_data = table_data.iloc[i]
            
            # 엔티티 리소스
            entity_uri = self.ns_entity[f"{table_id}/row/{i}"]
            g.add((entity_uri, RDF.type, self.ns["Entity"]))
            g.add((table_uri, self.ns_prop["hasRow"], entity_uri))
            
            # 속성 추가
            for j, col_name in enumerate(table_data.columns):
                cell_value = row_data.iloc[j]
                if pd.notna(cell_value):
                    col_uri = self.ns[f"{table_id}/column/{j}"]
                    
                    # 값 리터럴 생성
                    if pd.api.types.is_numeric_dtype(type(cell_value)):
                        if isinstance(cell_value, float):
                            literal = Literal(float(cell_value), datatype=XSD.double)
                        else:
                            literal = Literal(int(cell_value), datatype=XSD.integer)
                    elif pd.api.types.is_datetime64_any_dtype(type(cell_value)):
                        literal = Literal(str(cell_value), datatype=XSD.dateTime)
                    else:
                        literal = Literal(str(cell_value), datatype=XSD.string)
                    
                    g.add((entity_uri, col_uri, literal))
        
        return g
    
    def convert_to_dict(self,
                       table_data: pd.DataFrame,
                       table_id: str = "table_1") -> Dict[str, Any]:
        """
        테이블을 딕셔너리 형식의 그래프로 변환
        
        Args:
            table_data: 변환할 테이블
            table_id: 테이블 식별자
        
        Returns:
            그래프 딕셔너리
        """
        graph_dict = {
            'table_id': table_id,
            'nodes': [],
            'edges': []
        }
        
        n_rows, n_cols = table_data.shape
        
        # 테이블 노드
        graph_dict['nodes'].append({
            'id': f"table:{table_id}",
            'type': 'Table',
            'properties': {'table_id': table_id}
        })
        
        # 컬럼 노드
        for j, col_name in enumerate(table_data.columns):
            col_id = f"column:{table_id}:{j}"
            graph_dict['nodes'].append({
                'id': col_id,
                'type': 'Column',
                'properties': {'name': str(col_name), 'index': j}
            })
            graph_dict['edges'].append({
                'source': f"table:{table_id}",
                'target': col_id,
                'relation': 'hasColumn'
            })
        
        # 행 데이터를 엔티티로 변환
        for i in range(n_rows):
            row_data = table_data.iloc[i]
            
            entity_id = f"entity:{table_id}:{i}"
            graph_dict['nodes'].append({
                'id': entity_id,
                'type': 'Entity',
                'properties': {'row_index': i}
            })
            graph_dict['edges'].append({
                'source': f"table:{table_id}",
                'target': entity_id,
                'relation': 'hasRow'
            })
            
            # 속성 엣지
            for j, col_name in enumerate(table_data.columns):
                cell_value = row_data.iloc[j]
                if pd.notna(cell_value):
                    col_id = f"column:{table_id}:{j}"
                    graph_dict['edges'].append({
                        'source': entity_id,
                        'target': col_id,
                        'relation': 'hasAttribute',
                        'properties': {'value': str(cell_value)}
                    })
        
        return graph_dict
    
    def save_rdf(self,
                 table_data: pd.DataFrame,
                 table_id: str,
                 output_path: str,
                 format: str = "turtle"):
        """
        RDF 그래프를 파일로 저장
        
        Args:
            table_data: 변환할 테이블
            table_id: 테이블 식별자
            output_path: 저장 경로
            format: 저장 형식 ('turtle', 'xml', 'json-ld', 'nt')
        """
        rdf_graph = self.convert_to_rdf(table_data, table_id)
        rdf_graph.serialize(destination=output_path, format=format)
        print(f"RDF 그래프 저장 완료: {output_path}")


def main():
    """테스트 코드"""
    import pandas as pd
    
    # 샘플 테이블 생성
    sample_table = pd.DataFrame({
        '연도': [2020, 2021, 2022, 2023],
        '매출액(억원)': [1000, 1200, 1500, 1800],
        '순이익(억원)': [100, 150, 200, 250]
    })
    
    # Tab2KG 변환기 초기화
    converter = Tab2KGBaseline()
    
    # NetworkX 그래프로 변환
    nx_graph = converter.convert_to_networkx(sample_table, "sample_table")
    print(f"NetworkX 그래프: {len(nx_graph.nodes())}개 노드, {len(nx_graph.edges())}개 엣지")
    
    # RDF 그래프로 변환
    rdf_graph = converter.convert_to_rdf(sample_table, "sample_table")
    print(f"RDF 그래프: {len(rdf_graph)}개 트리플")
    
    # 딕셔너리로 변환
    graph_dict = converter.convert_to_dict(sample_table, "sample_table")
    print(f"딕셔너리 그래프: {len(graph_dict['nodes'])}개 노드, {len(graph_dict['edges'])}개 엣지")


if __name__ == "__main__":
    main()

