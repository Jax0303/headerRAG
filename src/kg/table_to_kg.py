"""
테이블을 Knowledge Graph로 변환하는 모듈
"""

from typing import List, Dict, Any, Optional, Tuple
import networkx as nx
from rdflib import Graph, Namespace, Literal, URIRef, BNode
from rdflib.namespace import RDF, RDFS, XSD
import pandas as pd
from ..parsing.labeled_parser import LabeledTableParser, CellLabel


class TableToKGConverter:
    """테이블을 Knowledge Graph로 변환"""
    
    def __init__(self, use_labeled_parsing: bool = True):
        """
        Args:
            use_labeled_parsing: 레이블링 기반 파싱 사용 여부
        """
        self.use_labeled_parsing = use_labeled_parsing
        if use_labeled_parsing:
            self.parser = LabeledTableParser()
        
        # RDF 네임스페이스 정의
        self.ns = Namespace("http://example.org/table/")
        self.ns_prop = Namespace("http://example.org/table/property/")
    
    def convert(self, table_data: pd.DataFrame,
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
        
        if self.use_labeled_parsing:
            labeled_cells = self.parser.parse(table_data)
            self._build_graph_from_labeled_cells(G, labeled_cells, table_id)
        else:
            self._build_graph_from_naive_parsing(G, table_data, table_id)
        
        return G
    
    def convert_to_rdf(self, table_data: pd.DataFrame,
                      table_id: str = "table_1") -> Graph:
        """
        테이블을 RDF 그래프로 변환
        
        Args:
            table_data: 변환할 테이블
            table_id: 테이블 식별자
        
        Returns:
            RDFLib Graph
        """
        rdf_graph = Graph()
        rdf_graph.bind("table", self.ns)
        rdf_graph.bind("prop", self.ns_prop)
        
        if self.use_labeled_parsing:
            labeled_cells = self.parser.parse(table_data)
            self._build_rdf_from_labeled_cells(rdf_graph, labeled_cells, table_id)
        else:
            self._build_rdf_from_naive_parsing(rdf_graph, table_data, table_id)
        
        return rdf_graph
    
    def _build_graph_from_labeled_cells(self, G: nx.DiGraph,
                                       labeled_cells: List[CellLabel],
                                       table_id: str):
        """레이블링된 셀들로부터 그래프 구축"""
        # 테이블 노드 생성
        table_node = f"table:{table_id}"
        G.add_node(table_node, type="Table", id=table_id)
        
        # 헤더와 데이터 셀 분류
        headers = {}
        row_headers = {}
        data_cells = []
        
        for cell in labeled_cells:
            if cell.cell_type == 'column_header':
                headers[cell.col] = cell
            elif cell.cell_type == 'row_header':
                row_headers[cell.row] = cell
            elif cell.cell_type == 'data':
                data_cells.append(cell)
        
        # 헤더 노드 생성 및 연결
        for col_idx, header_cell in headers.items():
            header_node = f"header:{table_id}:col:{col_idx}"
            G.add_node(header_node, 
                      type="ColumnHeader",
                      value=str(header_cell.value),
                      semantic_label=header_cell.semantic_label)
            G.add_edge(table_node, header_node, relation="hasHeader")
        
        # 행 헤더 노드 생성
        for row_idx, row_header_cell in row_headers.items():
            row_header_node = f"row_header:{table_id}:row:{row_idx}"
            G.add_node(row_header_node,
                      type="RowHeader",
                      value=str(row_header_cell.value),
                      semantic_label=row_header_cell.semantic_label)
            G.add_edge(table_node, row_header_node, relation="hasRowHeader")
        
        # 데이터 셀 노드 생성 및 연결
        for data_cell in data_cells:
            cell_node = f"cell:{table_id}:{data_cell.row}:{data_cell.col}"
            G.add_node(cell_node,
                      type="DataCell",
                      value=str(data_cell.value),
                      semantic_label=data_cell.semantic_label)
            
            # 헤더와 연결
            if data_cell.col in headers:
                header_node = f"header:{table_id}:col:{data_cell.col}"
                G.add_edge(header_node, cell_node, relation="hasCell")
            
            # 행 헤더와 연결
            if data_cell.row in row_headers:
                row_header_node = f"row_header:{table_id}:row:{data_cell.row}"
                G.add_edge(row_header_node, cell_node, relation="hasCell")
            
            # 테이블과 직접 연결
            G.add_edge(table_node, cell_node, relation="hasCell")
        
        # 인접 셀 관계 추가 (상하좌우)
        for i, cell1 in enumerate(data_cells):
            for j, cell2 in enumerate(data_cells[i+1:], start=i+1):
                if self._are_adjacent(cell1, cell2):
                    node1 = f"cell:{table_id}:{cell1.row}:{cell1.col}"
                    node2 = f"cell:{table_id}:{cell2.row}:{cell2.col}"
                    G.add_edge(node1, node2, relation="adjacent")
    
    def _build_graph_from_naive_parsing(self, G: nx.DiGraph,
                                       table_data: pd.DataFrame,
                                       table_id: str):
        """Naive 파싱으로부터 그래프 구축"""
        table_node = f"table:{table_id}"
        G.add_node(table_node, type="Table", id=table_id)
        
        # 각 셀을 노드로 생성
        for i in range(len(table_data)):
            for j, col_name in enumerate(table_data.columns):
                value = table_data.iloc[i, j]
                if pd.notna(value):
                    cell_node = f"cell:{table_id}:{i}:{j}"
                    G.add_node(cell_node,
                              type="Cell",
                              value=str(value),
                              column=col_name,
                              row=i)
                    G.add_edge(table_node, cell_node, relation="hasCell")
                    
                    # 컬럼 헤더와 연결
                    col_node = f"column:{table_id}:{col_name}"
                    if not G.has_node(col_node):
                        G.add_node(col_node, type="Column", name=col_name)
                        G.add_edge(table_node, col_node, relation="hasColumn")
                    G.add_edge(col_node, cell_node, relation="contains")
    
    def _build_rdf_from_labeled_cells(self, rdf_graph: Graph,
                                     labeled_cells: List[CellLabel],
                                     table_id: str):
        """레이블링된 셀들로부터 RDF 그래프 구축"""
        table_uri = self.ns[table_id]
        rdf_graph.add((table_uri, RDF.type, self.ns["Table"]))
        
        for cell in labeled_cells:
            cell_uri = self.ns[f"{table_id}_cell_{cell.row}_{cell.col}"]
            rdf_graph.add((cell_uri, RDF.type, self.ns["Cell"]))
            rdf_graph.add((cell_uri, self.ns_prop["value"], Literal(str(cell.value))))
            rdf_graph.add((cell_uri, self.ns_prop["cellType"], Literal(cell.cell_type)))
            rdf_graph.add((table_uri, self.ns_prop["hasCell"], cell_uri))
            
            if cell.semantic_label:
                rdf_graph.add((cell_uri, self.ns_prop["semanticLabel"], 
                             Literal(cell.semantic_label)))
    
    def _build_rdf_from_naive_parsing(self, rdf_graph: Graph,
                                     table_data: pd.DataFrame,
                                     table_id: str):
        """Naive 파싱으로부터 RDF 그래프 구축"""
        table_uri = self.ns[table_id]
        rdf_graph.add((table_uri, RDF.type, self.ns["Table"]))
        
        for i in range(len(table_data)):
            for j, col_name in enumerate(table_data.columns):
                value = table_data.iloc[i, j]
                if pd.notna(value):
                    cell_uri = self.ns[f"{table_id}_cell_{i}_{j}"]
                    rdf_graph.add((cell_uri, RDF.type, self.ns["Cell"]))
                    rdf_graph.add((cell_uri, self.ns_prop["value"], Literal(str(value))))
                    rdf_graph.add((cell_uri, self.ns_prop["column"], Literal(col_name)))
                    rdf_graph.add((table_uri, self.ns_prop["hasCell"], cell_uri))
    
    def _are_adjacent(self, cell1: CellLabel, cell2: CellLabel) -> bool:
        """두 셀이 인접한지 확인"""
        row_diff = abs(cell1.row - cell2.row)
        col_diff = abs(cell1.col - cell2.col)
        return (row_diff == 1 and col_diff == 0) or (row_diff == 0 and col_diff == 1)
    
    def graph_to_text(self, G: nx.DiGraph, format: str = "triples") -> str:
        """
        그래프를 텍스트 형식으로 변환 (RAG에 사용)
        
        Args:
            G: NetworkX 그래프
            format: 출력 형식 ('triples', 'nodes', 'full')
        
        Returns:
            텍스트 형식의 그래프 표현
        """
        if format == "triples":
            lines = []
            for u, v, data in G.edges(data=True):
                rel = data.get('relation', 'relatedTo')
                u_label = G.nodes[u].get('value', u)
                v_label = G.nodes[v].get('value', v)
                lines.append(f"{u_label} --[{rel}]--> {v_label}")
            return "\n".join(lines)
        
        elif format == "nodes":
            lines = []
            for node, data in G.nodes(data=True):
                node_type = data.get('type', 'Node')
                value = data.get('value', node)
                lines.append(f"{node_type}: {value}")
            return "\n".join(lines)
        
        else:  # full
            text_parts = []
            text_parts.append("=== Nodes ===")
            for node, data in G.nodes(data=True):
                text_parts.append(f"{node}: {data}")
            text_parts.append("\n=== Edges ===")
            for u, v, data in G.edges(data=True):
                text_parts.append(f"{u} -> {v}: {data}")
            return "\n".join(text_parts)

