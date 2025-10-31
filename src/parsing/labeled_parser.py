"""
레이블링 기반 테이블 파서
각 데이터셀, 헤더셀, 열 셀에 레이블을 부착하여 구조화된 파싱 수행
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class CellLabel:
    """셀 레이블 정보"""
    row: int
    col: int
    cell_type: str  # 'header', 'data', 'row_header', 'column_header', 'merged'
    value: Any
    row_span: int = 1
    col_span: int = 1
    semantic_label: Optional[str] = None  # 예: '연도', '매출액', '직원수' 등


class LabeledTableParser:
    """레이블링 기반 테이블 파서"""
    
    def __init__(self, detect_headers: bool = True, detect_merged_cells: bool = True):
        """
        Args:
            detect_headers: 헤더 자동 감지 여부
            detect_merged_cells: 병합 셀 감지 여부
        """
        self.detect_headers = detect_headers
        self.detect_merged_cells = detect_merged_cells
    
    def parse(self, table_data: pd.DataFrame, 
              row_headers: Optional[List[int]] = None,
              col_headers: Optional[List[int]] = None) -> List[CellLabel]:
        """
        테이블을 파싱하여 레이블링된 셀 리스트 반환
        
        Args:
            table_data: 파싱할 테이블 (DataFrame)
            row_headers: 행 헤더 인덱스 리스트 (None이면 자동 감지)
            col_headers: 열 헤더 인덱스 리스트 (None이면 자동 감지)
        
        Returns:
            레이블링된 셀 리스트
        """
        labeled_cells = []
        n_rows, n_cols = table_data.shape
        
        # 헤더 자동 감지
        if self.detect_headers:
            if row_headers is None:
                row_headers = self._detect_row_headers(table_data)
            if col_headers is None:
                col_headers = self._detect_column_headers(table_data)
        
        # 각 셀에 레이블 부착
        for i in range(n_rows):
            for j in range(n_cols):
                cell_value = table_data.iloc[i, j]
                
                # 셀 타입 결정
                cell_type = self._determine_cell_type(i, j, row_headers, col_headers)
                
                # 병합 셀 감지
                row_span, col_span = 1, 1
                if self.detect_merged_cells:
                    row_span, col_span = self._detect_merged_cell_span(
                        table_data, i, j
                    )
                
                # 시맨틱 레이블 추출 (간단한 휴리스틱)
                semantic_label = self._extract_semantic_label(
                    cell_value, cell_type, i, j, table_data
                )
                
                labeled_cell = CellLabel(
                    row=i,
                    col=j,
                    cell_type=cell_type,
                    value=cell_value,
                    row_span=row_span,
                    col_span=col_span,
                    semantic_label=semantic_label
                )
                labeled_cells.append(labeled_cell)
        
        return labeled_cells
    
    def _detect_row_headers(self, table_data: pd.DataFrame) -> List[int]:
        """행 헤더 자동 감지"""
        row_headers = []
        # 첫 번째 열이 주로 텍스트이고 반복 패턴이 있으면 행 헤더로 간주
        first_col = table_data.iloc[:, 0]
        text_ratio = first_col.apply(
            lambda x: isinstance(x, str) and not self._is_numeric(x)
        ).sum() / len(first_col)
        
        if text_ratio > 0.5:
            row_headers.append(0)
        
        return row_headers
    
    def _detect_column_headers(self, table_data: pd.DataFrame) -> List[int]:
        """열 헤더 자동 감지"""
        col_headers = []
        # 첫 번째 행이 주로 텍스트이면 열 헤더로 간주
        first_row = table_data.iloc[0, :]
        text_ratio = first_row.apply(
            lambda x: isinstance(x, str) and not self._is_numeric(x)
        ).sum() / len(first_row)
        
        if text_ratio > 0.5:
            col_headers.append(0)
        
        return col_headers
    
    def _determine_cell_type(self, row: int, col: int,
                            row_headers: List[int],
                            col_headers: List[int]) -> str:
        """셀 타입 결정"""
        if row in col_headers:
            if col in row_headers:
                return 'merged'  # 교차점
            return 'column_header'
        elif col in row_headers:
            return 'row_header'
        else:
            return 'data'
    
    def _detect_merged_cell_span(self, table_data: pd.DataFrame,
                                row: int, col: int) -> Tuple[int, int]:
        """병합 셀의 row_span, col_span 감지"""
        row_span, col_span = 1, 1
        value = table_data.iloc[row, col]
        
        # 같은 값이 연속된 셀 확인
        # Row span 확인
        for i in range(row + 1, len(table_data)):
            if pd.isna(table_data.iloc[i, col]) or table_data.iloc[i, col] == value:
                row_span += 1
            else:
                break
        
        # Col span 확인
        for j in range(col + 1, len(table_data.columns)):
            if pd.isna(table_data.iloc[row, j]) or table_data.iloc[row, j] == value:
                col_span += 1
            else:
                break
        
        return row_span, col_span
    
    def _extract_semantic_label(self, value: Any, cell_type: str,
                               row: int, col: int,
                               table_data: pd.DataFrame) -> Optional[str]:
        """시맨틱 레이블 추출 (간단한 휴리스틱)"""
        if pd.isna(value):
            return None
        
        value_str = str(value).strip()
        
        # 연도 패턴
        if self._is_year(value_str):
            return '연도'
        
        # 금액 패턴
        if '원' in value_str or '억' in value_str or '만' in value_str:
            return '금액'
        
        # 비율/퍼센트
        if '%' in value_str or 'percent' in value_str.lower():
            return '비율'
        
        # 숫자 패턴
        if self._is_numeric(value_str):
            if cell_type == 'column_header':
                return '측정값'
            return '데이터값'
        
        # 일반 텍스트
        if cell_type in ['row_header', 'column_header']:
            return value_str[:20]  # 레이블로 사용
        
        return None
    
    def _is_numeric(self, value: str) -> bool:
        """숫자 형식인지 확인"""
        try:
            float(value.replace(',', '').replace('원', '').replace('%', ''))
            return True
        except:
            return False
    
    def _is_year(self, value: str) -> bool:
        """연도 형식인지 확인"""
        try:
            year = int(value)
            return 1900 <= year <= 2100
        except:
            return False
    
    def to_structured_format(self, labeled_cells: List[CellLabel]) -> Dict[str, Any]:
        """
        레이블링된 셀들을 구조화된 형식으로 변환
        
        Returns:
            {
                'headers': [...],
                'data_cells': [...],
                'row_headers': [...],
                'column_headers': [...],
                'metadata': {...}
            }
        """
        result = {
            'headers': [],
            'data_cells': [],
            'row_headers': [],
            'column_headers': [],
            'metadata': {
                'total_cells': len(labeled_cells),
                'cell_types': {}
            }
        }
        
        for cell in labeled_cells:
            cell_info = {
                'row': cell.row,
                'col': cell.col,
                'value': cell.value,
                'semantic_label': cell.semantic_label,
                'span': {'row': cell.row_span, 'col': cell.col_span}
            }
            
            if cell.cell_type == 'data':
                result['data_cells'].append(cell_info)
            elif cell.cell_type == 'row_header':
                result['row_headers'].append(cell_info)
            elif cell.cell_type == 'column_header':
                result['column_headers'].append(cell_info)
            elif cell.cell_type == 'header':
                result['headers'].append(cell_info)
            
            # 통계 정보 수집
            result['metadata']['cell_types'][cell.cell_type] = \
                result['metadata']['cell_types'].get(cell.cell_type, 0) + 1
        
        return result

