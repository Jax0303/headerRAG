"""
Naive 테이블 파서
레이블링 없이 단순하게 테이블을 파싱
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


class NaiveTableParser:
    """Naive 테이블 파서 - 레이블링 없이 단순 파싱"""
    
    def __init__(self, header_row: Optional[int] = 0):
        """
        Args:
            header_row: 헤더 행 인덱스 (None이면 헤더 없음)
        """
        self.header_row = header_row
    
    def parse(self, table_data: pd.DataFrame) -> Dict[str, Any]:
        """
        테이블을 단순하게 파싱
        
        Args:
            table_data: 파싱할 테이블 (DataFrame)
        
        Returns:
            파싱된 테이블 정보 (단순 형식)
        """
        result = {
            'data': table_data.to_dict('records'),
            'columns': list(table_data.columns),
            'shape': table_data.shape,
            'dtypes': {col: str(dtype) for col, dtype in table_data.dtypes.items()}
        }
        
        # 헤더가 있으면 별도로 저장
        if self.header_row is not None and self.header_row < len(table_data):
            result['header'] = table_data.iloc[self.header_row].to_dict()
        
        return result
    
    def to_text_format(self, table_data: pd.DataFrame, 
                       include_headers: bool = True) -> str:
        """
        테이블을 텍스트 형식으로 변환 (RAG에 사용)
        
        Args:
            table_data: 변환할 테이블
            include_headers: 헤더 포함 여부
        
        Returns:
            텍스트 형식의 테이블
        """
        lines = []
        
        if include_headers and self.header_row is not None:
            header = table_data.columns.tolist()
            lines.append(" | ".join(str(h) for h in header))
            lines.append("-" * len(" | ".join(str(h) for h in header)))
        
        for idx, row in table_data.iterrows():
            if self.header_row is not None and idx == self.header_row:
                continue
            row_values = [str(val) if pd.notna(val) else "" for val in row]
            lines.append(" | ".join(row_values))
        
        return "\n".join(lines)
    
    def extract_values(self, table_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        테이블에서 값들만 추출 (구조 정보 없이)
        
        Returns:
            [{'row': i, 'col': j, 'value': value}, ...]
        """
        values = []
        for i in range(len(table_data)):
            for j, col in enumerate(table_data.columns):
                value = table_data.iloc[i, j]
                if pd.notna(value):
                    values.append({
                        'row': i,
                        'col': j,
                        'column_name': col,
                        'value': value
                    })
        return values

