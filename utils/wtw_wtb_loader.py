"""
WTW-Dataset 로더
XML 형식의 테이블 데이터를 DataFrame으로 변환
"""

import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np


class WTWLoader:
    """WTW-Dataset 로더"""
    
    def __init__(self, data_dir: str = "data/wtw"):
        self.data_dir = Path(data_dir)
    
    def parse_xml_to_table(self, xml_path: Path) -> Optional[pd.DataFrame]:
        """
        WTW XML 파일을 DataFrame으로 변환
        
        XML 구조:
        - image: 이미지 정보
        - table: 테이블 정보
          - cell: 셀 정보 (bbox, start_col, start_row, end_col, end_row)
        
        Args:
            xml_path: XML 파일 경로
        
        Returns:
            DataFrame 또는 None
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 모든 테이블 찾기
            tables = root.findall('.//table')
            
            if not tables:
                return None
            
            # 첫 번째 테이블 사용 (여러 개면 첫 번째만)
            table_elem = tables[0]
            
            # 셀 정보 수집
            cells = []
            for cell in table_elem.findall('.//cell'):
                start_col = int(cell.get('start_col', 0))
                start_row = int(cell.get('start_row', 0))
                end_col = int(cell.get('end_col', start_col))
                end_row = int(cell.get('end_row', start_row))
                
                # 텍스트 추출 (있는 경우)
                text = cell.text if cell.text else ''
                
                cells.append({
                    'row': start_row,
                    'col': start_col,
                    'end_row': end_row,
                    'end_col': end_col,
                    'text': text.strip()
                })
            
            if not cells:
                return None
            
            # 최대 행/열 찾기
            max_row = max(c['end_row'] for c in cells)
            max_col = max(c['end_col'] for c in cells)
            
            # 2D 배열 생성
            table_data = [['' for _ in range(max_col + 1)] for _ in range(max_row + 1)]
            
            # 셀 데이터 채우기
            for cell in cells:
                for r in range(cell['row'], cell['end_row'] + 1):
                    for c in range(cell['col'], cell['end_col'] + 1):
                        if r <= max_row and c <= max_col:
                            if not table_data[r][c]:  # 빈 셀만 채움
                                table_data[r][c] = cell['text']
            
            # DataFrame 생성
            df = pd.DataFrame(table_data)
            
            # 빈 행/열 제거
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            return df if not df.empty else None
            
        except Exception as e:
            print(f"XML 파싱 오류 ({xml_path.name}): {e}")
            return None
    
    def load_wtw(self, 
                 dataset_dir: Optional[str] = None,
                 split: str = 'train',
                 max_tables: Optional[int] = None) -> List[pd.DataFrame]:
        """
        WTW 데이터셋 로드
        
        Args:
            dataset_dir: 데이터셋 디렉토리 (None이면 data/wtw)
            split: 'train' 또는 'test'
            max_tables: 최대 로드할 테이블 수
        
        Returns:
            테이블 리스트
        """
        if dataset_dir is None:
            dataset_dir = self.data_dir
        else:
            dataset_dir = Path(dataset_dir)
        
        xml_dir = dataset_dir / "data" / split / "xml"
        
        if not xml_dir.exists():
            print(f"경고: WTW XML 디렉토리를 찾을 수 없습니다: {xml_dir}")
            print("다운로드 가이드를 확인하세요: data/wtw/DOWNLOAD_GUIDE.md")
            return []
        
        xml_files = list(xml_dir.glob("*.xml"))
        
        if max_tables:
            xml_files = xml_files[:max_tables]
        
        print(f"WTW-Dataset ({split})에서 {len(xml_files)}개 XML 파일 로드 중...")
        
        tables = []
        for xml_file in xml_files:
            table = self.parse_xml_to_table(xml_file)
            if table is not None:
                tables.append(table)
        
        print(f"✓ {len(tables)}개 테이블 로드 완료")
        return tables

