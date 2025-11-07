"""
PubTables-1M 데이터셋 로더
JSON 형식의 테이블 데이터를 DataFrame으로 변환
"""

import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np


class PubTables1MLoader:
    """PubTables-1M 데이터셋 로더"""
    
    def __init__(self, data_dir: str = "data/pubtables1m"):
        self.data_dir = Path(data_dir)
    
    def parse_json_to_table(self, json_data: Dict) -> Optional[pd.DataFrame]:
        """
        PubTables-1M JSON 데이터를 DataFrame으로 변환
        
        JSON 구조:
        - html: HTML 테이블
        - cells: 셀 정보 리스트
          - row: 행 번호
          - column: 열 번호
          - text: 셀 텍스트
        
        Args:
            json_data: JSON 딕셔너리
        
        Returns:
            DataFrame 또는 None
        """
        try:
            # cells 정보 추출
            cells = json_data.get('cells', [])
            
            if not cells:
                # HTML에서 파싱 시도
                html = json_data.get('html', '')
                if html:
                    try:
                        # pandas의 read_html 사용
                        dfs = pd.read_html(html)
                        if dfs:
                            return dfs[0]
                    except:
                        pass
                return None
            
            # 최대 행/열 찾기
            max_row = max(c.get('row', 0) for c in cells)
            max_col = max(c.get('column', 0) for c in cells)
            
            # 2D 배열 생성
            table_data = [['' for _ in range(max_col + 1)] for _ in range(max_row + 1)]
            
            # 셀 데이터 채우기
            for cell in cells:
                row = cell.get('row', 0)
                col = cell.get('column', 0)
                text = str(cell.get('text', ''))
                
                if row <= max_row and col <= max_col:
                    table_data[row][col] = text
            
            # DataFrame 생성
            df = pd.DataFrame(table_data)
            
            # 빈 행/열 제거
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            return df if not df.empty else None
            
        except Exception as e:
            print(f"JSON 파싱 오류: {e}")
            return None
    
    def load_pubtables1m(self,
                        dataset_dir: Optional[str] = None,
                        max_tables: Optional[int] = None) -> List[pd.DataFrame]:
        """
        PubTables-1M 데이터셋 로드
        
        Args:
            dataset_dir: 데이터셋 디렉토리 (None이면 data/pubtables1m)
            max_tables: 최대 로드할 테이블 수
        
        Returns:
            테이블 리스트
        """
        if dataset_dir is None:
            dataset_dir = self.data_dir
        else:
            dataset_dir = Path(dataset_dir)
        
        # 다양한 가능한 경로 확인
        possible_paths = [
            dataset_dir / "pubtables1m",
            dataset_dir / "data",
            dataset_dir / "train",
            dataset_dir / "test",
            dataset_dir,
        ]
        
        json_files = []
        for path in possible_paths:
            if path.exists():
                json_files.extend(list(path.glob("**/*.json")))
                # CSV나 HTML도 확인
                json_files.extend(list(path.glob("**/*.csv")))
                json_files.extend(list(path.glob("**/*.html")))
        
        if not json_files:
            print(f"경고: PubTables-1M 데이터 파일을 찾을 수 없습니다: {dataset_dir}")
            print("다운로드 가이드를 확인하세요: data/pubtables1m/DOWNLOAD_GUIDE.md")
            print("\n참고: PubTables-1M은 Hugging Face에서도 다운로드 가능합니다:")
            print("  https://huggingface.co/datasets/bsmock/pubtables-1m")
            return []
        
        if max_tables:
            json_files = json_files[:max_tables]
        
        print(f"PubTables-1M에서 {len(json_files)}개 파일 로드 중...")
        
        tables = []
        for file_path in json_files:
            try:
                if file_path.suffix == '.json':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    table = self.parse_json_to_table(json_data)
                    if table is not None:
                        tables.append(table)
                        
                elif file_path.suffix == '.csv':
                    table = pd.read_csv(file_path)
                    if not table.empty:
                        tables.append(table)
                        
                elif file_path.suffix == '.html':
                    dfs = pd.read_html(file_path)
                    if dfs:
                        tables.append(dfs[0])
                        
            except Exception as e:
                print(f"경고: 파일 로드 실패 ({file_path.name}): {e}")
                continue
        
        print(f"✓ {len(tables)}개 테이블 로드 완료")
        return tables




