"""
SynthTabNet 데이터셋 로더
JSONL 파일에서 HTML 테이블을 추출하여 DataFrame으로 변환
"""

import pandas as pd
import json
from pathlib import Path
from typing import List, Optional
from bs4 import BeautifulSoup


class SynthTabNetLoader:
    """SynthTabNet 데이터셋 로더"""
    
    def __init__(self, data_dir: str = "data/synthtabnet"):
        self.data_dir = Path(data_dir)
    
    def html_to_dataframe(self, html_str: str) -> Optional[pd.DataFrame]:
        """
        HTML 문자열을 DataFrame으로 변환
        
        Args:
            html_str: HTML 테이블 문자열
        
        Returns:
            DataFrame 또는 None
        """
        try:
            soup = BeautifulSoup(html_str, 'html.parser')
            table = soup.find('table')
            
            if table is None:
                return None
            
            # 테이블을 DataFrame으로 변환
            rows = []
            for tr in table.find_all('tr'):
                row_data = []
                for td in tr.find_all(['td', 'th']):
                    # 셀 내용 가져오기
                    cell_text = td.get_text(strip=True)
                    # rowspan/colspan 처리 (간단한 버전)
                    row_data.append(cell_text)
                if row_data:
                    rows.append(row_data)
            
            if not rows:
                return None
            
            # 최대 컬럼 수 맞추기
            max_cols = max(len(row) for row in rows)
            rows = [row + [''] * (max_cols - len(row)) for row in rows]
            
            # 첫 번째 행을 헤더로 사용 (th가 있으면)
            if rows:
                header_row = rows[0]
                data_rows = rows[1:]
                
                # DataFrame 생성
                df = pd.DataFrame(data_rows, columns=header_row)
                return df
            
            return None
            
        except Exception as e:
            print(f"HTML 파싱 오류: {e}")
            return None
    
    def parse_jsonl_entry(self, json_obj: dict) -> Optional[pd.DataFrame]:
        """
        JSONL 항목에서 테이블 추출
        
        Args:
            json_obj: JSON 객체 (JSONL 라인)
        
        Returns:
            DataFrame 또는 None
        """
        try:
            # HTML 형식의 테이블 추출
            html_str = json_obj.get('html', '')
            if not html_str:
                return None
            
            df = self.html_to_dataframe(html_str)
            return df
            
        except Exception as e:
            print(f"JSONL 파싱 오류: {e}")
            return None
    
    def load_synthtabnet(self,
                        dataset_dir: Optional[str] = None,
                        style: Optional[str] = None,
                        split: str = 'train',
                        max_tables: Optional[int] = None) -> List[pd.DataFrame]:
        """
        SynthTabNet 데이터셋 로드
        
        Args:
            dataset_dir: 데이터셋 디렉토리 (None이면 self.data_dir 사용)
            style: 스타일 ('fintabnet', 'marketing', 'pubtabnet', 'sparse', None이면 모두)
            split: 'train', 'test', 'val'
            max_tables: 최대 로드할 테이블 수
        
        Returns:
            테이블 리스트
        """
        if dataset_dir is None:
            dataset_dir = self.data_dir
        else:
            dataset_dir = Path(dataset_dir)
        
        if not dataset_dir.exists():
            print(f"경고: SynthTabNet 디렉토리를 찾을 수 없습니다: {dataset_dir}")
            return []
        
        tables = []
        
        # 스타일 디렉토리 찾기
        if style:
            style_dirs = [dataset_dir / style]
        else:
            # 모든 스타일 디렉토리 검색
            possible_styles = ['fintabnet', 'marketing', 'pubtabnet', 'sparse']
            style_dirs = [dataset_dir / s for s in possible_styles if (dataset_dir / s).exists()]
        
        for style_dir in style_dirs:
            if not style_dir.exists():
                continue
            
            # JSONL 파일 찾기
            jsonl_path = style_dir / "synthetic_data.jsonl"
            if not jsonl_path.exists():
                print(f"경고: JSONL 파일을 찾을 수 없습니다: {jsonl_path}")
                continue
            
            print(f"SynthTabNet 로드 중: {style_dir.name} ({split})")
            
            # JSONL 파일 읽기
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                count = 0
                for line_num, line in enumerate(f):
                    if max_tables and count >= max_tables:
                        break
                    
                    try:
                        json_obj = json.loads(line.strip())
                        
                        # split 필터링
                        if json_obj.get('split') != split:
                            continue
                        
                        # 테이블 추출
                        df = self.parse_jsonl_entry(json_obj)
                        if df is not None and not df.empty:
                            tables.append(df)
                            count += 1
                            
                    except json.JSONDecodeError as e:
                        print(f"경고: JSON 파싱 오류 (라인 {line_num}): {e}")
                        continue
                    except Exception as e:
                        print(f"경고: 처리 오류 (라인 {line_num}): {e}")
                        continue
        
        print(f"✓ {len(tables)}개 테이블 로드 완료")
        return tables

