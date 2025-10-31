"""
데이터셋 다운로드 유틸리티
"""

import os
import requests
import pandas as pd
from typing import List, Optional
from pathlib import Path
import zipfile
import json


class DatasetDownloader:
    """데이터셋 다운로더"""
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_public_data(self, 
                            dataset_url: str,
                            save_path: Optional[str] = None) -> str:
        """
        공공데이터포털에서 데이터 다운로드
        
        Args:
            dataset_url: 데이터셋 URL
            save_path: 저장 경로 (None이면 자동 생성)
        
        Returns:
            저장된 파일 경로
        """
        if save_path is None:
            filename = dataset_url.split('/')[-1]
            save_path = self.output_dir / filename
        
        print(f"다운로드 중: {dataset_url}")
        response = requests.get(dataset_url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"저장 완료: {save_path}")
        return str(save_path)
    
    def create_sample_tables(self, num_tables: int = 10) -> List[pd.DataFrame]:
        """
        샘플 테이블 생성 (테스트용)
        
        Args:
            num_tables: 생성할 테이블 수
        
        Returns:
            생성된 테이블 리스트
        """
        import numpy as np
        from datetime import datetime
        
        tables = []
        sample_dir = self.output_dir / "sample_tables"
        sample_dir.mkdir(exist_ok=True)
        
        for i in range(num_tables):
            # 다양한 구조의 샘플 테이블 생성
            if i % 3 == 0:
                # 단순 표
                table = self._create_simple_table(i)
            elif i % 3 == 1:
                # 중첩 헤더 표
                table = self._create_nested_header_table(i)
            else:
                # 병합 셀 표
                table = self._create_merged_cell_table(i)
            
            tables.append(table)
            
            # Excel 파일로 저장
            excel_path = sample_dir / f"table_{i}.xlsx"
            table.to_excel(excel_path, index=False)
        
        print(f"샘플 테이블 {num_tables}개 생성 완료: {sample_dir}")
        return tables
    
    def _create_simple_table(self, idx: int) -> pd.DataFrame:
        """단순 표 생성"""
        data = {
            '연도': [2020, 2021, 2022, 2023],
            '매출액(억원)': [1000, 1200, 1500, 1800],
            '순이익(억원)': [100, 150, 200, 250],
            '직원수': [500, 550, 600, 650]
        }
        return pd.DataFrame(data)
    
    def _create_nested_header_table(self, idx: int) -> pd.DataFrame:
        """중첩 헤더 표 생성"""
        # MultiIndex로 중첩 헤더 생성
        columns = pd.MultiIndex.from_tuples([
            ('매출', '국내'),
            ('매출', '해외'),
            ('비용', '인건비'),
            ('비용', '운영비')
        ])
        
        data = {
            ('매출', '국내'): [800, 900, 1000, 1100],
            ('매출', '해외'): [200, 300, 500, 700],
            ('비용', '인건비'): [300, 350, 400, 450],
            ('비용', '운영비'): [200, 250, 300, 350]
        }
        
        df = pd.DataFrame(data, index=[2020, 2021, 2022, 2023])
        df.index.name = '연도'
        return df.reset_index()
    
    def _create_merged_cell_table(self, idx: int) -> pd.DataFrame:
        """병합 셀 있는 표 생성 (Excel에서 병합되도록 설계)"""
        # 실제 병합은 Excel에서 처리, 여기서는 구조만 생성
        data = {
            '부서': ['영업부', '영업부', '개발부', '개발부', '마케팅부'],
            '직급': ['과장', '대리', '팀장', '선임', '과장'],
            '이름': ['홍길동', '김철수', '이영희', '박민수', '최지영'],
            '연봉(만원)': [5000, 4000, 7000, 5500, 5000],
            '입사일': ['2018-01-01', '2019-03-15', '2015-06-01', '2017-09-01', '2018-11-01']
        }
        return pd.DataFrame(data)
    
    def save_metadata(self, tables_info: List[Dict], filename: str = "metadata.json"):
        """테이블 메타데이터 저장"""
        metadata_path = self.output_dir / filename
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(tables_info, f, ensure_ascii=False, indent=2)
        print(f"메타데이터 저장: {metadata_path}")


def main():
    """예제 실행"""
    downloader = DatasetDownloader()
    
    # 샘플 테이블 생성
    print("샘플 테이블 생성 중...")
    tables = downloader.create_sample_tables(num_tables=10)
    
    # 메타데이터 생성
    tables_info = [
        {
            'table_id': f'table_{i}',
            'filename': f'table_{i}.xlsx',
            'shape': list(table.shape),
            'columns': list(table.columns.tolist())
        }
        for i, table in enumerate(tables)
    ]
    downloader.save_metadata(tables_info)
    
    print("\n사용 가능한 공개 데이터셋:")
    print("1. 공공데이터포털: https://www.data.go.kr")
    print("2. DART: https://dart.fss.or.kr")
    print("3. KOSIS: https://kosis.kr")


if __name__ == "__main__":
    main()

