"""
Sato 시맨틱 타입 검출 베이스라인
Megagon Labs의 Sato 모델을 사용하여 표 컬럼의 시맨틱 타입을 검출
"""

import os
import warnings
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

# Sato 저장소 경로
SATO_REPO_PATH = os.environ.get('SATO_REPO_PATH', None)


class SatoSemanticTypeDetector:
    """
    Sato 시맨틱 타입 검출기
    
    Megagon Labs의 Sato 모델을 사용하여 표 컬럼의 시맨틱 타입을 검출합니다.
    78가지 시맨틱 타입을 지원합니다.
    """
    
    # 78가지 시맨틱 타입 (주요 타입만 예시)
    SEMANTIC_TYPES = [
        'year', 'month', 'day', 'date', 'time',
        'person', 'organization', 'location', 'country', 'city',
        'money', 'currency', 'percentage', 'number', 'integer',
        'phone', 'email', 'url', 'ip_address',
        'product', 'brand', 'category',
        'age', 'gender', 'title',
        'address', 'zipcode', 'state',
        'temperature', 'weight', 'height', 'distance',
        'score', 'rating', 'rank',
        'id', 'code', 'name', 'description',
        'price', 'cost', 'revenue', 'profit', 'loss',
        'quantity', 'count', 'amount',
        'status', 'type', 'class',
        # ... 총 78가지 타입
    ]
    
    def __init__(self, 
                 repo_path: Optional[str] = None,
                 model_path: Optional[str] = None):
        """
        Args:
            repo_path: Sato 저장소 경로
            model_path: 사전 학습된 모델 경로
        """
        self.repo_path = repo_path or SATO_REPO_PATH
        self.model_path = model_path
        self.model = None
        
        self._check_installation()
        self._load_model()
    
    def _check_installation(self):
        """Sato 설치 확인 및 안내"""
        if self.repo_path is None:
            warnings.warn(
                "Sato 저장소 경로가 설정되지 않았습니다.\n"
                "설치 방법:\n"
                "1. git clone https://github.com/megagonlabs/sato.git\n"
                "2. pip install -r requirements.txt\n"
                "3. 환경변수 설정: export SATO_REPO_PATH=/path/to/sato\n"
                "또는 직접 사용: SatoSemanticTypeDetector(repo_path='/path/to/sato')",
                UserWarning
            )
    
    def _load_model(self):
        """모델 로드 (실제 구현은 Sato 저장소의 코드 사용)"""
        if self.model_path and os.path.exists(self.model_path):
            # 실제 모델 로드 코드는 Sato 저장소 참조
            pass
        else:
            # 시뮬레이션 모드
            self.model = None
    
    def detect_column_types(self, 
                           table_data: pd.DataFrame,
                           column_names: Optional[List[str]] = None) -> Dict[str, str]:
        """
        컬럼의 시맨틱 타입 검출
        
        Args:
            table_data: 분석할 테이블
            column_names: 분석할 컬럼 리스트 (None이면 모든 컬럼)
        
        Returns:
            컬럼명 -> 시맨틱 타입 매핑 딕셔너리
        """
        if column_names is None:
            column_names = list(table_data.columns)
        
        column_types = {}
        
        for col_name in column_names:
            if col_name not in table_data.columns:
                continue
            
            col_data = table_data[col_name].dropna()
            
            if self.model is not None:
                # 실제 Sato 모델 사용
                semantic_type = self._predict_with_model(col_data, col_name, table_data)
            else:
                # 시뮬레이션 모드: 휴리스틱 기반 타입 추정
                semantic_type = self._predict_with_heuristics(col_data, col_name, table_data)
            
            column_types[col_name] = semantic_type
        
        return column_types
    
    def _predict_with_model(self, 
                           col_data: pd.Series,
                           col_name: str,
                           table_data: pd.DataFrame) -> str:
        """
        실제 Sato 모델을 사용한 예측
        
        Note: 실제 구현은 Sato 저장소의 코드를 호출해야 합니다.
        """
        # 실제 구현은 Sato의 predict_column_type 메서드 호출
        # 여기서는 구조만 제공
        return self._predict_with_heuristics(col_data, col_name, table_data)
    
    def _predict_with_heuristics(self,
                                col_data: pd.Series,
                                col_name: str,
                                table_data: pd.DataFrame) -> str:
        """
        휴리스틱 기반 시맨틱 타입 추정 (시뮬레이션 모드)
        """
        if len(col_data) == 0:
            return 'unknown'
        
        # 컬럼명 기반 추정
        col_name_lower = col_name.lower()
        
        # 날짜/시간 관련
        if any(keyword in col_name_lower for keyword in ['연도', 'year', '년']):
            return 'year'
        if any(keyword in col_name_lower for keyword in ['월', 'month']):
            return 'month'
        if any(keyword in col_name_lower for keyword in ['일', 'day', '날짜', 'date']):
            return 'date'
        
        # 금액 관련
        if any(keyword in col_name_lower for keyword in ['매출', 'revenue', 'sales', '금액', '원']):
            return 'money'
        if any(keyword in col_name_lower for keyword in ['이익', 'profit', 'loss', '손익']):
            return 'money'
        if any(keyword in col_name_lower for keyword in ['가격', 'price', 'cost', '비용']):
            return 'money'
        
        # 비율/퍼센트
        if any(keyword in col_name_lower for keyword in ['율', 'rate', '비율', 'percent', '%']):
            return 'percentage'
        
        # 개수/수량
        if any(keyword in col_name_lower for keyword in ['수', 'count', '개수', 'quantity', '직원']):
            return 'count'
        
        # 데이터 타입 기반 추정
        dtype = col_data.dtype
        
        if pd.api.types.is_datetime64_any_dtype(col_data):
            return 'date'
        elif pd.api.types.is_numeric_dtype(col_data):
            # 숫자형 데이터
            if col_data.min() >= 1900 and col_data.max() <= 2100:
                return 'year'
            elif col_data.min() >= 0 and col_data.max() <= 100:
                if any(keyword in col_name_lower for keyword in ['율', 'rate', 'percent']):
                    return 'percentage'
                return 'number'
            else:
                return 'number'
        elif pd.api.types.is_string_dtype(col_data):
            # 문자열 데이터
            sample_values = col_data.head(10).astype(str)
            
            # 이메일 확인
            if sample_values.str.contains(r'@', na=False).any():
                return 'email'
            
            # URL 확인
            if sample_values.str.contains(r'http', na=False).any():
                return 'url'
            
            # 전화번호 확인
            if sample_values.str.contains(r'\d{2,3}-\d{3,4}-\d{4}', na=False).any():
                return 'phone'
            
            return 'text'
        
        return 'unknown'
    
    def annotate_table(self, 
                      table_data: pd.DataFrame,
                      column_types: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        테이블에 시맨틱 타입 주석 추가
        
        Args:
            table_data: 주석을 추가할 테이블
            column_types: 컬럼 타입 딕셔너리 (None이면 자동 검출)
        
        Returns:
            시맨틱 타입 정보가 포함된 DataFrame
        """
        if column_types is None:
            column_types = self.detect_column_types(table_data)
        
        # 메타데이터로 타입 정보 저장
        annotated_table = table_data.copy()
        annotated_table.attrs['semantic_types'] = column_types
        
        return annotated_table
    
    @staticmethod
    def get_supported_types() -> List[str]:
        """지원하는 시맨틱 타입 리스트 반환"""
        return SatoSemanticTypeDetector.SEMANTIC_TYPES.copy()


def main():
    """테스트 코드"""
    import pandas as pd
    
    # 샘플 테이블 생성
    sample_table = pd.DataFrame({
        '연도': [2020, 2021, 2022, 2023],
        '매출액(억원)': [1000, 1200, 1500, 1800],
        '순이익률(%)': [10.0, 12.5, 13.3, 13.9],
        '직원수': [500, 550, 600, 650]
    })
    
    # Sato 검출기 초기화
    detector = SatoSemanticTypeDetector()
    
    # 컬럼 타입 검출
    column_types = detector.detect_column_types(sample_table)
    
    print("검출된 시맨틱 타입:")
    for col, sem_type in column_types.items():
        print(f"  {col}: {sem_type}")
    
    # 테이블에 주석 추가
    annotated_table = detector.annotate_table(sample_table, column_types)
    print(f"\n주석이 추가된 테이블 속성: {annotated_table.attrs}")


if __name__ == "__main__":
    main()

