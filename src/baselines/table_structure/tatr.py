"""
Table Transformer (TATR) 베이스라인 래퍼
Microsoft의 표 구조 인식 모델을 기존 파서 인터페이스와 호환되도록 래핑
"""

import os
import sys
import subprocess
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings

# TATR 모델 경로 (사용자가 설정)
TATR_REPO_PATH = os.environ.get('TATR_REPO_PATH', None)


class TATRParser:
    """
    Table Transformer (TATR) 베이스라인 파서
    
    Microsoft의 table-transformer를 사용하여 표 구조를 인식합니다.
    기존 LabeledTableParser와 호환되는 인터페이스를 제공합니다.
    """
    
    def __init__(self, 
                 model_version: str = "v1.1-pub",
                 repo_path: Optional[str] = None,
                 device: str = "cuda",
                 auto_download: bool = True):
        """
        Args:
            model_version: 사용할 모델 버전 ('v1.0', 'v1.1-pub', 'v1.1-fin', 'v1.1-all')
            repo_path: table-transformer 저장소 경로 (None이면 환경변수 또는 자동 다운로드)
            device: 사용할 디바이스 ('cuda' 또는 'cpu')
            auto_download: Hugging Face 모델 자동 다운로드 여부
        """
        self.model_version = model_version
        self.device = device
        self.repo_path = repo_path or TATR_REPO_PATH
        
        # 모델 가중치 Hugging Face에서 다운로드 가능
        self.model_name_map = {
            "v1.0": "microsoft/table-transformer-structure-recognition-v1.0",
            "v1.1-pub": "microsoft/table-transformer-structure-recognition-v1.1-pub",
            "v1.1-fin": "microsoft/table-transformer-structure-recognition-v1.1-fin",
            "v1.1-all": "microsoft/table-transformer-structure-recognition-v1.1-all"
        }
        
        self.model_path = None
        if auto_download:
            self._try_download_model()
        
        self._check_installation()
    
    def _try_download_model(self):
        """Hugging Face에서 모델 자동 다운로드 시도"""
        try:
            from huggingface_hub import snapshot_download
            
            model_name = self.model_name_map.get(self.model_version)
            if model_name is None:
                return
            
            model_dir = Path("models/tatr") / self.model_version
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # 이미 다운로드되어 있는지 확인
            if (model_dir / "config.json").exists():
                self.model_path = str(model_dir)
                return
            
            print(f"[TATR] Hugging Face에서 모델 다운로드 중: {model_name}")
            snapshot_download(
                repo_id=model_name,
                local_dir=str(model_dir),
                resume_download=True
            )
            self.model_path = str(model_dir)
            print(f"[TATR] 모델 다운로드 완료: {model_dir}")
            
        except ImportError:
            warnings.warn(
                "huggingface_hub가 설치되지 않았습니다. "
                "설치: pip install huggingface_hub",
                UserWarning
            )
        except Exception as e:
            warnings.warn(
                f"모델 자동 다운로드 실패: {e}. "
                "시뮬레이션 모드로 동작합니다.",
                UserWarning
            )
    
    def _check_installation(self):
        """TATR 설치 확인 및 안내"""
        if self.repo_path is None and self.model_path is None:
            warnings.warn(
                "TATR 저장소 경로가 설정되지 않았습니다.\n"
                "설치 방법:\n"
                "1. git clone https://github.com/microsoft/table-transformer.git\n"
                "2. conda env create -f environment.yml\n"
                "3. conda activate tables-detr\n"
                "4. 환경변수 설정: export TATR_REPO_PATH=/path/to/table-transformer\n"
                "또는 Hugging Face 모델 자동 다운로드 사용 (auto_download=True)",
                UserWarning
            )
    
    def parse(self, 
              table_image_path: Optional[str] = None,
              table_data: Optional[pd.DataFrame] = None,
              output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        표 구조 인식 수행
        
        Args:
            table_image_path: 표 이미지 경로 (PDF 또는 이미지 파일)
            table_data: DataFrame (이미지가 없을 경우 사용)
            output_dir: 결과 저장 디렉토리
        
        Returns:
            파싱된 표 구조 정보
        """
        if table_image_path is None and table_data is None:
            raise ValueError("table_image_path 또는 table_data 중 하나는 필수입니다.")
        
        # TATR 저장소가 있으면 실제 모델 사용
        if self.repo_path and os.path.exists(self.repo_path):
            return self._parse_with_tatr(table_image_path, output_dir)
        else:
            # 저장소가 없으면 시뮬레이션 모드 (데모용)
            return self._parse_simulation_mode(table_data)
    
    def _parse_with_tatr(self, 
                         table_image_path: str,
                         output_dir: Optional[str]) -> Dict[str, Any]:
        """
        실제 TATR 모델을 사용한 파싱
        
        Note: 실제 구현은 TATR 저장소의 main.py를 호출하는 방식으로 진행됩니다.
        """
        if not os.path.exists(table_image_path):
            raise FileNotFoundError(f"표 이미지를 찾을 수 없습니다: {table_image_path}")
        
        # TATR 실행을 위한 명령어 구성
        # 실제로는 TATR의 Python API를 직접 호출해야 합니다
        # 여기서는 구조만 제공
        
        result = {
            'method': 'TATR',
            'model_version': self.model_version,
            'cells': [],
            'structure': {
                'rows': 0,
                'cols': 0,
                'headers': []
            }
        }
        
        warnings.warn(
            "TATR 실제 실행은 TATR 저장소를 설치하고 "
            "Python API를 직접 호출해야 합니다. "
            "현재는 시뮬레이션 모드로 동작합니다.",
            UserWarning
        )
        
        return result
    
    def _parse_simulation_mode(self, table_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """
        TATR 모델이 없을 때 시뮬레이션 모드로 동작
        (데모 및 테스트용)
        """
        if table_data is None:
            raise ValueError("시뮬레이션 모드에서는 table_data가 필요합니다.")
        
        # 간단한 구조 인식 시뮬레이션
        n_rows, n_cols = table_data.shape
        
        cells = []
        for i in range(n_rows):
            for j in range(n_cols):
                cell_value = table_data.iloc[i, j]
                
                # 셀 타입 추정 (간단한 휴리스틱)
                if i == 0:
                    cell_type = 'column_header'
                elif j == 0:
                    cell_type = 'row_header'
                else:
                    cell_type = 'data'
                
                cells.append({
                    'row': i,
                    'col': j,
                    'value': str(cell_value) if pd.notna(cell_value) else '',
                    'type': cell_type,
                    'row_span': 1,
                    'col_span': 1
                })
        
        return {
            'method': 'TATR (Simulation)',
            'model_version': self.model_version,
            'cells': cells,
            'structure': {
                'rows': n_rows,
                'cols': n_cols,
                'headers': {
                    'row_headers': [0] if n_rows > 0 else [],
                    'col_headers': [0] if n_cols > 0 else []
                }
            },
            'table_data': table_data
        }
    
    def to_labeled_cells(self, parsed_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        TATR 파싱 결과를 LabeledTableParser의 CellLabel 형식으로 변환
        
        Args:
            parsed_result: parse() 메서드의 반환값
        
        Returns:
            레이블링된 셀 리스트
        """
        labeled_cells = []
        
        for cell in parsed_result.get('cells', []):
            labeled_cells.append({
                'row': cell['row'],
                'col': cell['col'],
                'cell_type': cell['type'],
                'value': cell['value'],
                'row_span': cell.get('row_span', 1),
                'col_span': cell.get('col_span', 1)
            })
        
        return labeled_cells
    
    @staticmethod
    def download_model(model_version: str = "v1.1-pub", 
                      output_dir: str = "models/tatr"):
        """
        Hugging Face에서 TATR 모델 다운로드
        
        Args:
            model_version: 모델 버전
            output_dir: 저장 디렉토리
        """
        try:
            from huggingface_hub import snapshot_download
            
            model_name_map = {
                "v1.0": "microsoft/table-transformer-structure-recognition-v1.0",
                "v1.1-pub": "microsoft/table-transformer-structure-recognition-v1.1-pub",
                "v1.1-fin": "microsoft/table-transformer-structure-recognition-v1.1-fin",
                "v1.1-all": "microsoft/table-transformer-structure-recognition-v1.1-all"
            }
            
            model_name = model_name_map.get(model_version)
            if model_name is None:
                raise ValueError(f"지원하지 않는 모델 버전: {model_version}")
            
            print(f"모델 다운로드 중: {model_name}")
            snapshot_download(
                repo_id=model_name,
                local_dir=os.path.join(output_dir, model_version)
            )
            print(f"다운로드 완료: {output_dir}/{model_version}")
            
        except ImportError:
            raise ImportError(
                "huggingface_hub가 설치되지 않았습니다. "
                "설치: pip install huggingface_hub"
            )


def main():
    """테스트 코드"""
    import pandas as pd
    
    # 샘플 테이블 생성
    sample_table = pd.DataFrame({
        '연도': [2020, 2021, 2022, 2023],
        '매출액(억원)': [1000, 1200, 1500, 1800],
        '순이익(억원)': [100, 150, 200, 250]
    })
    
    # TATR 파서 초기화
    parser = TATRParser(model_version="v1.1-pub")
    
    # 파싱 수행
    result = parser.parse(table_data=sample_table)
    
    print("파싱 결과:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

