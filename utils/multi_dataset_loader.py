"""
다중 데이터셋 로더
PubTables-1M, TabRecSet, KorWikiTabular, WTW 등 다양한 데이터셋 지원
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import os

# WTW와 PubTables-1M 로더 import
try:
    from utils.wtw_wtb_loader import WTWLoader
    WTW_AVAILABLE = True
except ImportError:
    WTW_AVAILABLE = False

try:
    from utils.pubtables1m_loader import PubTables1MLoader
    PUBTABLES1M_AVAILABLE = True
except ImportError:
    PUBTABLES1M_AVAILABLE = False

try:
    from utils.synthtabnet_loader import SynthTabNetLoader
    SYNTHTABNET_AVAILABLE = True
except ImportError:
    SYNTHTABNET_AVAILABLE = False


class MultiDatasetLoader:
    """다중 데이터셋 로더"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
    
    def load_pubtables1m(self, 
                        dataset_dir: Optional[str] = None,
                        max_tables: Optional[int] = None) -> List[pd.DataFrame]:
        """
        PubTables-1M 데이터셋 로드
        
        Args:
            dataset_dir: 데이터셋 디렉토리 (None이면 자동 탐색)
            max_tables: 최대 로드할 테이블 수 (None이면 전체)
        
        Returns:
            테이블 리스트
        """
        if dataset_dir is None:
            dataset_dir = self.data_dir / "pubtables1m"
        else:
            dataset_dir = Path(dataset_dir)
        
        # 전용 로더 사용 (있는 경우)
        if PUBTABLES1M_AVAILABLE:
            try:
                loader = PubTables1MLoader(str(dataset_dir))
                return loader.load_pubtables1m(dataset_dir=dataset_dir, max_tables=max_tables)
            except Exception as e:
                print(f"전용 로더 사용 실패, 기본 로더로 폴백: {e}")
        
        # 기본 로더 (CSV/XLSX 파일)
        if not dataset_dir.exists():
            print(f"경고: PubTables-1M 디렉토리를 찾을 수 없습니다: {dataset_dir}")
            print("다운로드 가이드를 확인하세요: data/pubtables1m/DOWNLOAD_GUIDE.md")
            return []
        
        tables = []
        table_files = list(dataset_dir.glob("**/*.csv")) + list(dataset_dir.glob("**/*.xlsx"))
        
        if max_tables:
            table_files = table_files[:max_tables]
        
        print(f"PubTables-1M에서 {len(table_files)}개 테이블 로드 중...")
        for file_path in table_files:
            try:
                if file_path.suffix == '.csv':
                    table = pd.read_csv(file_path)
                elif file_path.suffix == '.xlsx':
                    table = pd.read_excel(file_path)
                else:
                    continue
                
                if not table.empty:
                    tables.append(table)
            except Exception as e:
                print(f"경고: 테이블 로드 실패 ({file_path.name}): {e}")
                continue
        
        print(f"✓ {len(tables)}개 테이블 로드 완료")
        return tables
    
    def load_wtw(self,
                 dataset_dir: Optional[str] = None,
                 split: str = 'train',
                 max_tables: Optional[int] = None) -> List[pd.DataFrame]:
        """
        WTW-Dataset 로드
        
        Args:
            dataset_dir: 데이터셋 디렉토리 (None이면 data/wtw)
            split: 'train' 또는 'test'
            max_tables: 최대 로드할 테이블 수
        
        Returns:
            테이블 리스트
        """
        if dataset_dir is None:
            dataset_dir = self.data_dir / "wtw"
        else:
            dataset_dir = Path(dataset_dir)
        
        if WTW_AVAILABLE:
            try:
                loader = WTWLoader(str(dataset_dir))
                return loader.load_wtw(dataset_dir=dataset_dir, split=split, max_tables=max_tables)
            except Exception as e:
                print(f"WTW 로더 사용 실패: {e}")
                return []
        else:
            print("경고: WTW 로더를 사용할 수 없습니다.")
            return []
    
    def load_tabrecset(self,
                      dataset_dir: Optional[str] = None,
                      max_tables: Optional[int] = None) -> List[pd.DataFrame]:
        """
        TabRecSet 데이터셋 로드
        
        Args:
            dataset_dir: 데이터셋 디렉토리
            max_tables: 최대 로드할 테이블 수
        
        Returns:
            테이블 리스트
        """
        if dataset_dir is None:
            dataset_dir = self.data_dir / "tabrecset"
        else:
            dataset_dir = Path(dataset_dir)
        
        if not dataset_dir.exists():
            print(f"경고: TabRecSet 디렉토리를 찾을 수 없습니다: {dataset_dir}")
            print("다운로드 가이드를 확인하세요: data/tabrecset/DOWNLOAD_GUIDE.md")
            print("다운로드 링크: https://figshare.com/articles/dataset/TabRecSet_A_Large_Scale_Dataset_for_End-to-end_Table_Recognition_in_the_Wild/20647788")
            return []
        
        tables = []
        table_files = list(dataset_dir.glob("**/*.csv")) + list(dataset_dir.glob("**/*.xlsx"))
        
        if max_tables:
            table_files = table_files[:max_tables]
        
        print(f"TabRecSet에서 {len(table_files)}개 테이블 로드 중...")
        for file_path in table_files:
            try:
                if file_path.suffix == '.csv':
                    table = pd.read_csv(file_path)
                elif file_path.suffix == '.xlsx':
                    table = pd.read_excel(file_path)
                else:
                    continue
                
                if not table.empty:
                    tables.append(table)
            except Exception as e:
                print(f"경고: 테이블 로드 실패 ({file_path.name}): {e}")
                continue
        
        print(f"✓ {len(tables)}개 테이블 로드 완료")
        return tables
    
    def load_korwiki_tabular(self,
                             dataset_dir: Optional[str] = None,
                             max_tables: Optional[int] = None) -> List[pd.DataFrame]:
        """
        KorWikiTabular 데이터셋 로드
        
        Args:
            dataset_dir: 데이터셋 디렉토리
            max_tables: 최대 로드할 테이블 수
        
        Returns:
            테이블 리스트
        """
        if dataset_dir is None:
            dataset_dir = self.data_dir / "korwiki_tabular"
        else:
            dataset_dir = Path(dataset_dir)
        
        if not dataset_dir.exists():
            print(f"경고: KorWikiTabular 디렉토리를 찾을 수 없습니다: {dataset_dir}")
            return []
        
        tables = []
        table_files = list(dataset_dir.glob("**/*.csv")) + list(dataset_dir.glob("**/*.xlsx"))
        
        if max_tables:
            table_files = table_files[:max_tables]
        
        print(f"KorWikiTabular에서 {len(table_files)}개 테이블 로드 중...")
        for file_path in table_files:
            try:
                if file_path.suffix == '.csv':
                    table = pd.read_csv(file_path)
                elif file_path.suffix == '.xlsx':
                    table = pd.read_excel(file_path)
                else:
                    continue
                
                if not table.empty:
                    tables.append(table)
            except Exception as e:
                print(f"경고: 테이블 로드 실패 ({file_path.name}): {e}")
                continue
        
        print(f"✓ {len(tables)}개 테이블 로드 완료")
        return tables
    
    def load_tablebank(self,
                      dataset_dir: Optional[str] = None,
                      max_tables: Optional[int] = None) -> List[pd.DataFrame]:
        """
        TableBank 데이터셋 로드
        
        Args:
            dataset_dir: 데이터셋 디렉토리
            max_tables: 최대 로드할 테이블 수
        
        Returns:
            테이블 리스트
        """
        if dataset_dir is None:
            dataset_dir = self.data_dir / "tablebank"
        else:
            dataset_dir = Path(dataset_dir)
        
        if not dataset_dir.exists():
            print(f"경고: TableBank 디렉토리를 찾을 수 없습니다: {dataset_dir}")
            print("다운로드 가이드를 확인하세요: data/tablebank/DOWNLOAD_GUIDE.md")
            return []
        
        tables = []
        # CSV, XLSX, XML 등 다양한 형식 지원
        table_files = (
            list(dataset_dir.glob("**/*.csv")) +
            list(dataset_dir.glob("**/*.xlsx")) +
            list(dataset_dir.glob("**/*.xls"))
        )
        
        if max_tables:
            table_files = table_files[:max_tables]
        
        print(f"TableBank에서 {len(table_files)}개 테이블 로드 중...")
        for file_path in table_files:
            try:
                if file_path.suffix == '.csv':
                    table = pd.read_csv(file_path)
                elif file_path.suffix in ['.xlsx', '.xls']:
                    table = pd.read_excel(file_path)
                else:
                    continue
                
                if not table.empty:
                    tables.append(table)
            except Exception as e:
                print(f"경고: 테이블 로드 실패 ({file_path.name}): {e}")
                continue
        
        print(f"✓ {len(tables)}개 테이블 로드 완료")
        return tables
    
    def load_synthtabnet(self,
                        dataset_dir: Optional[str] = None,
                        style: Optional[str] = None,
                        split: str = 'train',
                        max_tables: Optional[int] = None) -> List[pd.DataFrame]:
        """
        SynthTabNet 데이터셋 로드
        
        Args:
            dataset_dir: 데이터셋 디렉토리
            style: 스타일 ('fintabnet', 'marketing', 'pubtabnet', 'sparse', None이면 모두)
            split: 'train', 'test', 'val'
            max_tables: 최대 로드할 테이블 수
        
        Returns:
            테이블 리스트
        """
        if dataset_dir is None:
            dataset_dir = self.data_dir / "synthtabnet"
        else:
            dataset_dir = Path(dataset_dir)
        
        if SYNTHTABNET_AVAILABLE:
            try:
                loader = SynthTabNetLoader(str(dataset_dir))
                return loader.load_synthtabnet(
                    dataset_dir=dataset_dir,
                    style=style,
                    split=split,
                    max_tables=max_tables
                )
            except Exception as e:
                print(f"SynthTabNet 로더 사용 실패: {e}")
                return []
        else:
            print("경고: SynthTabNet 로더를 사용할 수 없습니다.")
            print("필요시: pip install beautifulsoup4")
            return []
    
    def load_tabrecset_maxkinny(self,
                                dataset_dir: Optional[str] = None,
                                max_tables: Optional[int] = None) -> List[pd.DataFrame]:
        """
        TabRecSet 데이터셋 로드 (MaxKinny GitHub 저장소)
        
        Args:
            dataset_dir: 데이터셋 디렉토리
            max_tables: 최대 로드할 테이블 수
        
        Returns:
            테이블 리스트
        """
        if dataset_dir is None:
            dataset_dir = self.data_dir / "tabrecset_maxkinny"
        else:
            dataset_dir = Path(dataset_dir)
        
        if not dataset_dir.exists():
            print(f"경고: TabRecSet (MaxKinny) 디렉토리를 찾을 수 없습니다: {dataset_dir}")
            print("다운로드 가이드를 확인하세요: data/tabrecset_maxkinny/DOWNLOAD_GUIDE.md")
            return []
        
        tables = []
        table_files = (
            list(dataset_dir.glob("**/*.csv")) +
            list(dataset_dir.glob("**/*.xlsx")) +
            list(dataset_dir.glob("**/*.xls"))
        )
        
        if max_tables:
            table_files = table_files[:max_tables]
        
        print(f"TabRecSet (MaxKinny)에서 {len(table_files)}개 테이블 로드 중...")
        for file_path in table_files:
            try:
                if file_path.suffix == '.csv':
                    table = pd.read_csv(file_path)
                elif file_path.suffix in ['.xlsx', '.xls']:
                    table = pd.read_excel(file_path)
                else:
                    continue
                
                if not table.empty:
                    tables.append(table)
            except Exception as e:
                print(f"경고: 테이블 로드 실패 ({file_path.name}): {e}")
                continue
        
        print(f"✓ {len(tables)}개 테이블 로드 완료")
        return tables
    
    def load_mixed_datasets(self,
                           datasets: List[str],
                           max_tables_per_dataset: Optional[int] = None) -> Tuple[List[pd.DataFrame], Dict[str, int]]:
        """
        여러 데이터셋을 혼합하여 로드
        
        Args:
            datasets: 데이터셋 이름 리스트 ('pubtables1m', 'tabrecset', 'korwiki_tabular', 'rag_eval_ko', 'tablebank', 'synthtabnet', 'tabrecset_maxkinny')
            max_tables_per_dataset: 데이터셋당 최대 테이블 수
        
        Returns:
            (테이블 리스트, 데이터셋별 개수 딕셔너리)
        """
        all_tables = []
        dataset_counts = {}
        
        print("="*70)
        print("다중 데이터셋 로드")
        print("="*70)
        
        for dataset_name in datasets:
            print(f"\n[{dataset_name}] 로드 중...")
            
            if dataset_name == 'pubtables1m':
                tables = self.load_pubtables1m(max_tables=max_tables_per_dataset)
            elif dataset_name == 'tabrecset':
                tables = self.load_tabrecset(max_tables=max_tables_per_dataset)
            elif dataset_name == 'wtw':
                # WTW는 train과 test 모두 로드
                train_tables = self.load_wtw(split='train', max_tables=max_tables_per_dataset)
                test_tables = self.load_wtw(split='test', max_tables=max_tables_per_dataset)
                tables = train_tables + test_tables
            elif dataset_name == 'korwiki_tabular':
                tables = self.load_korwiki_tabular(max_tables=max_tables_per_dataset)
            elif dataset_name == 'rag_eval_ko':
                # 기존 RAG-Evaluation-Dataset-KO 로드
                from experiments.run_experiments import ExperimentRunner
                runner = ExperimentRunner()
                tables = runner.load_test_data("", use_dataset=True)
            elif dataset_name == 'tablebank':
                tables = self.load_tablebank(max_tables=max_tables_per_dataset)
            elif dataset_name == 'synthtabnet':
                tables = self.load_synthtabnet(
                    split='train',
                    max_tables=max_tables_per_dataset
                )
            elif dataset_name == 'tabrecset_maxkinny':
                tables = self.load_tabrecset_maxkinny(max_tables=max_tables_per_dataset)
            else:
                print(f"경고: 알 수 없는 데이터셋: {dataset_name}")
                tables = []
            
            dataset_counts[dataset_name] = len(tables)
            all_tables.extend(tables)
        
        print("\n" + "="*70)
        print("로드 완료 요약")
        print("="*70)
        for name, count in dataset_counts.items():
            print(f"  {name}: {count}개")
        print(f"\n총 테이블 수: {len(all_tables)}개")
        
        return all_tables, dataset_counts


def main():
    """예제 실행"""
    loader = MultiDatasetLoader()
    
    print("="*70)
    print("다중 데이터셋 로더 예제")
    print("="*70)
    
    # 추천 조합 1: 초기 실험 (PubTables-1M 샘플)
    print("\n[추천 조합 1] 초기 실험용 (PubTables-1M 샘플)")
    tables1, counts1 = loader.load_mixed_datasets(
        datasets=['pubtables1m'],
        max_tables_per_dataset=100  # 샘플만 사용
    )
    
    # 추천 조합 2: 한국어 특화 실험
    print("\n[추천 조합 2] 한국어 특화 실험")
    tables2, counts2 = loader.load_mixed_datasets(
        datasets=['rag_eval_ko', 'korwiki_tabular'],
        max_tables_per_dataset=50
    )
    
    # 추천 조합 3: 극단 케이스 테스트
    print("\n[추천 조합 3] 극단 케이스 테스트")
    tables3, counts3 = loader.load_mixed_datasets(
        datasets=['tabrecset'],
        max_tables_per_dataset=100
    )
    
    # 추천 조합 4: 새로 추가된 데이터셋
    print("\n[추천 조합 4] 새로 추가된 데이터셋 테스트")
    tables4, counts4 = loader.load_mixed_datasets(
        datasets=['tablebank', 'synthtabnet', 'tabrecset_maxkinny'],
        max_tables_per_dataset=50
    )


if __name__ == "__main__":
    main()

