#!/usr/bin/env python
"""
PubTables-1M 다운로드 추천 방법
실험 목적에 가장 적합한 방법
"""
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.download_external_datasets import ExternalDatasetDownloader

def main():
    print("="*70)
    print("PubTables-1M 다운로드 추천 방법")
    print("="*70)
    print("\n실험 목적에 가장 적합한 방법: 샘플만 다운로드")
    print("이유:")
    print("  - 빠름 (수분 내 완료)")
    print("  - 디스크 공간 절약 (수 GB vs 수십 GB)")
    print("  - 실험에 충분한 데이터 (1000개 샘플)")
    print("  - 프로젝트에서 주로 샘플만 사용 (max_tables=100-1000)")
    print("\n" + "="*70)
    
    downloader = ExternalDatasetDownloader()
    
    # 샘플만 다운로드
    print("\n[추천] 샘플 다운로드 시작 (1000개)...")
    try:
        pubtables_dir = downloader.download_pubtables1m_hf(
            num_samples=1000,
            use_git_lfs=False
        )
        print(f"\n✅ 완료! 다운로드 위치: {pubtables_dir}")
        print("\n사용 방법:")
        print("```python")
        print("from utils.multi_dataset_loader import MultiDatasetLoader")
        print("loader = MultiDatasetLoader()")
        print("tables = loader.load_pubtables1m(max_tables=1000)")
        print("```")
    except Exception as e:
        print(f"\n❌ 다운로드 실패: {e}")
        print("\n대안: 기존 다운로드된 파일 사용")
        existing_dir = Path("data/pubtables1m/pubtables-1m")
        if existing_dir.exists():
            print(f"기존 디렉토리 확인: {existing_dir}")
            print("tar.gz 파일을 압축 해제하여 사용할 수 있습니다.")

if __name__ == "__main__":
    main()

