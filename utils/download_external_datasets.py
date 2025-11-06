"""
외부 데이터셋 다운로드 유틸리티
- TabRecSet (Figshare)
- WTW-Dataset (GitHub)
- PubTables-1M (Microsoft Research)
"""

import os
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import subprocess


class ExternalDatasetDownloader:
    """외부 데이터셋 다운로더"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url: str, output_path: Path, chunk_size: int = 8192):
        """대용량 파일 다운로드 (진행률 표시)"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=output_path.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    def download_tabrecset(self, output_dir: Optional[str] = None) -> Path:
        """
        TabRecSet 데이터셋 다운로드
        
        출처: https://figshare.com/articles/dataset/TabRecSet_A_Large_Scale_Dataset_for_End-to-end_Table_Recognition_in_the_Wild/20647788
        
        Args:
            output_dir: 출력 디렉토리 (None이면 data/tabrecset)
        
        Returns:
            다운로드된 데이터셋 디렉토리 경로
        """
        if output_dir is None:
            output_dir = self.data_dir / "tabrecset"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("TabRecSet 데이터셋 다운로드")
        print("="*70)
        print("출처: https://figshare.com/articles/dataset/TabRecSet_A_Large_Scale_Dataset_for_End-to-end_Table_Recognition_in_the_Wild/20647788")
        print("\n수동 다운로드 필요:")
        print("1. 위 링크 방문")
        print("2. 'Download all (5.28 GB)' 버튼 클릭")
        print("3. 다운로드한 파일을 다음 위치에 압축 해제:")
        print(f"   {output_dir}")
        print("\n또는 다음 명령어로 다운로드 (wget/curl 사용):")
        print("wget https://figshare.com/ndownloader/articles/20647788/versions/9 -O tabrecset.zip")
        print("="*70)
        
        # 다운로드 가이드 파일 생성
        guide_path = output_dir / "DOWNLOAD_GUIDE.md"
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write("""# TabRecSet 데이터셋 다운로드 가이드

## 출처
https://figshare.com/articles/dataset/TabRecSet_A_Large_Scale_Dataset_for_End-to-end_Table_Recognition_in_the_Wild/20647788

## 다운로드 방법

### 방법 1: 웹 브라우저에서 다운로드
1. 위 링크 방문
2. "Download all (5.28 GB)" 버튼 클릭
3. 다운로드한 파일을 이 디렉토리에 압축 해제

### 방법 2: 명령어로 다운로드
```bash
# wget 사용
wget https://figshare.com/ndownloader/articles/20647788/versions/9 -O tabrecset.zip

# 또는 curl 사용
curl -L https://figshare.com/ndownloader/articles/20647788/versions/9 -o tabrecset.zip

# 압축 해제
unzip tabrecset.zip -d .
```

## 데이터셋 특징
- 대규모 데이터셋 (5.28 GB)
- 실제 환경(인 와일드) 테이블 인식을 위한 데이터셋
- End-to-end 테이블 인식 태스크용

## 데이터 구조
다운로드 후 데이터 구조를 확인하여 로더를 조정하세요.
""")
        
        print(f"다운로드 가이드 저장: {guide_path}")
        return output_dir
    
    def download_wtw_dataset(self, output_dir: Optional[str] = None) -> Path:
        """
        WTW-Dataset 다운로드
        
        출처: https://github.com/wangwen-whu/WTW-Dataset
        
        Args:
            output_dir: 출력 디렉토리 (None이면 data/wtw)
        
        Returns:
            다운로드된 데이터셋 디렉토리 경로
        """
        if output_dir is None:
            output_dir = self.data_dir / "wtw"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("WTW-Dataset 다운로드")
        print("="*70)
        print("출처: https://github.com/wangwen-whu/WTW-Dataset")
        print("\nGitHub에서 클론:")
        print(f"git clone https://github.com/wangwen-whu/WTW-Dataset.git {output_dir}")
        print("\n또는 다운로드 링크에서 직접 다운로드:")
        print("README.md에 있는 다운로드 링크 확인")
        print("="*70)
        
        # GitHub 클론 시도
        try:
            print("\nGitHub에서 클론 시도 중...")
            result = subprocess.run(
                ['git', 'clone', 'https://github.com/wangwen-whu/WTW-Dataset.git', str(output_dir)],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                print("✓ GitHub 클론 완료")
            else:
                print(f"✗ GitHub 클론 실패: {result.stderr}")
                print("수동으로 클론하거나 다운로드 가이드를 참고하세요.")
        except Exception as e:
            print(f"✗ GitHub 클론 중 오류: {e}")
            print("수동으로 클론하거나 다운로드 가이드를 참고하세요.")
        
        # 다운로드 가이드 파일 생성
        guide_path = output_dir / "DOWNLOAD_GUIDE.md"
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write("""# WTW-Dataset 다운로드 가이드

## 출처
https://github.com/wangwen-whu/WTW-Dataset

## 다운로드 방법

### 방법 1: Git 클론
```bash
git clone https://github.com/wangwen-whu/WTW-Dataset.git data/wtw
```

### 방법 2: 직접 다운로드
1. GitHub 저장소 방문
2. README.md에서 다운로드 링크 확인
3. 다운로드한 파일을 이 디렉토리에 압축 해제

## 데이터셋 특징
- 실제 환경(인 와일드) 테이블 파싱 데이터셋
- 7가지 도전적인 케이스 포함:
  1. 기울어진 테이블 (Inclined tables)
  2. 곡선 테이블 (Curved tables)
  3. 가려지거나 흐릿한 테이블 (Occluded/blurred tables)
  4. 극단적 종횡비 테이블 (Extreme aspect ratio tables)
  5. 겹친 테이블 (Overlaid tables)
  6. 다중 색상 테이블 (Multi-color tables)
  7. 불규칙 테이블 (Irregular tables)

## 데이터 구조
- data/
  - train/
    - images/
    - xml/
  - test/
    - images/
    - xml/
    - class/ (7개 .txt 파일)

## 참고
ICCV 2021 논문: "Parsing Table Structures in the Wild"
""")
        
        print(f"다운로드 가이드 저장: {guide_path}")
        return output_dir
    
    def download_pubtables1m(self, output_dir: Optional[str] = None) -> Path:
        """
        PubTables-1M 데이터셋 다운로드
        
        출처: https://www.microsoft.com/en-us/research/publication/pubtables-1m/
        
        Args:
            output_dir: 출력 디렉토리 (None이면 data/pubtables1m)
        
        Returns:
            다운로드된 데이터셋 디렉토리 경로
        """
        if output_dir is None:
            output_dir = self.data_dir / "pubtables1m"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("PubTables-1M 데이터셋 다운로드")
        print("="*70)
        print("출처: https://www.microsoft.com/en-us/research/publication/pubtables-1m/")
        print("\n다운로드 방법:")
        print("1. Microsoft Research 페이지 방문")
        print("2. 'Related Tools' 섹션에서 'PubTables' 클릭")
        print("3. 또는 GitHub 저장소: https://github.com/microsoft/table-transformer")
        print("4. 데이터셋 다운로드 링크 확인")
        print("="*70)
        
        # GitHub 저장소 클론 시도
        try:
            print("\nGitHub 저장소에서 정보 확인 중...")
            github_dir = output_dir / "table-transformer"
            if not github_dir.exists():
                result = subprocess.run(
                    ['git', 'clone', 'https://github.com/microsoft/table-transformer.git', str(github_dir)],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    print("✓ GitHub 저장소 클론 완료")
                    print("  데이터셋 다운로드 스크립트를 확인하세요.")
                else:
                    print(f"✗ GitHub 클론 실패: {result.stderr}")
        except Exception as e:
            print(f"✗ GitHub 클론 중 오류: {e}")
        
        # 다운로드 가이드 파일 생성
        guide_path = output_dir / "DOWNLOAD_GUIDE.md"
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write("""# PubTables-1M 데이터셋 다운로드 가이드

## 출처
- Microsoft Research: https://www.microsoft.com/en-us/research/publication/pubtables-1m/
- GitHub: https://github.com/microsoft/table-transformer

## 다운로드 방법

### 방법 1: GitHub 저장소 사용
```bash
git clone https://github.com/microsoft/table-transformer.git
cd table-transformer
# 데이터셋 다운로드 스크립트 실행 (README 확인)
```

### 방법 2: 직접 다운로드
1. Microsoft Research 페이지 방문
2. "Related Tools" → "PubTables" 클릭
3. 데이터셋 다운로드 링크 확인
4. 다운로드한 파일을 이 디렉토리에 저장

## 데이터셋 특징
- 약 100만 개의 표
- 과학 논문에서 추출 (PubMed Open Access)
- 복잡한 표 구조 정보 풍부
- 헤더 및 위치 정보 포함

## 데이터 구조
다운로드 후 데이터 구조를 확인하여 로더를 조정하세요.

## 참고
- 대규모 데이터셋이므로 샘플만 사용하는 것을 권장
- 전체 다운로드는 시간이 오래 걸릴 수 있음
""")
        
        print(f"다운로드 가이드 저장: {guide_path}")
        return output_dir
    
    def download_all(self):
        """모든 외부 데이터셋 다운로드"""
        print("="*70)
        print("모든 외부 데이터셋 다운로드 시작")
        print("="*70)
        
        results = {}
        
        print("\n[1/3] TabRecSet 다운로드...")
        try:
            tabrecset_dir = self.download_tabrecset()
            results['tabrecset'] = str(tabrecset_dir)
        except Exception as e:
            print(f"✗ TabRecSet 다운로드 실패: {e}")
            results['tabrecset'] = None
        
        print("\n[2/3] WTW-Dataset 다운로드...")
        try:
            wtw_dir = self.download_wtw_dataset()
            results['wtw'] = str(wtw_dir)
        except Exception as e:
            print(f"✗ WTW-Dataset 다운로드 실패: {e}")
            results['wtw'] = None
        
        print("\n[3/3] PubTables-1M 다운로드...")
        try:
            pubtables_dir = self.download_pubtables1m()
            results['pubtables1m'] = str(pubtables_dir)
        except Exception as e:
            print(f"✗ PubTables-1M 다운로드 실패: {e}")
            results['pubtables1m'] = None
        
        print("\n" + "="*70)
        print("다운로드 완료 요약")
        print("="*70)
        for name, path in results.items():
            status = "✓" if path else "✗"
            print(f"{status} {name}: {path or '실패'}")
        
        return results


def main():
    """메인 실행 함수"""
    downloader = ExternalDatasetDownloader()
    
    print("외부 데이터셋 다운로더")
    print("지원 데이터셋:")
    print("  1. TabRecSet (Figshare)")
    print("  2. WTW-Dataset (GitHub)")
    print("  3. PubTables-1M (Microsoft Research)")
    print("\n모든 데이터셋 다운로드 시작...")
    
    results = downloader.download_all()
    
    print("\n" + "="*70)
    print("다운로드 가이드가 각 데이터셋 디렉토리에 생성되었습니다.")
    print("수동 다운로드가 필요한 경우 가이드를 참고하세요.")
    print("="*70)


if __name__ == '__main__':
    main()



