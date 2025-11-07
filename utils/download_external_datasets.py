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
- Hugging Face: https://huggingface.co/datasets/bsmock/pubtables-1m

## 다운로드 방법

### 방법 1: git-lfs를 사용한 전체 다운로드 (추천) ⭐

**전체 데이터셋이 필요한 경우:**

```bash
# 1. git-lfs 설치 확인 및 초기화
git lfs install

# 2. 전체 데이터셋 클론 (수십 GB, 시간 오래 걸림)
git clone https://huggingface.co/datasets/bsmock/pubtables-1m data/pubtables1m/pubtables-1m

# 3. 포인터만 다운로드 (큰 파일 없이, 나중에 필요할 때만 다운로드)
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/bsmock/pubtables-1m data/pubtables1m/pubtables-1m
```

**Python 코드로 사용:**

```python
from utils.download_external_datasets import ExternalDatasetDownloader

downloader = ExternalDatasetDownloader()
# 전체 다운로드 (git-lfs 사용)
pubtables_dir = downloader.download_pubtables1m_hf(use_git_lfs=True)
```

### 방법 2: Hugging Face Datasets 라이브러리 사용 (샘플만)

**샘플만 필요한 경우 (권장):**

```python
from datasets import load_dataset
import json
from pathlib import Path

# 샘플만 다운로드 (예: 1000개)
dataset = load_dataset("bsmock/pubtables-1m", split="train", streaming=True)
samples = []
for i, item in enumerate(dataset):
    if i >= 1000:
        break
    samples.append(item)

# JSON 파일로 저장
output_dir = Path("data/pubtables1m")
output_dir.mkdir(parents=True, exist_ok=True)

for i, sample in enumerate(samples):
    with open(output_dir / f"table_{i:06d}.json", 'w', encoding='utf-8') as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
```

**Python 코드로 사용:**

```python
from utils.download_external_datasets import ExternalDatasetDownloader

downloader = ExternalDatasetDownloader()
# 샘플만 다운로드 (1000개)
pubtables_dir = downloader.download_pubtables1m_hf(num_samples=1000)
```

### 방법 3: Hugging Face Datasets Server API (샘플 확인용)

**주의**: 이 방법은 첫 몇 행만 가져옵니다. 전체 다운로드에는 부적합합니다.

```bash
# 첫 100개 행만 가져오기 (샘플 확인용)
curl -X GET "https://datasets-server.huggingface.co/first-rows?dataset=bsmock%2Fpubtables-1m&config=default&split=train&length=100" > sample.json
```

### 방법 3: GitHub 저장소 사용
```bash
git clone https://github.com/microsoft/table-transformer.git
cd table-transformer
# 데이터셋 다운로드 스크립트 실행 (README 확인)
```

### 방법 4: 직접 다운로드
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
- 전체 다운로드는 시간이 오래 걸릴 수 있음 (수십 GB)
- Hugging Face에서 스트리밍 방식으로 샘플만 다운로드하는 것을 권장
""")
        
        print(f"다운로드 가이드 저장: {guide_path}")
        return output_dir
    
    def download_pubtables1m_hf(self, output_dir: Optional[str] = None, num_samples: int = 1000, use_git_lfs: bool = False) -> Path:
        """
        Hugging Face에서 PubTables-1M 데이터셋 다운로드
        
        Args:
            output_dir: 출력 디렉토리 (None이면 data/pubtables1m)
            num_samples: 다운로드할 샘플 수 (전체는 매우 큼, git-lfs 사용 시 무시됨)
            use_git_lfs: True면 git-lfs로 전체 다운로드, False면 datasets 라이브러리로 샘플만
        
        Returns:
            다운로드된 데이터셋 디렉토리 경로
        """
        if output_dir is None:
            output_dir = self.data_dir / "pubtables1m"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if use_git_lfs:
            return self._download_pubtables1m_git_lfs(output_dir)
        else:
            return self._download_pubtables1m_datasets(output_dir, num_samples)
    
    def _download_pubtables1m_git_lfs(self, output_dir: Path) -> Path:
        """git-lfs를 사용하여 전체 데이터셋 다운로드"""
        print("="*70)
        print("PubTables-1M 데이터셋 다운로드 (git-lfs)")
        print("="*70)
        
        # git-lfs 설치 확인
        try:
            result = subprocess.run(
                ['git', 'lfs', 'version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise FileNotFoundError("git-lfs가 설치되지 않았습니다")
            print("✓ git-lfs 확인됨")
        except FileNotFoundError:
            print("✗ git-lfs가 설치되지 않았습니다")
            print("\n설치 방법:")
            print("1. https://git-lfs.com 방문")
            print("2. 운영체제에 맞는 설치 파일 다운로드")
            print("3. 또는 패키지 매니저 사용:")
            print("   - macOS: brew install git-lfs")
            print("   - Ubuntu: sudo apt install git-lfs")
            print("   - Windows: choco install git-lfs")
            raise
        
        # git-lfs 초기화
        try:
            print("\ngit-lfs 초기화 중...")
            subprocess.run(['git', 'lfs', 'install'], check=True, timeout=30)
            print("✓ git-lfs 초기화 완료")
        except Exception as e:
            print(f"✗ git-lfs 초기화 실패: {e}")
            raise
        
        # 데이터셋 클론
        repo_url = "https://huggingface.co/datasets/bsmock/pubtables-1m"
        clone_dir = output_dir / "pubtables-1m"
        
        if clone_dir.exists():
            print(f"\n⚠️  디렉토리가 이미 존재합니다: {clone_dir}")
            response = input("덮어쓰시겠습니까? (y/N): ")
            if response.lower() != 'y':
                print("다운로드 취소됨")
                return clone_dir
            import shutil
            shutil.rmtree(clone_dir)
        
        try:
            print(f"\n데이터셋 클론 중... (이 작업은 시간이 오래 걸릴 수 있습니다)")
            print(f"대상: {repo_url}")
            result = subprocess.run(
                ['git', 'clone', repo_url, str(clone_dir)],
                capture_output=True,
                text=True,
                timeout=3600  # 1시간 타임아웃
            )
            
            if result.returncode == 0:
                print(f"\n✓ 다운로드 완료: {clone_dir}")
                print("\n참고: 전체 데이터셋은 매우 큽니다 (수십 GB)")
                print("필요한 경우 샘플만 사용하는 것을 권장합니다.")
            else:
                print(f"\n✗ 클론 실패: {result.stderr}")
                raise Exception(f"git clone 실패: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("\n✗ 다운로드 시간 초과")
            print("네트워크 상태를 확인하고 다시 시도하세요.")
            raise
        except Exception as e:
            print(f"\n✗ 다운로드 실패: {e}")
            print("\n대안:")
            print("1. 포인터만 다운로드: GIT_LFS_SKIP_SMUDGE=1 git clone ...")
            print("2. datasets 라이브러리로 샘플만 다운로드")
            raise
        
        return clone_dir
    
    def _download_pubtables1m_datasets(self, output_dir: Path, num_samples: int) -> Path:
        """datasets 라이브러리를 사용하여 샘플만 다운로드"""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets 라이브러리가 필요합니다. 설치: pip install datasets"
            )
        
        print("="*70)
        print(f"PubTables-1M 데이터셋 다운로드 (Hugging Face Datasets)")
        print(f"샘플 수: {num_samples}개")
        print("="*70)
        
        try:
            # 스트리밍 방식으로 샘플만 다운로드
            print("\n데이터셋 로드 중...")
            dataset = load_dataset("bsmock/pubtables-1m", split="train", streaming=True)
            
            import json
            samples = []
            print(f"\n{num_samples}개 샘플 다운로드 중...")
            for i, item in enumerate(tqdm(dataset, total=num_samples, desc="다운로드")):
                if i >= num_samples:
                    break
                samples.append(item)
            
            # JSON 파일로 저장
            print(f"\n{len(samples)}개 샘플 저장 중...")
            for i, sample in enumerate(tqdm(samples, desc="저장")):
                output_file = output_dir / f"table_{i:06d}.json"
                
                # bytes 타입을 base64로 인코딩하거나 제거
                sample_clean = {}
                for key, value in sample.items():
                    if isinstance(value, bytes):
                        # bytes는 base64로 인코딩
                        import base64
                        sample_clean[key] = base64.b64encode(value).decode('utf-8')
                    elif isinstance(value, dict):
                        # 딕셔너리 재귀 처리
                        sample_clean[key] = {
                            k: base64.b64encode(v).decode('utf-8') if isinstance(v, bytes) else v
                            for k, v in value.items()
                        }
                    else:
                        sample_clean[key] = value
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(sample_clean, f, ensure_ascii=False, indent=2)
            
            print(f"\n✓ 다운로드 완료: {output_dir}")
            print(f"  총 {len(samples)}개 테이블 저장됨")
            
        except Exception as e:
            print(f"\n✗ 다운로드 실패: {e}")
            print("\n대안:")
            print("1. git-lfs로 전체 다운로드: use_git_lfs=True")
            print("2. 수동으로 Hugging Face에서 다운로드")
            print("3. Microsoft Research 페이지에서 다운로드")
            raise
        
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




