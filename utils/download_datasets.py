"""
데이터셋 다운로드 유틸리티
다양한 표 데이터셋 지원: PubTables-1M, TabRecSet, KorWikiTabular 등
"""

import os
import requests
import pandas as pd
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import zipfile
import json
import subprocess
from tqdm import tqdm
import shutil


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
    
    def download_pubtables1m(self, 
                            output_subdir: str = "pubtables1m",
                            use_sample: bool = True) -> Path:
        """
        PubTables-1M 데이터셋 다운로드 (Microsoft Research)
        
        Args:
            output_subdir: 저장할 하위 디렉토리
            use_sample: 샘플 데이터만 사용 여부 (전체는 매우 큼)
        
        Returns:
            저장된 디렉토리 경로
        """
        dataset_dir = self.output_dir / output_subdir
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("PubTables-1M 데이터셋 다운로드")
        print("="*70)
        print("\n📝 다운로드 방법:")
        print("1. GitHub 저장소: https://github.com/microsoft/table-transformer")
        print("2. DatasetNinja: https://datasetninja.com/pubtables-1m")
        print("3. Microsoft Research: https://www.microsoft.com/en-us/research/publication/pubtables-1m/")
        print("\n💡 특징:")
        print("  - 약 100만 개의 과학 논문 표")
        print("  - 복잡한 표 구조 정보 풍부")
        print("  - 헤더 및 위치 정보 포함")
        print("\n⚠️  주의: 전체 데이터셋은 매우 큽니다 (수십 GB)")
        if use_sample:
            print("  → 샘플 데이터 사용 권장")
        
        # 다운로드 가이드 파일 생성
        guide_path = dataset_dir / "DOWNLOAD_GUIDE.md"
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write("""# PubTables-1M 다운로드 가이드

## 다운로드 방법

### 방법 1: GitHub 저장소에서
```bash
git clone https://github.com/microsoft/table-transformer.git
cd table-transformer
# 데이터셋 다운로드 스크립트 실행
```

### 방법 2: 직접 다운로드
- Microsoft Research 페이지에서 다운로드 링크 확인
- DatasetNinja에서 데이터셋 정보 확인

## 데이터셋 특징
- 약 100만 개의 표
- 과학 논문에서 추출
- 복잡한 표 구조 정보 풍부
- 헤더 및 위치 정보 포함

## 사용 방법
```python
from utils.download_datasets import DatasetDownloader

downloader = DatasetDownloader()
pubtables_dir = downloader.download_pubtables1m(use_sample=True)
```
""")
        
        print(f"\n✅ 가이드 파일 생성: {guide_path}")
        return dataset_dir
    
    def download_tabrecset(self,
                          output_subdir: str = "tabrecset") -> Path:
        """
        TabRecSet 데이터셋 다운로드 (Figshare)
        
        Args:
            output_subdir: 저장할 하위 디렉토리
        
        Returns:
            저장된 디렉토리 경로
        """
        dataset_dir = self.output_dir / output_subdir
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("TabRecSet 데이터셋 다운로드")
        print("="*70)
        print("\n📝 다운로드 방법:")
        print("1. Figshare: https://figshare.com/articles/dataset/TabRecSet/...")
        print("\n💡 특징:")
        print("  - 이중언어 (영어/중국어) 표 데이터")
        print("  - 극단적인 케이스 포함")
        print("  - 다양한 복잡도 레벨")
        
        # 다운로드 가이드 파일 생성
        guide_path = dataset_dir / "DOWNLOAD_GUIDE.md"
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write("""# TabRecSet 다운로드 가이드

## 다운로드 방법
1. Figshare에서 데이터셋 페이지 접속
2. 다운로드 링크를 통해 데이터셋 다운로드
3. 압축 해제 후 이 디렉토리에 저장

## 데이터셋 특징
- 이중언어 (영어/중국어) 표 데이터
- 극단적인 케이스 포함
- 다양한 복잡도 레벨

## 사용 방법
```python
from utils.download_datasets import DatasetDownloader

downloader = DatasetDownloader()
tabrecset_dir = downloader.download_tabrecset()
```
""")
        
        print(f"\n✅ 가이드 파일 생성: {guide_path}")
        return dataset_dir
    
    def download_korwiki_tabular(self,
                                 output_subdir: str = "korwiki_tabular",
                                 github_repo: Optional[str] = None) -> Path:
        """
        KorWikiTabular/TQ 데이터셋 다운로드 (GitHub)
        
        Args:
            output_subdir: 저장할 하위 디렉토리
            github_repo: GitHub 저장소 URL (None이면 가이드만 생성)
        
        Returns:
            저장된 디렉토리 경로
        """
        dataset_dir = self.output_dir / output_subdir
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("KorWikiTabular/TQ 데이터셋 다운로드")
        print("="*70)
        print("\n📝 다운로드 방법:")
        print("1. 논문 저장소 GitHub에서 데이터셋 확인")
        print("2. 해당 논문의 데이터셋 링크 참조")
        print("\n💡 특징:")
        print("  - 한국어 위키피디아 표 데이터")
        print("  - 한국어 표 구조 특화")
        print("  - TQ (Table Question) 태스크 지원")
        
        # 다운로드 가이드 파일 생성
        guide_path = dataset_dir / "DOWNLOAD_GUIDE.md"
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write("""# KorWikiTabular/TQ 다운로드 가이드

## 다운로드 방법
1. 논문의 GitHub 저장소에서 데이터셋 링크 확인
2. 데이터셋 다운로드
3. 압축 해제 후 이 디렉토리에 저장

## 데이터셋 특징
- 한국어 위키피디아 표 데이터
- 한국어 표 구조 특화
- TQ (Table Question) 태스크 지원

## 사용 방법
```python
from utils.download_datasets import DatasetDownloader

downloader = DatasetDownloader()
korwiki_dir = downloader.download_korwiki_tabular()
```
""")
        
        if github_repo:
            try:
                print(f"\nGitHub 저장소 클론 시도: {github_repo}")
                # Git 클론은 사용자가 직접 해야 할 수도 있음
                print("💡 필요시 직접 git clone 명령어 실행:")
                print(f"   git clone {github_repo} {dataset_dir}")
            except Exception as e:
                print(f"경고: 자동 다운로드 실패: {e}")
        
        print(f"\n✅ 가이드 파일 생성: {guide_path}")
        return dataset_dir
    
    def download_rag_eval_ko(self, 
                             output_subdir: str = "rag_eval_ko",
                             use_huggingface: bool = True) -> Path:
        """
        RAG-Evaluation-Dataset-KO 데이터셋 다운로드 (Hugging Face)
        
        Args:
            output_subdir: 저장할 하위 디렉토리
            use_huggingface: Hugging Face datasets 라이브러리 사용 여부
        
        Returns:
            저장된 디렉토리 경로
        """
        dataset_dir = self.output_dir / output_subdir
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("RAG-Evaluation-Dataset-KO 데이터셋 다운로드")
        print("="*70)
        
        if use_huggingface:
            try:
                print("\n📥 Hugging Face에서 다운로드 중...")
                from datasets import load_dataset
                
                # 데이터셋 로드
                dataset = load_dataset("allganize/RAG-Evaluation-Dataset-KO")
                
                # CSV 파일 저장
                if 'train' in dataset:
                    df_documents = dataset['train'].to_pandas()
                    df_documents.to_csv(dataset_dir / "documents.csv", index=False, encoding='utf-8')
                    print(f"✅ documents.csv 저장 완료: {len(df_documents)}개 문서")
                
                # 평가 결과 CSV (있는 경우)
                if 'test' in dataset:
                    df_eval = dataset['test'].to_pandas()
                    df_eval.to_csv(dataset_dir / "rag_evaluation_result.csv", index=False, encoding='utf-8')
                    print(f"✅ rag_evaluation_result.csv 저장 완료: {len(df_eval)}개 질문")
                
                print(f"\n✅ 다운로드 완료: {dataset_dir}")
                return dataset_dir
                
            except ImportError:
                print("⚠️  datasets 라이브러리가 설치되지 않았습니다.")
                print("   설치: pip install datasets")
                use_huggingface = False
        
        if not use_huggingface:
            print("\n📝 수동 다운로드 방법:")
            print("1. Git으로 클론:")
            print("   git clone https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO")
            print(f"2. 또는 다운로드 후 {dataset_dir}에 저장")
            
            # 가이드 파일 생성
            guide_path = dataset_dir / "DOWNLOAD_GUIDE.md"
            with open(guide_path, 'w', encoding='utf-8') as f:
                f.write("""# RAG-Evaluation-Dataset-KO 다운로드 가이드

## 다운로드 방법

### 방법 1: Hugging Face datasets 라이브러리 사용
```python
from datasets import load_dataset
dataset = load_dataset("allganize/RAG-Evaluation-Dataset-KO")
```

### 방법 2: Git으로 클론
```bash
git clone https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO
```

### 방법 3: 직접 다운로드
Hugging Face 페이지에서 직접 다운로드:
https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO

## 데이터셋 특징
- 한국어 RAG 평가 데이터셋
- 5개 도메인 (finance, public, medical, law, commerce)
- 300개 질문
- PDF 문서 포함
""")
            print(f"\n✅ 가이드 파일 생성: {guide_path}")
        
        return dataset_dir
    
    def download_pubtables1m_hf(self,
                                output_subdir: str = "pubtables1m",
                                num_samples: int = 1000) -> Path:
        """
        PubTables-1M 데이터셋을 Hugging Face에서 다운로드
        
        Args:
            output_subdir: 저장할 하위 디렉토리
            num_samples: 다운로드할 샘플 수
        
        Returns:
            저장된 디렉토리 경로
        """
        dataset_dir = self.output_dir / output_subdir / "data"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print(f"PubTables-1M 데이터셋 다운로드 (샘플 {num_samples}개)")
        print("="*70)
        
        try:
            print("\n📥 Hugging Face에서 다운로드 중...")
            from datasets import load_dataset
            
            # 데이터셋 로드 (샘플만)
            split = f'train[:{num_samples}]'
            dataset = load_dataset('bsmock/pubtables-1m', split=split)
            
            print(f"다운로드된 샘플 수: {len(dataset)}")
            
            # 데이터 저장
            saved_count = 0
            for i, item in enumerate(tqdm(dataset, desc="데이터 저장")):
                json_path = dataset_dir / f'table_{i}.json'
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(item, f, ensure_ascii=False, indent=2)
                saved_count += 1
            
            print(f"\n✅ 다운로드 완료: {saved_count}개 테이블 저장됨")
            print(f"   저장 위치: {dataset_dir}")
            return dataset_dir
            
        except ImportError:
            print("⚠️  datasets 라이브러리가 설치되지 않았습니다.")
            print("   설치: pip install datasets")
            return self.download_pubtables1m(output_subdir, use_sample=True)
        except Exception as e:
            print(f"⚠️  다운로드 실패: {e}")
            print("   수동 다운로드 가이드를 참고하세요.")
            return self.download_pubtables1m(output_subdir, use_sample=True)
    
    def download_tabrecset_from_url(self,
                                    output_subdir: str = "tabrecset",
                                    url: Optional[str] = None) -> Path:
        """
        TabRecSet 데이터셋을 URL에서 다운로드
        
        Args:
            output_subdir: 저장할 하위 디렉토리
            url: 다운로드 URL (None이면 기본 URL 사용)
        
        Returns:
            저장된 디렉토리 경로
        """
        dataset_dir = self.output_dir / output_subdir
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("TabRecSet 데이터셋 다운로드")
        print("="*70)
        
        # Figshare 다운로드 URL (예시)
        if url is None:
            # 실제 URL은 확인 필요
            url = "https://figshare.com/ndownloader/articles/20647788/versions/9"
        
        zip_path = dataset_dir / "tabrecset.zip"
        
        try:
            print(f"\n📥 다운로드 중: {url}")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            with open(zip_path, 'wb') as f, tqdm(
                desc="다운로드",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            
            print(f"\n✅ 다운로드 완료: {zip_path}")
            
            # 압축 해제
            print("압축 해제 중...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            
            print(f"✅ 압축 해제 완료: {dataset_dir}")
            
            # ZIP 파일 삭제 (선택사항)
            # zip_path.unlink()
            
            return dataset_dir
            
        except Exception as e:
            print(f"⚠️  다운로드 실패: {e}")
            print("   수동 다운로드 가이드를 참고하세요.")
            return self.download_tabrecset(output_subdir)
    
    def download_tablebank(self,
                          output_subdir: str = "tablebank",
                          github_repo: str = "https://github.com/doc-analysis/TableBank.git") -> Path:
        """
        TableBank 데이터셋 다운로드 (GitHub)
        
        Args:
            output_subdir: 저장할 하위 디렉토리
            github_repo: GitHub 저장소 URL
        
        Returns:
            저장된 디렉토리 경로
        """
        dataset_dir = self.output_dir / output_subdir
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("TableBank 데이터셋 다운로드")
        print("="*70)
        print("\n📝 다운로드 방법:")
        print(f"1. GitHub 저장소: {github_repo}")
        print("\n💡 특징:")
        print("  - 문서 분석용 표 검출 및 인식 데이터셋")
        print("  - Word 및 LaTeX 문서에서 추출")
        print("  - 다양한 표 구조 포함")
        
        # 다운로드 가이드 파일 생성
        guide_path = dataset_dir / "DOWNLOAD_GUIDE.md"
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(f"""# TableBank 다운로드 가이드

## 다운로드 방법

### 방법 1: Git으로 클론
```bash
git clone {github_repo}
cd TableBank
# 데이터셋 파일 확인 및 다운로드
```

### 방법 2: 직접 다운로드
GitHub 저장소에서 데이터셋 파일 다운로드:
{github_repo}

## 데이터셋 특징
- 문서 분석용 표 검출 및 인식 데이터셋
- Word 및 LaTeX 문서에서 추출
- 다양한 표 구조 포함

## 사용 방법
```python
from utils.download_datasets import DatasetDownloader

downloader = DatasetDownloader()
tablebank_dir = downloader.download_tablebank()
```
""")
        
        # Git 클론 시도
        if not (dataset_dir / ".git").exists():
            try:
                print(f"\nGitHub 저장소 클론 시도: {github_repo}")
                subprocess.run(
                    ["git", "clone", github_repo, str(dataset_dir)],
                    check=False,
                    capture_output=True
                )
                if (dataset_dir / ".git").exists():
                    print(f"✅ 클론 완료: {dataset_dir}")
                else:
                    print("💡 수동 클론 필요:")
                    print(f"   git clone {github_repo} {dataset_dir}")
            except Exception as e:
                print(f"경고: 자동 다운로드 실패: {e}")
                print(f"💡 직접 실행: git clone {github_repo} {dataset_dir}")
        
        print(f"\n✅ 가이드 파일 생성: {guide_path}")
        return dataset_dir
    
    def download_synthtabnet(self,
                            output_subdir: str = "synthtabnet",
                            github_repo: str = "https://github.com/IBM/SynthTabNet.git",
                            download_data: bool = False) -> Path:
        """
        SynthTabNet 데이터셋 다운로드
        
        Args:
            output_subdir: 저장할 하위 디렉토리
            github_repo: GitHub 저장소 URL
            download_data: 실제 데이터 다운로드 여부 (False면 가이드만 생성)
        
        Returns:
            저장된 디렉토리 경로
        """
        dataset_dir = self.output_dir / output_subdir
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("SynthTabNet 데이터셋 다운로드")
        print("="*70)
        print("\n📝 다운로드 방법:")
        print(f"1. GitHub 저장소: {github_repo}")
        print("\n💡 특징:")
        print("  - 합성적으로 생성된 표 레이아웃 데이터셋")
        print("  - 약 60만 개의 PNG 이미지와 JSONL 주석")
        print("  - 4가지 스타일: Fintabnet, Marketing, PubTabNet, Sparse")
        print("  - 모든 셀에 바운딩 박스 포함")
        
        # 다운로드 가이드 파일 생성
        guide_path = dataset_dir / "DOWNLOAD_GUIDE.md"
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(f"""# SynthTabNet 다운로드 가이드

## 다운로드 방법

### 방법 1: GitHub 코드 클론
```bash
git clone {github_repo}
cd SynthTabNet
```

### 방법 2: 실제 데이터 다운로드 (v2.0.0)
데이터는 별도로 다운로드해야 합니다:

| 스타일 | 레코드 | 크기 | URL |
|--------|--------|------|-----|
| Fintabnet | 150k | 10GB | https://ds4sd-public-artifacts.s3.eu-de.cloud-object-storage.appdomain.cloud/datasets/synthtabnet%5Fpublic/v2.0.0/fintabnet.zip |
| Marketing | 150k | 8GB | https://ds4sd-public-artifacts.s3.eu-de.cloud-object-storage.appdomain.cloud/datasets/synthtabnet%5Fpublic/v2.0.0/marketing.zip |
| PubTabNet | 150k | 6GB | https://ds4sd-public-artifacts.s3.eu-de.cloud-object-storage.appdomain.cloud/datasets/synthtabnet%5Fpublic/v2.0.0/pubtabnet.zip |
| Sparse | 150k | 3GB | https://ds4sd-public-artifacts.s3.eu-de.cloud-object-storage.appdomain.cloud/datasets/synthtabnet%5Fpublic/v2.0.0/sparse.zip |

각 ZIP 파일을 다운로드한 후 압축 해제하여 이 디렉토리에 저장하세요.

## 데이터셋 특징
- 합성적으로 생성된 표 레이아웃
- 약 60만 개의 PNG 이미지와 JSONL 주석
- 4가지 스타일 (Fintabnet, Marketing, PubTabNet, Sparse)
- 모든 셀에 바운딩 박스 포함
- HTML 형식의 표 구조 포함

## 데이터 형식
- 이미지: PNG 파일
- 주석: JSONL 파일 (각 라인이 JSON 객체)
- 각 JSON 객체는 HTML 형식의 표 구조와 셀 정보 포함

## 사용 방법
```python
from utils.download_datasets import DatasetDownloader

downloader = DatasetDownloader()
synthtabnet_dir = downloader.download_synthtabnet()
```
""")
        
        # GitHub 저장소 클론
        if not (dataset_dir / ".git").exists():
            try:
                print(f"\nGitHub 저장소 클론 시도: {github_repo}")
                subprocess.run(
                    ["git", "clone", github_repo, str(dataset_dir)],
                    check=False,
                    capture_output=True
                )
                if (dataset_dir / ".git").exists():
                    print(f"✅ 클론 완료: {dataset_dir}")
                else:
                    print("💡 수동 클론 필요:")
                    print(f"   git clone {github_repo} {dataset_dir}")
            except Exception as e:
                print(f"경고: 자동 클론 실패: {e}")
        
        print(f"\n✅ 가이드 파일 생성: {guide_path}")
        print("⚠️  주의: 실제 데이터(이미지/JSONL)는 별도로 다운로드해야 합니다.")
        return dataset_dir
    
    def download_tabrecset_github(self,
                                  output_subdir: str = "tabrecset_maxkinny",
                                  github_repo: str = "https://github.com/MaxKinny/TabRecSet.git") -> Path:
        """
        TabRecSet 데이터셋 다운로드 (MaxKinny GitHub 저장소)
        
        Args:
            output_subdir: 저장할 하위 디렉토리
            github_repo: GitHub 저장소 URL
        
        Returns:
            저장된 디렉토리 경로
        """
        dataset_dir = self.output_dir / output_subdir
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("TabRecSet 데이터셋 다운로드 (MaxKinny)")
        print("="*70)
        print("\n📝 다운로드 방법:")
        print(f"1. GitHub 저장소: {github_repo}")
        print("\n💡 특징:")
        print("  - 대규모 카메라 촬영 표 검출 및 인식 데이터셋")
        print("  - 실제 환경에서 촬영된 표 이미지")
        print("  - 다양한 각도와 조명 조건")
        
        # 다운로드 가이드 파일 생성
        guide_path = dataset_dir / "DOWNLOAD_GUIDE.md"
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(f"""# TabRecSet (MaxKinny) 다운로드 가이드

## 다운로드 방법

### 방법 1: Git으로 클론
```bash
git clone {github_repo}
cd TabRecSet
# 데이터셋 파일 확인 및 다운로드
```

## 데이터셋 특징
- 대규모 카메라 촬영 표 검출 및 인식 데이터셋
- 실제 환경에서 촬영된 표 이미지
- 다양한 각도와 조명 조건
- OCR 어노테이션 포함

## 사용 방법
```python
from utils.download_datasets import DatasetDownloader

downloader = DatasetDownloader()
tabrecset_dir = downloader.download_tabrecset_github()
```
""")
        
        # Git 클론 시도
        if not (dataset_dir / ".git").exists():
            try:
                print(f"\nGitHub 저장소 클론 시도: {github_repo}")
                subprocess.run(
                    ["git", "clone", github_repo, str(dataset_dir)],
                    check=False,
                    capture_output=True
                )
                if (dataset_dir / ".git").exists():
                    print(f"✅ 클론 완료: {dataset_dir}")
                else:
                    print("💡 수동 클론 필요:")
                    print(f"   git clone {github_repo} {dataset_dir}")
            except Exception as e:
                print(f"경고: 자동 다운로드 실패: {e}")
                print(f"💡 직접 실행: git clone {github_repo} {dataset_dir}")
        
        print(f"\n✅ 가이드 파일 생성: {guide_path}")
        return dataset_dir
    
    def save_metadata(self, tables_info: List[Dict], filename: str = "metadata.json"):
        """테이블 메타데이터 저장"""
        metadata_path = self.output_dir / filename
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(tables_info, f, ensure_ascii=False, indent=2)
        print(f"메타데이터 저장: {metadata_path}")


def main():
    """예제 실행"""
    downloader = DatasetDownloader()
    
    print("="*70)
    print("데이터셋 다운로더")
    print("="*70)
    
    print("\n📚 지원하는 데이터셋:")
    print("\n1. PubTables-1M (Microsoft Research)")
    print("   - 대규모 과학 논문 표 데이터셋 (약 100만 개)")
    print("   - 복잡한 표 구조 정보 풍부")
    print("   - 초기 실험에 추천")
    print("   사용법: downloader.download_pubtables1m(use_sample=True)")
    
    print("\n2. TabRecSet (Figshare)")
    print("   - 이중언어 (영어/중국어) 표 데이터")
    print("   - 극단적인 케이스 포함")
    print("   - 초기 실험에 추천")
    print("   사용법: downloader.download_tabrecset()")
    
    print("\n3. KorWikiTabular/TQ (GitHub)")
    print("   - 한국어 위키피디아 표 데이터")
    print("   - 한국어 표 구조 특화")
    print("   - 한국 기업 표 특화 실험에 추천")
    print("   사용법: downloader.download_korwiki_tabular()")
    
    print("\n4. RAG-Evaluation-Dataset-KO (기존)")
    print("   - 한국어 RAG 평가 데이터셋")
    print("   - 5개 도메인, 300개 질문")
    print("   사용법: 기존 방식대로 사용")
    
    print("\n5. TableBank (GitHub)")
    print("   - 문서 분석용 표 검출 및 인식 데이터셋")
    print("   - Word 및 LaTeX 문서에서 추출")
    print("   사용법: downloader.download_tablebank()")
    
    print("\n6. SynthTabNet (IBM)")
    print("   - 합성적으로 생성된 표 레이아웃 (약 60만 개)")
    print("   - PNG 이미지와 JSONL 주석")
    print("   사용법: downloader.download_synthtabnet()")
    
    print("\n7. TabRecSet (MaxKinny GitHub)")
    print("   - 대규모 카메라 촬영 표 데이터셋")
    print("   - 실제 환경 촬영 이미지")
    print("   사용법: downloader.download_tabrecset_github()")
    
    print("\n📖 공개 데이터셋:")
    print("1. 공공데이터포털: https://www.data.go.kr")
    print("2. DART: https://dart.fss.or.kr")
    print("3. KOSIS: https://kosis.kr")
    
    print("\n💡 추천 사용 전략:")
    print("  - 초기 실험: PubTables-1M (샘플) 또는 TabRecSet")
    print("  - 한국어 특화: KorWikiTabular + RAG-Evaluation-Dataset-KO")
    print("  - 극단 케이스: TabRecSet")
    print("  - 합성 데이터: SynthTabNet")
    print("  - 문서 기반: TableBank")


if __name__ == "__main__":
    main()

