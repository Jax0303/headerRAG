# 데이터셋 전략 가이드

## 📚 추천 데이터셋 및 사용 전략

### 데이터셋 목록

#### 1. PubTables-1M (Microsoft Research)
- **규모**: 약 100만 개의 표
- **출처**: 과학 논문에서 추출
- **특징**: 
  - 복잡한 표 구조 정보 풍부
  - 헤더 및 위치 정보 포함
  - 대규모 데이터셋
- **다운로드**: 
  - GitHub: https://github.com/microsoft/table-transformer
  - DatasetNinja: https://datasetninja.com/pubtables-1m
  - Microsoft Research: https://www.microsoft.com/en-us/research/publication/pubtables-1m/
- **사용 시나리오**: 초기 실험, 대규모 검증

#### 2. TabRecSet (Figshare)
- **규모**: 중규모
- **출처**: Figshare
- **특징**:
  - 이중언어 (영어/중국어) 표 데이터
  - 극단적인 케이스 포함
  - 다양한 복잡도 레벨
- **다운로드**: Figshare에서 직접 다운로드
- **사용 시나리오**: 초기 실험, 극단 케이스 테스트

#### 3. KorWikiTabular/TQ (GitHub)
- **규모**: 중규모
- **출처**: 한국어 위키피디아
- **특징**:
  - 한국어 위키피디아 표 데이터
  - 한국어 표 구조 특화
  - TQ (Table Question) 태스크 지원
- **다운로드**: 논문 GitHub 저장소에서 확인
- **사용 시나리오**: 한국어 특화 실험

#### 4. RAG-Evaluation-Dataset-KO (기존)
- **규모**: 64개 문서, 300개 질문
- **출처**: Allganize
- **특징**:
  - 한국어 RAG 평가 데이터셋
  - 5개 도메인 (finance, public, medical, law, commerce)
  - 실제 한국 기업/공공기관 데이터
- **사용 시나리오**: 한국어 특화 실험, 실제 평가

---

## 🎯 추천 사용 전략

### 전략 1: 초기 실험 (권장)

**목표**: 대규모 데이터셋으로 일반화 성능 검증

**데이터셋 조합**:
- PubTables-1M (샘플 100-1000개)
- 또는 TabRecSet (전체)

**실행 방법**:
```bash
# PubTables-1M 샘플 사용
python experiments/run_multi_dataset_experiments.py \
    --datasets pubtables1m \
    --max_tables_per_dataset 100 \
    --experiment all

# TabRecSet 사용
python experiments/run_multi_dataset_experiments.py \
    --datasets tabrecset \
    --experiment all
```

**장점**:
- 복잡한 표 구조 처리 능력 검증
- 대규모 데이터로 일반화 성능 확인
- 극단 케이스 처리 능력 평가

---

### 전략 2: 한국어 특화 실험 (권장)

**목표**: 한국어 표 데이터에 대한 특화 성능 검증

**데이터셋 조합**:
- KorWikiTabular + RAG-Evaluation-Dataset-KO

**실행 방법**:
```bash
python experiments/run_multi_dataset_experiments.py \
    --datasets korwiki_tabular rag_eval_ko \
    --max_tables_per_dataset 50 \
    --experiment all
```

**장점**:
- 한국어 표 구조 특화 성능 확인
- 실제 한국 기업 데이터로 검증
- 한국어 RAG 태스크에 최적화

---

### 전략 3: 극단 케이스 테스트

**목표**: 다양한 복잡도와 케이스 처리 능력 검증

**데이터셋 조합**:
- TabRecSet (전체)

**실행 방법**:
```bash
python experiments/run_multi_dataset_experiments.py \
    --datasets tabrecset \
    --experiment all
```

**장점**:
- 극단적인 표 구조 처리 능력 확인
- 이중언어 처리 능력 평가

---

### 전략 4: 종합 실험

**목표**: 모든 데이터셋으로 포괄적 검증

**데이터셋 조합**:
- PubTables-1M (샘플) + TabRecSet + KorWikiTabular + RAG-Evaluation-Dataset-KO

**실행 방법**:
```bash
python experiments/run_multi_dataset_experiments.py \
    --datasets pubtables1m tabrecset korwiki_tabular rag_eval_ko \
    --max_tables_per_dataset 50 \
    --experiment all
```

**장점**:
- 다양한 데이터셋으로 포괄적 검증
- 일반화 성능 및 특화 성능 모두 확인

---

## 📝 데이터셋 다운로드 방법

### PubTables-1M

```python
from utils.download_datasets import DatasetDownloader

downloader = DatasetDownloader()
pubtables_dir = downloader.download_pubtables1m(use_sample=True)
# 가이드 파일 생성됨: data/pubtables1m/DOWNLOAD_GUIDE.md
```

**수동 다운로드**:
1. GitHub 저장소 클론: `git clone https://github.com/microsoft/table-transformer.git`
2. 데이터셋 다운로드 스크립트 실행
3. 데이터를 `data/pubtables1m/` 디렉토리에 저장

### TabRecSet

```python
from utils.download_datasets import DatasetDownloader

downloader = DatasetDownloader()
tabrecset_dir = downloader.download_tabrecset()
# 가이드 파일 생성됨: data/tabrecset/DOWNLOAD_GUIDE.md
```

**수동 다운로드**:
1. Figshare에서 데이터셋 다운로드
2. 압축 해제
3. 데이터를 `data/tabrecset/` 디렉토리에 저장

### KorWikiTabular

```python
from utils.download_datasets import DatasetDownloader

downloader = DatasetDownloader()
korwiki_dir = downloader.download_korwiki_tabular()
# 가이드 파일 생성됨: data/korwiki_tabular/DOWNLOAD_GUIDE.md
```

**수동 다운로드**:
1. 논문 GitHub 저장소에서 데이터셋 링크 확인
2. 데이터셋 다운로드
3. 데이터를 `data/korwiki_tabular/` 디렉토리에 저장

---

## 🔄 실험 실행 예제

### 예제 1: 초기 실험 (PubTables-1M 샘플)

```bash
# 1. 데이터셋 다운로드 가이드 생성
python utils/download_datasets.py

# 2. PubTables-1M 샘플 데이터 준비 (수동)
# data/pubtables1m/DOWNLOAD_GUIDE.md 참조

# 3. 실험 실행
python experiments/run_multi_dataset_experiments.py \
    --datasets pubtables1m \
    --max_tables_per_dataset 100 \
    --experiment all \
    --include_baselines
```

### 예제 2: 한국어 특화 실험

```bash
# 1. KorWikiTabular 데이터 준비 (수동)
# data/korwiki_tabular/DOWNLOAD_GUIDE.md 참조

# 2. 실험 실행
python experiments/run_multi_dataset_experiments.py \
    --datasets korwiki_tabular rag_eval_ko \
    --max_tables_per_dataset 50 \
    --experiment all \
    --include_baselines
```

### 예제 3: 극단 케이스 테스트

```bash
# 1. TabRecSet 데이터 준비 (수동)
# data/tabrecset/DOWNLOAD_GUIDE.md 참조

# 2. 실험 실행
python experiments/run_multi_dataset_experiments.py \
    --datasets tabrecset \
    --experiment all \
    --include_baselines
```

---

## 💡 실험 전략 요약

| 시나리오 | 데이터셋 | 목적 | 권장 테이블 수 |
|:--------|:--------|:-----|:------------|
| **초기 실험** | PubTables-1M (샘플) | 대규모 검증 | 100-1000개 |
| **초기 실험** | TabRecSet | 극단 케이스 | 전체 |
| **한국어 특화** | KorWikiTabular + RAG-Eval-KO | 한국어 특화 | 각 50개 |
| **종합 실험** | 모든 데이터셋 | 포괄적 검증 | 각 50개 |

---

## 📊 데이터셋별 특징 비교

| 데이터셋 | 규모 | 언어 | 복잡도 | 특화 영역 |
|:--------|:----|:----|:------|:---------|
| **PubTables-1M** | 대규모 (100만) | 영어 | 높음 | 과학 논문 표 |
| **TabRecSet** | 중규모 | 영어/중국어 | 매우 높음 | 극단 케이스 |
| **KorWikiTabular** | 중규모 | 한국어 | 중간 | 위키피디아 표 |
| **RAG-Eval-KO** | 소규모 (64문서) | 한국어 | 중간 | 한국 기업/공공 데이터 |

---

## ✅ 다음 단계

1. **데이터셋 다운로드**: 각 데이터셋의 DOWNLOAD_GUIDE.md 참조
2. **데이터 준비**: 데이터를 해당 디렉토리에 저장
3. **실험 실행**: 위의 예제 명령어 사용
4. **결과 분석**: 데이터셋별 성능 비교

