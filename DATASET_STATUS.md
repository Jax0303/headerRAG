# 데이터셋 현황 및 실험 결과

## 📊 현재 사용 가능한 데이터셋

### ✅ 사용 가능한 데이터셋

#### 1. RAG-Evaluation-Dataset-KO (활성)
- **위치**: `data/extracted_tables/`
- **로드된 테이블 수**: 26개
- **파일 수**: 42개 (일부는 로드 실패)
- **도메인**: 
  - Finance (금융)
  - Public (공공)
  - Medical (의료)
  - Law (법률)
  - Commerce (커머스)
- **특징**: 한국어 RAG 평가용 실제 기업/공공기관 데이터

### ⚠️ 다운로드 필요한 데이터셋

#### 2. PubTables-1M
- **상태**: 디렉토리 존재, 테이블 파일 없음
- **다운로드 가이드**: `data/pubtables1m/DOWNLOAD_GUIDE.md`
- **규모**: 약 100만 개의 표
- **특징**: 과학 논문에서 추출, 복잡한 표 구조

#### 3. TabRecSet
- **상태**: 디렉토리 존재, 테이블 파일 없음
- **규모**: 중규모
- **특징**: 이중언어 (영어/중국어), 극단적인 케이스 포함

#### 4. KorWikiTabular
- **상태**: 디렉토리 존재, 테이블 파일 없음
- **다운로드 가이드**: `data/korwiki_tabular/DOWNLOAD_GUIDE.md`
- **규모**: 중규모
- **특징**: 한국어 위키피디아 표 데이터

---

## 🧪 실험 결과 요약

### 실험 1: RAG-Evaluation-Dataset-KO만 사용

**데이터셋**: RAG-Evaluation-Dataset-KO (26개 테이블)

**결과**:
- ✅ 파싱 실험 완료 (26개 테이블)
- ✅ RAG 실험 완료 (300개 쿼리)
- 📊 성능:
  - KG-RAG Precision: 0.1471
  - KG-RAG Recall: 0.0963
  - KG-RAG F1: 0.1146
  - Naive RAG Precision: 0.1480
  - Naive RAG Recall: 0.0981
  - Naive RAG F1: 0.1158

### 실험 2: 모든 데이터셋 시도 (진행 중)

**시도한 데이터셋**:
- pubtables1m: 0개 테이블 (다운로드 필요)
- tabrecset: 0개 테이블 (다운로드 필요)
- korwiki_tabular: 0개 테이블 (다운로드 필요)
- rag_eval_ko: 26개 테이블 (사용 가능)

**결과**: RAG-Evaluation-Dataset-KO만 사용하여 실험 진행

---

## 📥 다른 데이터셋 다운로드 방법

### PubTables-1M 다운로드

```bash
# 방법 1: GitHub에서
git clone https://github.com/microsoft/table-transformer.git
cd table-transformer
# 데이터셋 다운로드 스크립트 실행

# 방법 2: DatasetNinja에서
# https://datasetninja.com/pubtables-1m 방문하여 다운로드

# 방법 3: Python 스크립트 사용
python utils/download_datasets.py
```

### TabRecSet 다운로드

```bash
# Figshare에서 직접 다운로드 필요
# 데이터를 data/tabrecset/ 디렉토리에 저장
```

### KorWikiTabular 다운로드

```bash
# 논문 GitHub 저장소에서 데이터셋 링크 확인
# 데이터를 data/korwiki_tabular/ 디렉토리에 저장
```

---

## 🔄 현재 실험 진행 상황

현재 **모든 데이터셋 실험**이 백그라운드에서 실행 중입니다:
- 실행 스크립트: `experiments/run_all_datasets_experiment.py`
- 로그 파일: `results/all_datasets_experiment_log.txt`

**주요 발견**:
- 다른 데이터셋(pubtables1m, tabrecset, korwiki_tabular)은 아직 다운로드되지 않음
- RAG-Evaluation-Dataset-KO만 사용하여 실험 진행
- extracted_tables에 42개 파일이 있지만 실제 로드되는 것은 26개

---

## 💡 권장 사항

### 즉시 가능한 실험
현재 **RAG-Evaluation-Dataset-KO (26개 테이블)**로 실험 진행 중입니다.

### 향후 확장 계획
1. **PubTables-1M 다운로드** (우선순위 높음)
   - 대규모 데이터셋으로 일반화 성능 검증
   - 100-1000개 샘플로 시작 권장

2. **KorWikiTabular 다운로드** (한국어 특화)
   - 한국어 표 구조 특화 성능 검증
   - 위키피디아 데이터로 보완

3. **TabRecSet 다운로드** (극단 케이스)
   - 극단적인 표 구조 처리 능력 검증
   - 다국어 처리 능력 평가

---

## 📁 결과 파일 위치

- **전체 데이터셋 실험 요약**: `results/all_datasets_experiment_summary.json` (생성 예정)
- **실험 로그**: `results/all_datasets_experiment_log.txt`
- **이전 실험 결과**: `results/full_experiment_summary.json`

---

## 결론

현재는 **RAG-Evaluation-Dataset-KO만 사용 가능**하며, 다른 데이터셋들은 다운로드가 필요합니다. 현재 실험은 사용 가능한 데이터로 진행 중이며, 다른 데이터셋을 다운로드하면 더 포괄적인 실험을 수행할 수 있습니다.

