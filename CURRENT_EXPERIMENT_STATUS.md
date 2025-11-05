# 현재 실험 상태 및 결과 요약

## ✅ 완료된 실험

### 실험 1: RAG-Evaluation-Dataset-KO 전체 데이터셋

**실행 시간**: 2025-11-05 19:26:27  
**데이터셋**: RAG-Evaluation-Dataset-KO  
**테이블 수**: 26개  
**쿼리 수**: 300개

#### 파싱 성능 결과
- ✅ 26개 테이블 처리 완료
- 베이스라인 포함 (TATR, Sato)

#### RAG 성능 결과

| 메트릭 | KG-RAG (HeaderRAG) | Naive RAG | 차이 |
|:-------|:------------------|:----------|:-----|
| Precision | 0.1471 ± 0.3502 | 0.1480 ± 0.3500 | -0.0009 (-0.62%) |
| Recall | 0.0963 ± 0.2192 | 0.0981 ± 0.2195 | -0.0018 (-1.87%) |
| F1 Score | 0.1146 ± 0.2682 | 0.1158 ± 0.2681 | -0.0012 (-1.06%) |
| MRR | 0.0148 ± 0.0336 | 0.0145 ± 0.0325 | +0.0003 (+2.31%) |

**결과 파일**:
- `results/full_experiment_summary.json`
- `results/analysis/parsing_performance_table.csv`
- `results/analysis/rag_performance_table.csv`
- `EXPERIMENT_RESULTS.md`

---

## 🔄 진행 중인 실험

### 실험 2: 3개 데이터셋 통합 실험

**대상 데이터셋**:
1. TabRecSet (Figshare) - 다운로드 필요
2. WTW-Dataset (GitHub) - 실제 데이터 다운로드 필요
3. PubTables-1M (Microsoft Research) - 실제 데이터 다운로드 필요

**현재 상태**:
- ✅ 데이터셋 로더 구현 완료
- ✅ 실험 스크립트 준비 완료
- ⚠️ 실제 테이블 데이터 다운로드 필요

**실행 스크립트**: `experiments/run_three_datasets_experiment.py`

---

## 📊 데이터셋 준비 상태

### ✅ 사용 가능

1. **RAG-Evaluation-Dataset-KO**
   - 테이블: 26개
   - 위치: `data/extracted_tables/`
   - 상태: ✅ 실험 완료

### ⚠️ 다운로드 필요

1. **TabRecSet**
   - 상태: 다운로드 가이드만 생성됨
   - 다운로드: https://figshare.com/articles/dataset/TabRecSet_A_Large_Scale_Dataset_for_End-to-end_Table_Recognition_in_the_Wild/20647788
   - 크기: 5.28 GB

2. **WTW-Dataset**
   - 상태: GitHub 저장소 클론 완료, 실제 데이터 없음
   - 다운로드: https://tianchi.aliyun.com/dataset/dataDetail?dataId=108587
   - 형식: XML 파일 (테이블 구조 정보)

3. **PubTables-1M**
   - 상태: GitHub 저장소 클론 완료, 실제 데이터 없음
   - 다운로드 옵션:
     - Hugging Face: `bsmock/pubtables-1m` (권장)
     - Microsoft Research Open Data
   - 크기: 매우 큼 (샘플 사용 권장)

---

## 🎯 다음 단계

### 옵션 1: 다른 데이터셋 다운로드 후 실험

1. **TabRecSet 다운로드** (가장 간단)
   ```bash
   cd data/tabrecset
   wget https://figshare.com/ndownloader/articles/20647788/versions/9 -O tabrecset.zip
   unzip tabrecset.zip
   ```

2. **PubTables-1M 샘플 다운로드** (Hugging Face)
   ```python
   from datasets import load_dataset
   dataset = load_dataset('bsmock/pubtables-1m', split='train[:1000]')
   ```

3. **실험 실행**
   ```bash
   python experiments/run_three_datasets_experiment.py
   ```

### 옵션 2: 현재 데이터로 추가 분석

현재 RAG-Evaluation-Dataset-KO 데이터로 추가 분석 진행:
- 복잡도별 성능 분석
- 도메인별 성능 분석
- 오류 분석

---

## 📁 생성된 파일

### 실험 결과
- `results/full_experiment_summary.json` - 전체 실험 요약
- `results/analysis/` - 성능 테이블 및 분석 결과
- `EXPERIMENT_RESULTS.md` - 상세 결과 리포트

### 다운로드 가이드
- `data/tabrecset/DOWNLOAD_GUIDE.md`
- `data/wtw/DOWNLOAD_GUIDE.md`
- `data/pubtables1m/DOWNLOAD_GUIDE.md`
- `DATASET_DOWNLOAD_INSTRUCTIONS.md` - 종합 가이드
- `THREE_DATASETS_STATUS.md` - 상태 요약

### 로더 구현
- `utils/wtw_wtb_loader.py` - WTW XML 파서
- `utils/pubtables1m_loader.py` - PubTables-1M JSON 파서
- `utils/multi_dataset_loader.py` - 통합 로더 (업데이트됨)

### 실험 스크립트
- `experiments/run_three_datasets_experiment.py` - 3개 데이터셋 실험
- `experiments/run_all_datasets_experiment.py` - 모든 데이터셋 실험

---

## 💡 권장 사항

1. **즉시 가능**: 현재 RAG-Evaluation-Dataset-KO 데이터로 추가 분석
2. **단기**: TabRecSet 다운로드 (wget으로 간단)
3. **중기**: PubTables-1M 샘플 다운로드 (Hugging Face)
4. **장기**: WTW-Dataset 전체 다운로드 (Tianchi)

모든 준비가 완료되었으므로, 데이터셋만 다운로드하면 바로 실험을 실행할 수 있습니다!

