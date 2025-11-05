# 3개 데이터셋 실험 준비 상태

## 📊 현재 상황

### ✅ 완료된 작업

1. **데이터셋 로더 구현 완료**
   - ✅ WTW-Dataset XML 파서 (`utils/wtw_wtb_loader.py`)
   - ✅ PubTables-1M JSON 파서 (`utils/pubtables1m_loader.py`)
   - ✅ TabRecSet 로더 (CSV/XLSX 지원)
   - ✅ 통합 로더 업데이트 (`utils/multi_dataset_loader.py`)

2. **실험 스크립트 준비 완료**
   - ✅ `experiments/run_three_datasets_experiment.py`
   - ✅ 다운로드 가이드 생성

3. **GitHub 저장소 클론 완료**
   - ✅ WTW-Dataset: `data/wtw/`
   - ✅ PubTables-1M: `data/pubtables1m/table-transformer/`

### ⚠️ 다운로드 필요한 데이터

#### 1. TabRecSet
- **상태**: 다운로드 가이드만 생성됨
- **다운로드 링크**: https://figshare.com/articles/dataset/TabRecSet_A_Large_Scale_Dataset_for_End-to-end_Table_Recognition_in_the_Wild/20647788
- **크기**: 5.28 GB
- **방법**: 
  ```bash
  cd data/tabrecset
  wget https://figshare.com/ndownloader/articles/20647788/versions/9 -O tabrecset.zip
  unzip tabrecset.zip
  ```

#### 2. WTW-Dataset
- **상태**: GitHub 저장소만 클론됨 (README, 스크립트만 있음)
- **실제 데이터 다운로드**: Tianchi Alibaba Cloud
  - 링크: https://tianchi.aliyun.com/dataset/dataDetail?dataId=108587
  - 필요: Alibaba Cloud 계정 (선택적)
- **데이터 구조**: 
  ```
  data/wtw/data/
    train/
      images/  (다운로드 필요)
      xml/     (다운로드 필요)
    test/
      images/  (다운로드 필요)
      xml/     (다운로드 필요)
  ```

#### 3. PubTables-1M
- **상태**: GitHub 저장소만 클론됨 (코드만 있음)
- **실제 데이터 다운로드 옵션**:
  
  **옵션 A: Hugging Face (권장, 가장 쉬움)**
  ```python
  from datasets import load_dataset
  dataset = load_dataset('bsmock/pubtables-1m', split='train[:1000]')  # 샘플
  ```
  
  **옵션 B: Microsoft Research Open Data**
  - 링크: https://msropendata.com/datasets/505fcbe3-1383-42b1-913a-f651b8b712d3
  
  **옵션 C: GitHub 저장소의 다운로드 스크립트 사용**
  ```bash
  cd data/pubtables1m/table-transformer
  # README.md 참고하여 다운로드 스크립트 실행
  ```

---

## 🚀 실험 실행 방법

### 현재 가능한 실험

현재는 **RAG-Evaluation-Dataset-KO (26개 테이블)**만 사용 가능합니다.

```bash
# 기존 실험 (RAG-Evaluation-Dataset-KO만)
python experiments/run_full_experiment_with_new_metrics.py
```

### 다른 데이터셋 다운로드 후

다운로드 완료 후:

```bash
# 3개 데이터셋 실험 실행
python experiments/run_three_datasets_experiment.py
```

---

## 📥 빠른 다운로드 가이드

### TabRecSet 다운로드 (가장 간단)

```bash
cd data/tabrecset
wget https://figshare.com/ndownloader/articles/20647788/versions/9 -O tabrecset.zip
unzip tabrecset.zip
# 압축 해제 후 구조 확인
```

### PubTables-1M 다운로드 (Hugging Face 권장)

```python
# Python에서 실행
from datasets import load_dataset
import pandas as pd
from pathlib import Path

# 샘플 데이터 다운로드 (1000개)
dataset = load_dataset('bsmock/pubtables-1m', split='train[:1000]')

# 데이터 저장
output_dir = Path('data/pubtables1m/data')
output_dir.mkdir(parents=True, exist_ok=True)

for i, item in enumerate(dataset):
    # JSON으로 저장
    with open(output_dir / f'table_{i}.json', 'w') as f:
        json.dump(item, f)
```

### WTW-Dataset 다운로드

1. https://tianchi.aliyun.com/dataset/dataDetail?dataId=108587 방문
2. 데이터 다운로드
3. `data/wtw/data/` 디렉토리에 압축 해제

---

## 🔍 데이터셋별 상세 정보

### TabRecSet
- **출처**: [Figshare](https://figshare.com/articles/dataset/TabRecSet_A_Large_Scale_Dataset_for_End-to-end_Table_Recognition_in_the_Wild/20647788)
- **규모**: 대규모
- **특징**: 실제 환경(인 와일드) 테이블 인식용
- **크기**: 5.28 GB

### WTW-Dataset  
- **출처**: [GitHub](https://github.com/wangwen-whu/WTW-Dataset) | [Tianchi](https://tianchi.aliyun.com/dataset/dataDetail?dataId=108587)
- **규모**: 14,581개 이미지
- **특징**: 7가지 도전적인 케이스 포함
- **형식**: XML (테이블 구조 정보)

### PubTables-1M
- **출처**: [Microsoft Research](https://www.microsoft.com/en-us/research/publication/pubtables-1m/) | [Hugging Face](https://huggingface.co/datasets/bsmock/pubtables-1m)
- **규모**: 약 100만 개의 표
- **특징**: 과학 논문에서 추출, 복잡한 표 구조
- **형식**: JSON, HTML

---

## 💡 권장 다운로드 순서

1. **TabRecSet** (가장 간단, wget으로 바로 다운로드 가능)
2. **PubTables-1M** (Hugging Face에서 샘플 다운로드 권장)
3. **WTW-Dataset** (Tianchi 계정 필요할 수 있음)

---

## 📝 다음 단계

1. 위 가이드에 따라 데이터셋 다운로드
2. 다운로드 완료 후 `python experiments/run_three_datasets_experiment.py` 실행
3. 결과 분석 및 비교

각 데이터셋의 DOWNLOAD_GUIDE.md 파일에 더 자세한 정보가 있습니다.

