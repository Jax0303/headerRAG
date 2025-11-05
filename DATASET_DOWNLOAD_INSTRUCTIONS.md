# 3개 데이터셋 다운로드 가이드

사용자가 요청한 3개 데이터셋의 다운로드 방법입니다.

## 📥 데이터셋 다운로드 방법

### 1. TabRecSet (Figshare)

**출처**: https://figshare.com/articles/dataset/TabRecSet_A_Large_Scale_Dataset_for_End-to-end_Table_Recognition_in_the_Wild/20647788

**다운로드 방법**:

```bash
# 방법 1: 웹 브라우저에서
# 1. 위 링크 방문
# 2. "Download all (5.28 GB)" 버튼 클릭
# 3. 다운로드한 파일을 data/tabrecset/ 디렉토리에 압축 해제

# 방법 2: 명령어로 다운로드
cd data/tabrecset
wget https://figshare.com/ndownloader/articles/20647788/versions/9 -O tabrecset.zip
unzip tabrecset.zip
```

**데이터 구조**: 다운로드 후 데이터 구조를 확인하여 필요시 로더를 조정하세요.

---

### 2. WTW-Dataset (GitHub)

**출처**: https://github.com/wangwen-whu/WTW-Dataset

**다운로드 방법**:

```bash
# GitHub에서 클론 (이미 완료됨)
cd data/wtw
git clone https://github.com/wangwen-whu/WTW-Dataset.git .

# 실제 데이터 다운로드 (Tianchi Alibaba Cloud)
# README.md에 있는 다운로드 링크 확인:
# https://tianchi.aliyun.com/dataset/dataDetail?dataId=108587
```

**데이터 구조**:
```
data/
  train/
    images/
    xml/  (테이블 구조 정보)
  test/
    images/
    xml/
    class/  (7개 케이스별 분류)
```

**특징**: 
- 14,581개 이미지
- XML 형식의 테이블 구조 정보
- 7가지 도전적인 케이스 포함:
  1. 기울어진 테이블 (Inclined tables)
  2. 곡선 테이블 (Curved tables)
  3. 가려지거나 흐릿한 테이블 (Occluded/blurred tables)
  4. 극단적 종횡비 테이블 (Extreme aspect ratio tables)
  5. 겹친 테이블 (Overlaid tables)
  6. 다중 색상 테이블 (Multi-color tables)
  7. 불규칙 테이블 (Irregular tables)

**참고**: 실제 테이블 이미지와 XML 파일은 Tianchi Alibaba Cloud에서 다운로드해야 합니다.

---

### 3. PubTables-1M (Microsoft Research)

**출처**: 
- Microsoft Research: https://www.microsoft.com/en-us/research/publication/pubtables-1m/
- GitHub: https://github.com/microsoft/table-transformer
- Hugging Face: https://huggingface.co/datasets/bsmock/pubtables-1m

**다운로드 방법**:

```bash
# 방법 1: Hugging Face에서 (권장)
pip install datasets
python -c "
from datasets import load_dataset
dataset = load_dataset('bsmock/pubtables-1m', split='train[:1000]')  # 샘플만
# 데이터 저장
"

# 방법 2: GitHub 저장소 (이미 클론됨)
cd data/pubtables1m/table-transformer
# README.md에 있는 다운로드 스크립트 확인

# 방법 3: Microsoft Research Open Data
# https://msropendata.com/datasets/505fcbe3-1383-42b1-913a-f651b8b712d3
```

**데이터 구조**:
- JSON 형식의 테이블 데이터
- HTML 형식의 테이블
- 셀별 위치 정보 (bounding box)

**규모**:
- 575,305개 문서 페이지
- 947,642개 완전히 주석 처리된 테이블
- 매우 큰 데이터셋이므로 샘플만 사용하는 것을 권장

---

## 🚀 빠른 시작

### 현재 상태

1. **WTW-Dataset**: GitHub 저장소 클론 완료 ✓
   - 실제 데이터는 Tianchi에서 다운로드 필요

2. **PubTables-1M**: GitHub 저장소 클론 완료 ✓
   - 실제 데이터는 Hugging Face 또는 Microsoft Research에서 다운로드 필요

3. **TabRecSet**: 다운로드 가이드 생성 완료 ✓
   - Figshare에서 직접 다운로드 필요

### 실험 실행

현재는 사용 가능한 데이터(RAG-Evaluation-Dataset-KO)로 실험이 진행 중입니다.

다른 데이터셋을 다운로드한 후:

```bash
# 3개 데이터셋 실험 실행
python experiments/run_three_datasets_experiment.py
```

---

## 📝 참고 사항

1. **TabRecSet**: 5.28 GB 크기로 다운로드에 시간이 걸릴 수 있습니다.
2. **WTW-Dataset**: Tianchi Alibaba Cloud 계정이 필요할 수 있습니다.
3. **PubTables-1M**: 전체 데이터셋은 매우 크므로 샘플만 사용하는 것을 권장합니다.

각 데이터셋의 DOWNLOAD_GUIDE.md 파일에 더 자세한 정보가 있습니다.

