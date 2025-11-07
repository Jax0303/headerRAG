# 데이터셋 정보

이 문서는 HeaderRAG 프로젝트에서 사용하는 데이터셋에 대한 상세 정보를 제공합니다.

## 목차

1. [RAG-Evaluation-Dataset-KO](#rag-evaluation-dataset-ko)
2. [샘플 데이터셋](#샘플-데이터셋)
3. [공개 데이터셋 추천](#공개-데이터셋-추천)
4. [데이터셋 사용 방법](#데이터셋-사용-방법)

---

## RAG-Evaluation-Dataset-KO

### 개요

Allganize에서 제공하는 한국어 RAG 평가 데이터셋입니다. 5개 도메인에 대한 문서, 질문, 답변을 포함합니다.

**GitHub/Hugging Face**: https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO

### 구조

```
RAG-Evaluation-Dataset-KO/
├── README.md                    # 데이터셋 설명
├── documents.csv                # 문서 메타데이터
├── rag_evaluation_result.csv   # 평가 결과 및 질문
├── finance/                     # 금융 도메인 PDF
├── public/                      # 공공 도메인 PDF
├── medical/                     # 의료 도메인 PDF
├── law/                         # 법률 도메인 PDF
└── commerce/                    # 커머스 도메인 PDF
```

### 도메인별 통계

| 도메인 | 문서 수 | 총 페이지 수 | 질문 수 | Context Type 비율 |
|:------:|:-------:|:------------:|:-------:|:-----------------:|
| finance (금융) | 10 | 301 | 60 | paragraph: 50%, table: 17%, image: 33% |
| public (공공) | 12 | 258 | 60 | paragraph: 67%, table: 25%, image: 8% |
| medical (의료) | 20 | 276 | 60 | paragraph: 75%, table: 8%, image: 17% |
| law (법률) | 12 | 291 | 60 | paragraph: 67%, table: 25%, image: 8% |
| commerce (커머스) | 9 | 211 | 60 | paragraph: 64%, table: 8%, image: 28% |

**총계**: 63개 문서, 1,337 페이지, 300개 질문

### documents.csv 구조

```csv
file_name,domain,page_count,url
finance_doc1.pdf,finance,25,https://...
public_doc1.pdf,public,20,https://...
...
```

**컬럼 설명**:
- `file_name`: PDF 파일명
- `domain`: 도메인 (finance, public, medical, law, commerce)
- `page_count`: 문서 페이지 수
- `url`: 문서 다운로드 URL

### rag_evaluation_result.csv 구조

```csv
domain,question,target_answer,target_file_name,context_type,...
finance,질문 내용,답변 내용,finance_doc1.pdf,table,...
...
```

**주요 컬럼**:
- `domain`: 도메인
- `question`: 질문
- `target_answer`: 정답 답변
- `target_file_name`: 정답이 있는 문서 파일명
- `context_type`: 근거 유형 (paragraph, table, image)

### Context Type 분포

데이터셋의 질문은 다음 세 가지 유형의 근거를 가집니다:

1. **Paragraph (문단)**: 텍스트 문단에서 답을 찾을 수 있는 질문
2. **Table (표)**: 표에서 답을 찾을 수 있는 질문
3. **Image (이미지)**: 이미지에서 답을 찾을 수 있는 질문

표 기반 질문은 HeaderRAG 실험에 특히 유용합니다.

### 사용 방법

```python
from experiments.run_experiments import ExperimentRunner

runner = ExperimentRunner()

# RAG-Evaluation-Dataset-KO 사용
tables = runner.load_test_data("", use_dataset=True)
```

또는 직접 사용:

```python
import pandas as pd

# 문서 메타데이터 로드
documents = pd.read_csv("RAG-Evaluation-Dataset-KO/documents.csv")

# 평가 결과 로드
evaluation_results = pd.read_csv("RAG-Evaluation-Dataset-KO/rag_evaluation_result.csv")

# 금융 도메인 질문만 필터링
finance_questions = evaluation_results[
    evaluation_results['domain'] == 'finance'
]
```

---

## 샘플 데이터셋

### 개요

프로젝트에 포함된 테스트용 샘플 테이블 데이터셋입니다.

### 위치

```
data/
├── sample_tables/          # 샘플 테이블 Excel 파일들
│   ├── table_0.xlsx
│   ├── table_1.xlsx
│   └── ...
├── extracted_tables/       # PDF에서 추출된 테이블
│   ├── finance/
│   ├── public/
│   └── ...
└── metadata.json           # 테이블 메타데이터
```

### 테이블 유형

샘플 데이터셋은 세 가지 유형의 테이블을 포함합니다:

1. **단순 표** (Simple Table)
   - 기본적인 행/열 구조
   - 예: 연도별 매출액, 직원수 등

2. **중첩 헤더 표** (Nested Header Table)
   - 다중 레벨 헤더
   - 예: 매출(국내/해외), 비용(인건비/운영비) 등

3. **병합 셀 표** (Merged Cell Table)
   - 병합된 셀이 있는 표
   - 예: 부서별 직원 정보 등

### 생성 방법

```python
from utils.download_datasets import DatasetDownloader

downloader = DatasetDownloader()

# 샘플 테이블 생성 (10개)
tables = downloader.create_sample_tables(num_tables=10)

# 메타데이터 저장
tables_info = [
    {
        'table_id': f'table_{i}',
        'filename': f'table_{i}.xlsx',
        'shape': list(table.shape),
        'columns': list(table.columns.tolist())
    }
    for i, table in enumerate(tables)
]
downloader.save_metadata(tables_info)
```

### 샘플 테이블 예시

**단순 표**:
```
연도 | 매출액(억원) | 순이익(억원) | 직원수
-----|------------|------------|--------
2020 | 1000       | 100        | 500
2021 | 1200       | 150        | 550
2022 | 1500       | 200        | 600
2023 | 1800       | 250        | 650
```

**중첩 헤더 표**:
```
연도 | 매출        |        | 비용      |        |
     | 국내 | 해외 |        | 인건비 | 운영비 |
-----|------|------|--------|--------|--------|
2020 | 800  | 200  |        | 300    | 200    |
2021 | 900  | 300  |        | 350    | 250    |
```

---

## 공개 데이터셋 추천

### 1. KOSIS 국가통계포털

**URL**: https://kosis.kr/

**특징**:
- 다양한 산업/경제 통계 데이터
- 복잡한 표 구조 포함
- 대규모 데이터 제공
- Excel, CSV 형식 지원

**사용 사례**:
- 산업별 통계 데이터
- 경제 지표 데이터
- 인구 통계 데이터

### 2. 공공데이터포털 (data.go.kr)

**URL**: https://www.data.go.kr/

**특징**:
- 한국 정부 기관의 공개 데이터
- 표/차트 포함
- 기업 재무제표, 통계 데이터, 정책 데이터

**사용 사례**:
- 기업 재무제표 분석
- 정책 데이터 분석
- 공공 통계 데이터

### 3. 금융감독원 전자공시시스템 (DART)

**URL**: https://dart.fss.or.kr/

**특징**:
- 실제 기업 재무제표
- 매우 복잡한 표 구조
- 거대한 규모
- XBRL, Excel 형식

**사용 사례**:
- 기업 재무제표 분석
- 금융 데이터 분석
- 복잡한 표 구조 인식

### 4. 한국은행 경제통계시스템 (ECOS)

**URL**: https://ecos.bok.or.kr/

**특징**:
- 경제 지표 데이터
- 복잡한 시계열 표 데이터
- 정기적으로 업데이트

**사용 사례**:
- 경제 지표 분석
- 시계열 데이터 분석

### 다운로드 방법

```python
from utils.download_datasets import DatasetDownloader

downloader = DatasetDownloader()

# 공공데이터포털에서 데이터 다운로드
dataset_url = "https://www.data.go.kr/.../dataset.csv"
save_path = downloader.download_public_data(dataset_url)
```

---

## 데이터셋 사용 방법

### 방법 1: 샘플 데이터셋 사용

```python
from experiments.run_experiments import ExperimentRunner

runner = ExperimentRunner()

# 샘플 테이블 로드
tables = runner.load_test_data("data/sample_tables")
```

### 방법 2: RAG-Evaluation-Dataset-KO 사용

```python
from experiments.run_experiments import ExperimentRunner

runner = ExperimentRunner()

# 실제 평가 데이터셋 사용
tables = runner.load_test_data("", use_dataset=True)
```

### 방법 3: PDF에서 테이블 추출

```python
from utils.pdf_table_extractor import PDFTableExtractor

extractor = PDFTableExtractor(output_dir="data/extracted_tables")

# PDF에서 테이블 추출
tables_info = extractor.extract_all_from_dataset(
    documents_csv="RAG-Evaluation-Dataset-KO/documents.csv",
    pdf_base_dir="RAG-Evaluation-Dataset-KO",
    save_to_excel=True
)

tables = [info['table_data'] for info in tables_info]
```

### 방법 4: 직접 데이터 로드

```python
import pandas as pd

# Excel 파일 로드
table = pd.read_excel("data/sample_tables/table_0.xlsx")

# CSV 파일 로드
table = pd.read_csv("data/dataset.csv")

# 여러 파일 로드
import os
tables = []
for file in os.listdir("data/sample_tables"):
    if file.endswith('.xlsx'):
        df = pd.read_excel(os.path.join("data/sample_tables", file))
        tables.append(df)
```

### 방법 5: 쿼리 및 Ground Truth 준비

```python
from utils.prepare_rag_queries import prepare_queries_from_dataset

# 데이터셋에서 쿼리와 ground truth 추출
queries, ground_truth = prepare_queries_from_dataset(
    rag_result_csv="RAG-Evaluation-Dataset-KO/rag_evaluation_result.csv",
    documents_csv="RAG-Evaluation-Dataset-KO/documents.csv"
)

print(f"쿼리 수: {len(queries)}")
print(f"Ground truth 매핑 수: {len(ground_truth)}")
```

---

## 데이터셋 메타데이터

### metadata.json 구조

```json
[
  {
    "table_id": "table_0",
    "filename": "table_0.xlsx",
    "shape": [4, 4],
    "columns": ["연도", "매출액(억원)", "순이익(억원)", "직원수"]
  },
  ...
]
```

### 메타데이터 생성

```python
from utils.download_datasets import DatasetDownloader

downloader = DatasetDownloader()

tables_info = [
    {
        'table_id': f'table_{i}',
        'filename': f'table_{i}.xlsx',
        'shape': list(table.shape),
        'columns': list(table.columns.tolist())
    }
    for i, table in enumerate(tables)
]

downloader.save_metadata(tables_info)
```

---

## 데이터셋 전처리

### 테이블 정규화

```python
import pandas as pd

def normalize_table(table: pd.DataFrame) -> pd.DataFrame:
    """테이블 정규화"""
    # 결측값 처리
    table = table.fillna('')
    
    # 데이터 타입 변환
    for col in table.columns:
        if table[col].dtype == 'object':
            # 숫자로 변환 가능한지 확인
            try:
                table[col] = pd.to_numeric(table[col], errors='ignore')
            except:
                pass
    
    return table

# 사용 예시
normalized_table = normalize_table(table)
```

### 테이블 검증

```python
def validate_table(table: pd.DataFrame) -> bool:
    """테이블 유효성 검증"""
    # 빈 테이블 확인
    if table.empty:
        return False
    
    # 최소 행/열 수 확인
    if table.shape[0] < 1 or table.shape[1] < 1:
        return False
    
    # 모든 행이 비어있는지 확인
    if table.isna().all().all():
        return False
    
    return True

# 사용 예시
if validate_table(table):
    print("유효한 테이블입니다")
else:
    print("유효하지 않은 테이블입니다")
```

---

## 실험에서 데이터셋 사용

### 실험 1: 파싱 성능 비교

```python
from experiments.run_experiments import ExperimentRunner

runner = ExperimentRunner()

# 데이터셋 로드
tables = runner.load_test_data("data/sample_tables")

# 실험 실행
results = runner.experiment_1_parsing_comparison(
    tables=tables,
    include_baselines=True
)
```

### 실험 2: RAG 성능 비교

```python
from experiments.run_experiments import ExperimentRunner
from utils.prepare_rag_queries import prepare_queries_from_dataset

runner = ExperimentRunner()

# 테이블 로드
tables = runner.load_test_data("", use_dataset=True)

# 쿼리 및 ground truth 준비
queries, ground_truth = prepare_queries_from_dataset()

# 실험 실행
results = runner.experiment_2_rag_comparison(
    tables=tables,
    queries=queries,
    ground_truth=ground_truth,
    include_baselines=True
)
```

---

## 데이터셋 다운로드 및 준비

### 1. RAG-Evaluation-Dataset-KO 다운로드

```bash
# Git으로 클론 (있는 경우)
git clone https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO

# 또는 직접 다운로드
# Hugging Face에서 다운로드: https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO
```

### 2. PDF 다운로드

```python
from utils.download_pdfs import download_pdfs_from_dataset

# 데이터셋의 PDF 다운로드
downloaded, failed = download_pdfs_from_dataset(
    documents_csv="RAG-Evaluation-Dataset-KO/documents.csv",
    output_base_dir="RAG-Evaluation-Dataset-KO"
)

print(f"다운로드 성공: {len(downloaded)}개")
print(f"다운로드 실패: {len(failed)}개")
```

### 3. 테이블 추출

```python
from utils.pdf_table_extractor import PDFTableExtractor

extractor = PDFTableExtractor(output_dir="data/extracted_tables")

# PDF에서 테이블 추출
tables_info = extractor.extract_all_from_dataset(
    documents_csv="RAG-Evaluation-Dataset-KO/documents.csv",
    pdf_base_dir="RAG-Evaluation-Dataset-KO",
    save_to_excel=True
)
```

---

## 참고 자료

- [RAG-Evaluation-Dataset-KO README](RAG-Evaluation-Dataset-KO/README.md)
- [Allganize RAG Leaderboard](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO)
- [공공데이터포털](https://www.data.go.kr/)
- [KOSIS 국가통계포털](https://kosis.kr/)
- [DART 전자공시시스템](https://dart.fss.or.kr/)




