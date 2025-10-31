# 빠른 시작 가이드

## 1. 설치

```bash
# 저장소 클론 (또는 현재 디렉토리 사용)
cd headerRAG

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

## 2. 샘플 데이터 생성

```bash
python utils/download_datasets.py
```

이 명령어는 `data/sample_tables/` 디렉토리에 샘플 테이블 10개를 생성합니다.

## 3. 기본 사용법

### 3.1 레이블링 기반 파싱

```python
import pandas as pd
from src.parsing.labeled_parser import LabeledTableParser

# 테이블 로드
table = pd.read_excel("data/sample_tables/table_0.xlsx")

# 파서 생성 및 파싱
parser = LabeledTableParser()
labeled_cells = parser.parse(table)

# 구조화된 형식으로 변환
structured = parser.to_structured_format(labeled_cells)
print(structured)
```

### 3.2 Naive 파싱

```python
from src.parsing.naive_parser import NaiveTableParser

parser = NaiveTableParser()
naive_result = parser.parse(table)
text_format = parser.to_text_format(table)
print(text_format)
```

### 3.3 Knowledge Graph 변환

```python
from src.kg.table_to_kg import TableToKGConverter

converter = TableToKGConverter(use_labeled_parsing=True)
graph = converter.convert(table, table_id="table_0")

# 그래프를 텍스트로 변환
graph_text = converter.graph_to_text(graph, format="triples")
print(graph_text)
```

### 3.4 KG 기반 RAG

```python
from src.rag.kg_rag import KGRAGSystem
import pandas as pd

# RAG 시스템 생성
rag = KGRAGSystem(use_labeled_parsing=True)

# 테이블 추가
table1 = pd.read_excel("data/sample_tables/table_0.xlsx")
rag.add_table(table1, "table_0")

# 인덱스 구축
rag.build_index()

# 검색
results = rag.retrieve("2023년 매출액은?", top_k=3)
for result in results:
    print(f"Table: {result['table_id']}, Score: {result['score']}")
    print(result['graph_text'][:200])
```

### 3.5 Naive RAG

```python
from src.rag.naive_rag import NaiveRAGSystem

rag = NaiveRAGSystem()
rag.add_table(table1, "table_0")
rag.build_index()

results = rag.retrieve("2023년 매출액은?", top_k=3)
for result in results:
    print(f"Table: {result['table_id']}, Score: {result['score']}")
```

## 4. 실험 실행

```bash
# 전체 실험 실행
python experiments/run_experiments.py
```

실험 결과는 `results/` 디렉토리에 JSON 형식으로 저장됩니다.

## 5. 데이터셋 다운로드

### 공공데이터포털에서 데이터 수집

1. https://www.data.go.kr 접속
2. 원하는 데이터셋 검색 및 다운로드
3. `data/` 디렉토리에 저장

### DART 데이터 수집

1. https://dart.fss.or.kr 접속
2. 전자공시시스템에서 재무제표 다운로드
3. Excel 형식으로 변환하여 저장

## 6. 커스터마이징

### 다른 임베딩 모델 사용

```python
# 한국어 특화 모델 사용
rag = KGRAGSystem(
    embedding_model="jhgan/ko-sroberta-multitask",
    use_labeled_parsing=True
)
```

### 평가 메트릭 추가

`src/evaluation/metrics.py` 파일을 수정하여 새로운 평가 메트릭을 추가할 수 있습니다.

## 7. 문제 해결

### Import 오류
- Python 경로가 올바르게 설정되었는지 확인
- `sys.path.append()` 사용 필요 시 `experiments/run_experiments.py` 참고

### 메모리 부족
- 대규모 데이터셋의 경우 배치 처리 사용
- FAISS 인덱스 대신 작은 인덱스 사용

### 한국어 인코딩 오류
- 파일 읽기/쓰기 시 `encoding='utf-8'` 사용 확인

