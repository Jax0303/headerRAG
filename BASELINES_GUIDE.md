# 베이스라인 모델 설치 및 사용 가이드

이 문서는 HeaderRAG 프로젝트에서 사용하는 베이스라인 모델들의 설치 및 사용 방법을 설명합니다.

## 목차

1. [Table Transformer (TATR)](#table-transformer-tatr)
2. [Sato 시맨틱 타입 검출](#sato-시맨틱-타입-검출)
3. [TableRAG](#tablerag)
4. [Tab2KG](#tab2kg)

---

## Table Transformer (TATR)

### 개요

Microsoft에서 개발한 표 구조 인식 모델로, PDF 및 이미지에서 표를 감지하고 구조를 인식합니다.

### 설치 방법

```bash
# 1. 저장소 클론
git clone https://github.com/microsoft/table-transformer.git
cd table-transformer

# 2. Conda 환경 생성
conda env create -f environment.yml
conda activate tables-detr

# 3. 환경변수 설정 (선택적)
export TATR_REPO_PATH=/path/to/table-transformer
```

### 사용 방법

```python
from src.baselines import TATRParser
import pandas as pd

# 파서 초기화
parser = TATRParser(
    model_version="v1.1-pub",  # 또는 "v1.1-fin", "v1.1-all"
    repo_path="/path/to/table-transformer",  # 선택적
    device="cuda"  # 또는 "cpu"
)

# 표 이미지에서 구조 인식
result = parser.parse(table_image_path="path/to/table.png")

# 또는 DataFrame에서 시뮬레이션 모드 사용
table_data = pd.read_excel("table.xlsx")
result = parser.parse(table_data=table_data)

# 레이블링된 셀 형식으로 변환
labeled_cells = parser.to_labeled_cells(result)
```

### 모델 버전

- **v1.0**: 초기 버전 (PubTables-1M 기반)
- **v1.1-pub**: PubTables-1M 기반 개선 버전
- **v1.1-fin**: 금융 문서용 (FinTabNet.c 기반)
- **v1.1-all**: 혼합 학습 버전

### Hugging Face 모델 다운로드

```python
# Hugging Face에서 모델 다운로드
TATRParser.download_model(
    model_version="v1.1-pub",
    output_dir="models/tatr"
)
```

---

## Sato 시맨틱 타입 검출

### 개요

Megagon Labs에서 개발한 시맨틱 타입 검출 모델로, 표 컬럼의 의미적 타입을 78가지 중에서 자동으로 검출합니다.

### 설치 방법

```bash
# 1. 저장소 클론
git clone https://github.com/megagonlabs/sato.git
cd sato

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 환경변수 설정 (선택적)
export SATO_REPO_PATH=/path/to/sato
```

### 사용 방법

```python
from src.baselines import SatoSemanticTypeDetector
import pandas as pd

# 검출기 초기화
detector = SatoSemanticTypeDetector(
    repo_path="/path/to/sato",  # 선택적
    model_path="/path/to/pretrained_model"  # 선택적
)

# 테이블 로드
table_data = pd.read_excel("table.xlsx")

# 컬럼 타입 검출
column_types = detector.detect_column_types(table_data)

# 결과 확인
for col, sem_type in column_types.items():
    print(f"{col}: {sem_type}")

# 테이블에 시맨틱 타입 주석 추가
annotated_table = detector.annotate_table(table_data, column_types)
```

### 지원하는 시맨틱 타입

주요 타입:
- **날짜/시간**: `year`, `month`, `day`, `date`, `time`
- **위치**: `location`, `country`, `city`, `address`
- **금액**: `money`, `currency`, `price`, `cost`, `revenue`
- **수치**: `number`, `integer`, `percentage`, `count`
- **개인정보**: `person`, `email`, `phone`
- **기타**: 총 78가지 타입 지원

---

## TableRAG

### 개요

표 기반 RAG(Retrieval-Augmented Generation) 시스템으로, 표 데이터를 처리하여 질의응답을 수행합니다.

### 설치 방법

#### 옵션 1: ColBERT 기반 TableRAG

```bash
# 저장소 클론
git clone https://github.com/YuhangWuAI/tablerag.git
cd tablerag

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
export TABLERAG_REPO_PATH=/path/to/tablerag
```

#### 옵션 2: SQL 하이브리드 TableRAG

```bash
# 저장소 클론
git clone https://github.com/yxh-y/TableRAG.git
cd TableRAG

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
export TABLERAG_REPO_PATH=/path/to/TableRAG
```

### 사용 방법

```python
from src.baselines import TableRAGBaseline
import pandas as pd

# TableRAG 초기화
rag = TableRAGBaseline(
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    use_colbert=False,  # True면 ColBERT 사용 (저장소 필요)
    repo_path="/path/to/tablerag"  # 선택적
)

# 테이블 추가
table_data = pd.read_excel("table.xlsx")
rag.add_table(
    table_data=table_data,
    table_id="table_1",
    chunk_strategy="row"  # 또는 "cell", "hybrid"
)

# 인덱스 구축
rag.build_index()

# 질의 수행
results = rag.query(
    query_text="2023년 매출액은 얼마인가요?",
    top_k=5,
    return_context=True
)

# 답변 생성 (LLM 사용 가능)
answer = rag.answer(
    query_text="2023년 매출액은 얼마인가요?",
    top_k=5,
    use_llm=True,  # LLM 사용
    llm_model="gpt-4"  # 또는 다른 LLM 모델
)
```

### 청킹 전략

- **row**: 행 단위로 청킹 (기본값)
- **cell**: 셀 단위로 청킹
- **hybrid**: 행 + 셀 조합

---

## Tab2KG

### 개요

표를 Knowledge Graph(RDF)로 자동 변환하는 도구입니다.

### 설치 방법

```bash
# 1. 저장소 클론
git clone https://github.com/sgottsch/Tab2KG.git
cd Tab2KG

# 2. Java 설치 확인 (필요한 경우)
java -version

# 3. 환경변수 설정 (선택적)
export TAB2KG_REPO_PATH=/path/to/Tab2KG
```

### 사용 방법

```python
from src.baselines import Tab2KGBaseline
import pandas as pd

# 변환기 초기화
converter = Tab2KGBaseline(
    repo_path="/path/to/Tab2KG",  # 선택적
    use_semantic_profiles=True
)

# 테이블 로드
table_data = pd.read_excel("table.xlsx")

# NetworkX 그래프로 변환
nx_graph = converter.convert_to_networkx(table_data, "table_1")
print(f"노드 수: {len(nx_graph.nodes())}, 엣지 수: {len(nx_graph.edges())}")

# RDF 그래프로 변환
rdf_graph = converter.convert_to_rdf(table_data, "table_1")
print(f"트리플 수: {len(rdf_graph)}")

# RDF 파일로 저장
converter.save_rdf(
    table_data=table_data,
    table_id="table_1",
    output_path="output/table_1.ttl",
    format="turtle"  # 또는 "xml", "json-ld", "nt"
)

# 딕셔너리 형식으로 변환
graph_dict = converter.convert_to_dict(table_data, "table_1")
```

### 출력 형식

- **networkx**: NetworkX DiGraph 객체
- **rdf**: RDFlib Graph 객체
- **dict**: Python 딕셔너리

---

## 실험에서 베이스라인 사용하기

### 실험 1: 파싱 성능 비교

```python
from src.baselines import TATRParser, SatoSemanticTypeDetector
from src.parsing import LabeledTableParser, NaiveTableParser

# 베이스라인 파서
tatr_parser = TATRParser(model_version="v1.1-pub")
sato_detector = SatoSemanticTypeDetector()

# 기존 파서
labeled_parser = LabeledTableParser()
naive_parser = NaiveTableParser()

# 성능 비교
# ... 실험 코드 ...
```

### 실험 2: RAG 성능 비교

```python
from src.baselines import TableRAGBaseline, Tab2KGBaseline
from src.rag import KGRAGSystem, NaiveRAGSystem

# 베이스라인 RAG
tablerag = TableRAGBaseline()
tab2kg = Tab2KGBaseline()

# 기존 RAG 시스템
kg_rag = KGRAGSystem()
naive_rag = NaiveRAGSystem()

# 성능 비교
# ... 실험 코드 ...
```

---

## 문제 해결

### TATR 모델을 찾을 수 없음

환경변수 `TATR_REPO_PATH`를 설정하거나, `TATRParser` 초기화 시 `repo_path` 인자를 제공하세요.

### Sato 모델 로드 실패

Sato 저장소를 클론하고 사전 학습된 모델을 다운로드한 후, `model_path` 인자로 경로를 지정하세요.

### TableRAG ColBERT 오류

ColBERT를 사용하려면 TableRAG 저장소를 설치하고 `repo_path`를 설정해야 합니다. 또는 `use_colbert=False`로 설정하여 일반 SentenceTransformer를 사용하세요.

### Tab2KG Java 오류

Tab2KG의 일부 기능은 Java가 필요합니다. Java가 설치되어 있는지 확인하세요: `java -version`

---

## 참고 자료

- [Table Transformer GitHub](https://github.com/microsoft/table-transformer)
- [Sato GitHub](https://github.com/megagonlabs/sato)
- [TableRAG GitHub (ColBERT)](https://github.com/YuhangWuAI/tablerag)
- [TableRAG GitHub (SQL)](https://github.com/yxh-y/TableRAG)
- [Tab2KG GitHub](https://github.com/sgottsch/Tab2KG)

