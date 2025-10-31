# 프로젝트 구조

## 디렉토리 구조

```
headerRAG/
├── data/                          # 데이터셋 저장 디렉토리 (생성됨)
│   └── sample_tables/            # 샘플 테이블 (utils/download_datasets.py로 생성)
│
├── src/                          # 소스 코드
│   ├── parsing/                  # 테이블 파싱 모듈
│   │   ├── __init__.py
│   │   ├── labeled_parser.py     # 레이블링 기반 파서
│   │   └── naive_parser.py       # Naive 파서
│   │
│   ├── kg/                       # Knowledge Graph 모듈
│   │   ├── __init__.py
│   │   └── table_to_kg.py        # 테이블 -> KG 변환기
│   │
│   ├── rag/                      # RAG 시스템 모듈
│   │   ├── __init__.py
│   │   ├── kg_rag.py             # KG 기반 RAG 시스템
│   │   └── naive_rag.py          # Naive 파싱 기반 RAG 시스템
│   │
│   └── evaluation/               # 평가 모듈
│       ├── __init__.py
│       └── metrics.py            # 평가 메트릭 (Precision, Recall, F1, MRR, ROUGE 등)
│
├── experiments/                  # 실험 스크립트
│   ├── run_experiments.py        # 실험 실행 메인 스크립트
│   └── EXPERIMENT_PLAN.md        # 상세 실험 계획서
│
├── utils/                        # 유틸리티
│   ├── __init__.py
│   └── download_datasets.py      # 데이터셋 다운로드 및 샘플 생성
│
├── results/                      # 실험 결과 저장 디렉토리 (자동 생성)
│
├── README.md                     # 프로젝트 개요 및 사용법
├── QUICKSTART.md                 # 빠른 시작 가이드
├── PROJECT_STRUCTURE.md          # 이 파일
├── requirements.txt              # Python 의존성 패키지
└── .gitignore                    # Git 무시 파일 목록
```

## 주요 모듈 설명

### 1. Parsing 모듈 (`src/parsing/`)

#### `labeled_parser.py`
- **목적**: 각 셀에 레이블(헤더, 데이터, 행 헤더, 열 헤더 등)을 부착하여 구조화된 파싱 수행
- **주요 클래스**: `LabeledTableParser`, `CellLabel`
- **기능**:
  - 헤더 자동 감지
  - 병합 셀 감지
  - 시맨틱 레이블 추출 (연도, 금액, 비율 등)
  - 구조화된 형식으로 변환

#### `naive_parser.py`
- **목적**: 레이블링 없이 단순하게 테이블 파싱
- **주요 클래스**: `NaiveTableParser`
- **기능**:
  - 기본 DataFrame 파싱
  - 텍스트 형식 변환
  - 값 추출

### 2. Knowledge Graph 모듈 (`src/kg/`)

#### `table_to_kg.py`
- **목적**: 테이블을 Knowledge Graph로 변환
- **주요 클래스**: `TableToKGConverter`
- **기능**:
  - NetworkX 그래프 변환
  - RDF 그래프 변환
  - 그래프를 텍스트 형식으로 변환 (RAG용)
  - 레이블링 기반 및 Naive 파싱 지원

### 3. RAG 모듈 (`src/rag/`)

#### `kg_rag.py`
- **목적**: Knowledge Graph 기반 RAG 시스템
- **주요 클래스**: `KGRAGSystem`
- **기능**:
  - 테이블을 KG로 변환하여 추가
  - 임베딩 인덱스 구축 (FAISS)
  - 쿼리 기반 검색
  - 서브그래프 추출
  - 컨텍스트 생성

#### `naive_rag.py`
- **목적**: Naive 파싱 기반 RAG 시스템
- **주요 클래스**: `NaiveRAGSystem`
- **기능**:
  - 테이블을 텍스트로 변환하여 추가
  - 임베딩 인덱스 구축 (FAISS)
  - 쿼리 기반 검색
  - 컨텍스트 생성
  - 간단한 답변 추출

### 4. Evaluation 모듈 (`src/evaluation/`)

#### `metrics.py`
- **목적**: RAG 시스템 평가 메트릭
- **주요 클래스**: `RAGEvaluator`
- **평가 메트릭**:
  - 검색 성능: Precision@K, Recall@K, F1, MRR
  - 생성 성능: ROUGE-1/2/L
  - 답변 정확도: Exact Match, Partial Match
  - 파싱 품질: 헤더 감지 정확도, 셀 추출 정확도
  - 시스템 비교: 개선율 계산

### 5. Experiments 모듈 (`experiments/`)

#### `run_experiments.py`
- **목적**: 전체 실험 실행 및 결과 저장
- **주요 클래스**: `ExperimentRunner`
- **실험 유형**:
  - 실험 1: 레이블링 기반 파싱 vs Naive 파싱 성능 비교
  - 실험 2: KG 기반 RAG vs Naive 파싱 RAG 비교

#### `EXPERIMENT_PLAN.md`
- 상세한 실험 계획서
- 데이터셋 추천 및 수집 가이드
- 추가 실험 아이디어 (10가지)

### 6. Utils 모듈 (`utils/`)

#### `download_datasets.py`
- **목적**: 데이터셋 다운로드 및 샘플 테이블 생성
- **주요 클래스**: `DatasetDownloader`
- **기능**:
  - 공공데이터 다운로드
  - 샘플 테이블 생성 (단순 표, 중첩 헤더 표, 병합 셀 표)
  - 메타데이터 저장

## 데이터 흐름

```
테이블 데이터 (Excel/CSV)
    ↓
[파싱 단계]
    ├─→ 레이블링 기반 파싱 → 구조화된 셀 정보
    └─→ Naive 파싱 → 단순 텍스트
    ↓
[KG 변환 단계] (레이블링 기반만)
    └─→ Knowledge Graph (NetworkX/RDF)
    ↓
[RAG 구축 단계]
    ├─→ KG RAG: 그래프 → 텍스트 → 임베딩 → FAISS 인덱스
    └─→ Naive RAG: 텍스트 → 임베딩 → FAISS 인덱스
    ↓
[검색 단계]
    ├─→ 쿼리 임베딩
    ├─→ 유사도 검색
    └─→ 관련 테이블/그래프 반환
    ↓
[평가 단계]
    └─→ 메트릭 계산 및 비교
```

## 주요 의존성

- **데이터 처리**: pandas, numpy, openpyxl
- **테이블 파싱**: pdfplumber, tabula-py, camelot-py
- **Knowledge Graph**: networkx, rdflib
- **RAG 및 임베딩**: sentence-transformers, faiss-cpu, langchain
- **한국어 NLP**: transformers, torch
- **평가**: scikit-learn, rouge-score, nltk
- **시각화**: matplotlib, seaborn, plotly

## 확장 가능성

### 새로운 파서 추가
- `src/parsing/` 디렉토리에 새로운 파서 클래스 추가
- 공통 인터페이스 구현 필요

### 새로운 RAG 시스템 추가
- `src/rag/` 디렉토리에 새로운 RAG 클래스 추가
- `retrieve()`, `build_index()` 메서드 구현 필요

### 새로운 평가 메트릭 추가
- `src/evaluation/metrics.py`에 새로운 메서드 추가
- `RAGEvaluator` 클래스 확장

### 새로운 실험 추가
- `experiments/run_experiments.py`에 새로운 실험 메서드 추가
- 또는 별도 스크립트로 분리

