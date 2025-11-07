# HeaderRAG: 테이블 파싱 및 RAG 실험 프레임워크

## 프로젝트 개요

한국 기업에서 사용하는 복잡한 표 데이터를 대상으로 한 RAG(Retrieval-Augmented Generation) 실험 프레임워크입니다.

### 핵심 목표
- 레이블링 기반 표 파싱의 효과성 검증
- Knowledge Graph 기반 RAG의 성능 향상 검증
- 베이스라인 모델과의 성능 비교

## 주요 실험

1. **레이블링 기반 파싱 vs Naive 파싱 성능 비교**
   - 각 데이터셀, 헤더셀, 열 셀에 레이블을 부착한 파싱 방식
   - 레이블 없이 단순 파싱하는 방식
   - **베이스라인**: TATR (Table Transformer), Sato (시맨틱 타입 검출)

2. **KG 기반 RAG vs Naive 파싱 RAG 비교**
   - 테이블을 Knowledge Graph로 변환 후 RAG
   - Naive하게 테이블 파싱 후 RAG
   - **베이스라인**: TableRAG, Tab2KG

## 프로젝트 구조

```
headerRAG/
├── data/                    # 데이터셋 저장
├── models/                  # 모델 저장
├── src/
│   ├── parsing/            # 파싱 모듈
│   │   ├── labeled_parser.py
│   │   └── naive_parser.py
│   ├── baselines/          # 베이스라인 모델
│   │   ├── table_structure/  # TATR 등
│   │   ├── semantic/         # Sato 등
│   │   ├── rag/              # TableRAG 등
│   │   └── kg/               # Tab2KG 등
│   ├── kg/                 # Knowledge Graph 변환
│   │   └── table_to_kg.py
│   ├── rag/                # RAG 시스템
│   │   ├── kg_rag.py
│   │   └── naive_rag.py
│   └── evaluation/         # 평가 모듈
│       └── metrics.py
├── experiments/            # 실험 스크립트
├── utils/                  # 유틸리티
├── requirements.txt
└── README.md
```

## 빠른 시작

### 1. 설치

```bash
# 저장소 클론 (또는 현재 디렉토리 사용)
cd headerRAG

# 가상환경 생성 및 활성화 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 샘플 데이터 생성

```bash
python download_datasets_now.py
```

### 3. 기본 사용법

#### 레이블링 기반 파싱

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

#### KG 기반 RAG

```python
from src.rag.kg_rag import KGRAGSystem

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
```

### 4. 실험 실행

```bash
# 전체 실험 실행 (베이스라인 포함)
python run_full_experiment.py

# 모든 데이터셋으로 실험 실행
python experiments/run_all_datasets_experiment.py

# 특정 데이터셋으로 실험
python experiments/run_multi_dataset_experiments.py --datasets rag_eval_ko
```

### 5. 주피터 노트북 사용 (추천)

```bash
# 주피터 노트북 설치
pip install jupyter

# 노트북 실행
cd experiments
jupyter notebook experiments.ipynb
```

노트북에서는 다음을 수행할 수 있습니다:
- ✅ 실험 1, 2, 3을 순차적으로 실행
- ✅ 실시간으로 결과 확인 및 시각화
- ✅ 인터랙티브한 데이터 분석
- ✅ 종합 보고서 자동 생성

## 지원 데이터셋

### 주요 데이터셋

1. **RAG-Evaluation-Dataset-KO** (활성 사용 중)
   - 한국어 RAG 평가 데이터셋 (Allganize)
   - 26개 테이블, 300개 질문
   - 5개 도메인: finance, public, medical, law, commerce
   - 자동 로드 지원

2. **TabRecSet** (Figshare)
   - 대규모 표 데이터셋
   - 실제 환경(인 와일드) 테이블 인식용

3. **WTW-Dataset** (GitHub)
   - 14,581개 이미지
   - 7가지 도전적인 케이스 포함
   - XML 형식 지원

4. **PubTables-1M** (Microsoft Research)
   - 약 100만 개의 표
   - 과학 논문에서 추출
   - Hugging Face 지원

자세한 데이터셋 정보는 [DATASET_INFO.md](DATASET_INFO.md)를 참조하세요.

## 문서

### 가이드 문서
- **[빠른 시작 가이드](QUICKSTART.md)** - 상세한 사용법 및 예제
- **[베이스라인 모델 가이드](BASELINES_GUIDE.md)** - 베이스라인 모델 설치 및 사용법
- **[비교 실험 가이드](COMPARISON_EXPERIMENTS_GUIDE.md)** - 실험 실행 방법
- **[데이터셋 정보](DATASET_INFO.md)** - 지원 데이터셋 상세 정보

### 설명 문서
- **[라벨링 방식 설명](LABELING_EXPLANATION.md)** - 라벨링 시스템 상세 설명
- **[사이클 기능 가이드](experiments/CYCLE_USAGE.md)** - 실험 사이클 사용법

### 계획 및 로드맵
- **[실험 로드맵](EXPERIMENT_ROADMAP.md)** - 실험 설계 및 실행 계획
- **[개선 로드맵](IMPROVEMENT_ROADMAP.md)** - 라벨링 시스템 개선 계획

### 결과 문서
- **[실험 결과](EXPERIMENT_RESULTS.md)** - 최신 실험 결과

## 추가 실험 아이디어

1. **다양한 표 구조에 따른 성능 비교**
   - 단순 표 vs 중첩 헤더 표 vs 병합 셀 표

2. **도메인별 성능 평가**
   - 금융 데이터 vs 의료 데이터 vs 교육 데이터

3. **노이즈 수준에 따른 성능 비교**
   - 완벽한 표 vs 오탈자 포함 vs 누락된 값

4. **임베딩 모델 비교**
   - 한국어 특화 모델 (KoBERT, KoGPT) vs 다국어 모델

5. **파싱 전처리 기법 비교**
   - OCR 후처리 vs 직접 파싱
   - 표 구조 인식 정확도가 RAG 성능에 미치는 영향

6. **질의 유형별 성능 분석**
   - 단순 조회 질의 vs 집계 질의 vs 추론 질의

## 문제 해결

### Import 오류
- Python 경로가 올바르게 설정되었는지 확인
- `sys.path.append()` 사용 필요 시 `experiments/run_experiments.py` 참고

### 메모리 부족
- 대규모 데이터셋의 경우 배치 처리 사용
- FAISS 인덱스 대신 작은 인덱스 사용

### 한국어 인코딩 오류
- 파일 읽기/쓰기 시 `encoding='utf-8'` 사용 확인

## 참고 자료

- [RAG-Evaluation-Dataset-KO](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO)
- [Table Transformer](https://github.com/microsoft/table-transformer)
- [Sato](https://github.com/megagonlabs/sato)
- [TableRAG](https://github.com/YuhangWuAI/tablerag)
- [Tab2KG](https://github.com/sgottsch/Tab2KG)
