# HeaderRAG: 테이블 파싱 및 RAG 실험 프레임워크

## 프로젝트 개요

한국 기업에서 사용하는 복잡한 표 데이터를 대상으로 한 RAG(Retrieval-Augmented Generation) 실험 프레임워크입니다.

## 주요 실험

1. **레이블링 기반 파싱 vs Naive 파싱 성능 비교**
   - 각 데이터셀, 헤더셀, 열 셀에 레이블을 부착한 파싱 방식
   - 레이블 없이 단순 파싱하는 방식
   - **베이스라인**: TATR (Table Transformer), Sato (시맨틱 타입 검출)

2. **KG 기반 RAG vs Naive 파싱 RAG 비교**
   - 테이블을 Knowledge Graph로 변환 후 RAG
   - Naive하게 테이블 파싱 후 RAG
   - **베이스라인**: TableRAG, Tab2KG

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
├── BASELINES_GUIDE.md     # 베이스라인 모델 가이드
├── requirements.txt
└── README.md
```

## 빠른 시작 (주피터 노트북)

가장 편리한 방법은 주피터 노트북을 사용하는 것입니다:

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

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 기본 실험 실행

```bash
# 전체 실험 실행 (베이스라인 포함)
python run_full_experiment.py

# 모든 데이터셋으로 실험 실행
python experiments/run_all_datasets_experiment.py

# 특정 데이터셋으로 실험
python experiments/run_multi_dataset_experiments.py --datasets rag_eval_ko
```

### 데이터셋 다운로드

```bash
# 주요 데이터셋 자동 다운로드
python download_datasets_now.py
```

베이스라인 모델 사용:
베이스라인 모델 설치 및 사용 방법은 [BASELINES_GUIDE.md](BASELINES_GUIDE.md)를 참조하세요.

베이스라인 포함 실험 실행:

**방법 1: 명령줄 실행 (추천)**
```bash
# 전체 실험 실행 (베이스라인 포함)
python experiments/run_comparison_experiments.py --include_baselines

# 특정 실험만 실행
python experiments/run_comparison_experiments.py --experiment 1 --include_baselines

# 실제 데이터셋 사용
python experiments/run_comparison_experiments.py --use_dataset --include_baselines
```

**방법 2: Python 스크립트**
```python
from experiments.run_experiments import ExperimentRunner

runner = ExperimentRunner()
tables = runner.load_test_data("data/sample_tables")

# 실험 1: 베이스라인 포함
results = runner.experiment_1_parsing_comparison(tables, include_baselines=True)

# 실험 2: 베이스라인 포함
queries = ["2023년 매출액은?", ...]
ground_truth = {...}
results = runner.experiment_2_rag_comparison(tables, queries, ground_truth, include_baselines=True)
```

자세한 내용은 [COMPARISON_EXPERIMENTS_GUIDE.md](COMPARISON_EXPERIMENTS_GUIDE.md)를 참조하세요.

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

## 📚 문서

- **[프로젝트 전체 요약](PROJECT_SUMMARY.md)** - 🚀 **시작하기 전에 읽어보세요!**
- [빠른 시작](QUICKSTART.md) - 빠른 실험 가이드
- [데이터셋 정보](DATASET_INFO.md) - 지원 데이터셋 상세 정보
- [프로젝트 구조](PROJECT_STRUCTURE.md) - 프로젝트 구조 설명
- [베이스라인 모델 가이드](BASELINES_GUIDE.md) - 베이스라인 모델 사용법
- [비교 실험 가이드](COMPARISON_EXPERIMENTS_GUIDE.md) - 실험 실행 방법
- [실험 로드맵](EXPERIMENT_ROADMAP.md) - 실험 계획 및 진행 상황
- [실험 결과](EXPERIMENT_RESULTS.md) - 최신 실험 결과

