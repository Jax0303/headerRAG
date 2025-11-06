# 실험 설계 및 실행 로드맵

이 문서는 HeaderRAG 프로젝트의 실험 설계 및 실행 단계별 계획을 담고 있습니다.

## 📋 목차

1. [Phase 1: 실험 메트릭 확정 및 검증](#phase-1-실험-메트릭-확정-및-검증)
2. [Phase 2: 파일럿 실험](#phase-2-파일럿-실험)
3. [Phase 3: 전체 규모 실험](#phase-3-전체-규모-실험)
4. [Phase 4: 결과 분석 및 시각화](#phase-4-결과-분석-및-시각화)

---

## 🎯 Phase 1: 실험 메트릭 확정 및 검증 (1-2주)

### 1.1 표 파싱 평가 메트릭 구현 ✅

#### 핵심 메트릭

**TEDS (Tree Edit Distance-based Similarity)**
- 표 구조 정확도의 학술 표준 메트릭
- HTML 트리 구조로 변환 후 트리 편집 거리 계산
- 0.90 이상이면 우수, 0.80 이하면 문제 있음
- 단점: 행 누락이 열 누락보다 높은 페널티 (트리 구조 특성)

**GriTS (Grid Table Similarity)** ⭐ 가장 추천
- 표를 2D 배열로 해석, 행/열 동등 처리
- 세 가지 세부 메트릭:
  - **GriTS-Content**: 셀 텍스트 편집 거리
  - **GriTS-Topology**: rowspan/colspan 일치도
  - **GriTS-Location**: 셀 공간 좌표 IoU
- 0.85-0.95가 우수 점수

**헤더 감지 정확도**
- Precision, Recall, F1 for header cells
- 병합 셀 처리 정확도

#### 구현 상태

✅ `src/evaluation/parsing_metrics.py`에 구현 완료
- `ParsingMetrics` 클래스
- `calculate_teds()`: TEDS 계산
- `calculate_grits()`: GriTS 계산 (Content, Topology, Location)
- `calculate_header_metrics()`: 헤더 감지 정확도

#### 사용 예시

```python
from src.evaluation.parsing_metrics import ParsingMetrics

metrics = ParsingMetrics()

# GriTS 계산
grits_results = metrics.calculate_grits(predicted_table, ground_truth_table)
# {'grits_content': 0.92, 'grits_topology': 0.88, 'grits_location': 0.90, 'grits_overall': 0.90}

# 헤더 메트릭
header_results = metrics.calculate_header_metrics(predicted_structure, ground_truth_structure)
# {'header_precision': 0.95, 'header_recall': 0.93, 'header_f1': 0.94, 'merged_cell_accuracy': 0.91}

# 종합 평가
all_metrics = metrics.evaluate_parsing(predicted_table, ground_truth_table)
```

---

### 1.2 RAG 평가 메트릭 구현 ✅

#### RAGAS 프레임워크 필수 메트릭

**검색(Retrieval) 평가:**
- **Context Precision**: 관련 문서가 상위 순위에 있는지 (0-1)
- **Context Recall**: Ground truth 대비 검색된 문서 비율
- **Context Relevance**: 검색된 문서와 질문의 관련성

**생성(Generation) 평가:**
- **Faithfulness**: 답변이 검색된 컨텍스트에 충실한지 (0-1)
  - 답변을 개별 주장(claims)으로 분해
  - 각 주장이 컨텍스트에서 지원되는지 검증
  - 공식: Faithfulness = |지원된 주장| / |전체 주장|
- **Answer Relevance**: 답변과 질문의 관련성
  - 답변으로부터 역으로 질문 생성
  - 원래 질문과 cosine similarity 계산
- **Answer Correctness**: Ground truth 대비 정답 정확도
  - Factual Correctness + Answer Similarity 가중 합
- **Answer Hallucination**: 컨텍스트에 없는 정보 생성 비율

**권장 조합**: Faithfulness + Factual Correctness가 전문가 평가와 가장 일치

#### 구현 상태

✅ `src/evaluation/ragas_metrics.py`에 구현 완료
- `RAGASMetrics` 클래스
- 모든 RAGAS 메트릭 구현
- RAGAS 라이브러리 통합 (선택적)

#### 사용 예시

```python
from src.evaluation.ragas_metrics import RAGASMetrics

ragas_metrics = RAGASMetrics()

# 종합 RAG 평가
results = ragas_metrics.evaluate_rag(
    question="매출액은 얼마인가요?",
    answer="2023년 매출액은 100억원입니다.",
    contexts=["매출액: 100억원", "연도: 2023"],
    ground_truth_answer="100억원",
    ground_truth_contexts=["매출액: 100억원"]
)
# {'faithfulness': 0.95, 'answer_relevancy': 0.92, 'context_precision': 1.0, ...}
```

---

### 1.3 표 복잡도 메트릭 정의 ✅

#### 구조적 복잡도 지표

- **병합 셀 비율**: merged cells / total cells
- **헤더 계층 깊이**: 중첩된 헤더 레벨 수
- **Nested sub-table 수**: 테이블 내 분리된 영역 수
- **행/열 비대칭성**: |rows - cols| / max(rows, cols)
- **빈 셀 비율**: empty cells / total cells

#### 시각적 복잡도 지표 (WTW 데이터셋 기반)

- 기울기 각도
- 텍스트 겹침 정도
- 테두리 완전성

#### 복잡도 등급 분류

- **Low**: 병합 셀 <10%, 헤더 1레벨
- **Medium**: 병합 셀 10-30%, 헤더 2레벨
- **High**: 병합 셀 >30%, 헤더 3+ 레벨, nested sub-tables

#### 구현 상태

✅ `src/evaluation/complexity_metrics.py`에 구현 완료
- `ComplexityMetrics` 클래스
- 구조적/시각적 복잡도 계산
- 복잡도 등급 분류

#### 사용 예시

```python
from src.evaluation.complexity_metrics import ComplexityMetrics

complexity = ComplexityMetrics()

# 종합 복잡도 계산
results = complexity.calculate_complexity(table_structure)
# {
#   'merged_cell_ratio': 0.15,
#   'header_depth': 2.0,
#   'structural_complexity_score': 0.45,
#   'complexity_level': 'medium',
#   ...
# }
```

---

## 🧪 Phase 2: 파일럿 실험 (2-3주)

### 2.1 소규모 데이터셋 실험 ✅

#### 데이터 샘플링

- **PubTables-1M**: 100개 (복잡도별 균등 샘플링)
- **TabRecSet**: 50개 (영문/중문 각 25개)
- **KorWikiTabular**: 50개

#### 실험 구성

**실험 1A: 파싱 성능**
- 레이블링 기반 파싱 (HeaderRAG)
- TATR 베이스라인
- Sato 베이스라인 (컬럼 타입 검출)
- Naive 파싱 (pandas)
- 평가: GriTS, TEDS, 헤더 정확도

**실험 2A: RAG 성능**
- KG 기반 RAG (HeaderRAG)
- TableRAG 베이스라인
- Tab2KG 베이스라인
- Naive RAG (단순 텍스트)
- 평가: Faithfulness, Answer Relevancy, Context Precision

**실험 3A: 복잡도 분석 (파일럿)**
- Low complexity: 20개
- Medium complexity: 20개
- High complexity: 20개
- 각 복잡도별 실험 1+2 반복

#### 구현 상태

✅ `experiments/pilot_experiments.py`에 구현 완료
- `PilotExperimentRunner` 클래스
- 계층적 샘플링 (`stratified_sampling`)
- 실험 1A, 2A, 3A 구현

#### 실행 방법

```bash
# 전체 파일럿 실험 실행
python experiments/run_pilot_experiments.py --experiment all --dataset pubtables1m --max_tables 100

# 특정 실험만 실행
python experiments/run_pilot_experiments.py --experiment 1a --dataset korwiki_tabular
```

---

### 2.2 Ablation Study 설계 ✅

#### 실험 1 Ablation

- **Baseline (Full)**: 레이블링 + 헤더 감지 + 병합 셀 처리
- **Ablation 1**: 레이블링 - 헤더 감지
- **Ablation 2**: 레이블링 - 병합 셀 처리
- **Ablation 3**: 헤더 감지만 (레이블링 제거)
- **Ablation 4**: Naive 파싱

#### 실험 2 Ablation

- **Baseline (Full)**: KG 변환 + 구조 정보 + 컨텍스트 임베딩
- **Ablation 1**: KG - 구조 정보 (노드만)
- **Ablation 2**: KG - 컨텍스트 임베딩
- **Ablation 3**: 단순 그래프 (레이블 없음)
- **Ablation 4**: Naive RAG

#### 통계 분석

- Paired t-test로 각 컴포넌트의 유의성 검증
- p-value < 0.05면 통계적 유의미

#### 구현 상태

✅ `experiments/pilot_experiments.py`에 구현 완료
- `ablation_study_parsing()` 메서드
- 통계 분석 통합

#### 실행 방법

```bash
python experiments/run_pilot_experiments.py --experiment ablation --dataset pubtables1m
```

---

## 📊 Phase 3: 전체 규모 실험 (4-6주)

### 3.1 데이터셋 확장

#### 실험 규모

- **PubTables-1M**: 1,000-5,000개 (복잡도 계층 샘플링)
- **TabRecSet**: 전체 (38,177개) 또는 대표 샘플 5,000개
- **KorWikiTabular**: 1,000-2,000개
- **FinTabNet**: 500개 (금융 도메인 특화)
- **WTW**: 500개 (극단 케이스)

#### 계층적 샘플링 전략

```python
# 복잡도별 균등 샘플링
stratified_sample = (
    df.groupby('complexity_level')
    .sample(n=samples_per_level, random_state=42)
)
```

---

### 3.2 교차 검증 및 통계 분석 ✅

#### K-Fold Cross Validation (k=5)

- 데이터를 5개 fold로 분할
- 각 fold에서 모든 실험 반복
- 평균 및 표준편차 보고

#### 통계 검증

- **Paired t-test**: HeaderRAG vs 각 베이스라인
- **Wilcoxon signed-rank test**: 비정규분포 시
- **Bonferroni correction**: 다중 비교 보정

#### 효과 크기 계산

- **Cohen's d**: 실질적 성능 차이 측정
- d > 0.8이면 large effect

#### 구현 상태

✅ `experiments/statistical_analysis.py`에 구현 완료
- `StatisticalAnalyzer` 클래스
- K-Fold Cross Validation
- Paired t-test, Wilcoxon test
- Cohen's d 계산
- Bonferroni correction

#### 사용 예시

```python
from experiments.statistical_analysis import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()

# 두 방법 비교
comparison = analyzer.compare_methods(
    method1_scores=[0.92, 0.93, 0.91, ...],
    method2_scores=[0.85, 0.86, 0.84, ...],
    method1_name="HeaderRAG",
    method2_name="Baseline"
)
# {'test': {'p_value': 0.001, 'is_significant': True}, 'effect_size': {'cohens_d': 1.2}, ...}
```

---

### 3.3 도메인별 분석

#### 데이터셋별 성능 비교

| 데이터셋 | 도메인 | 언어 | 복잡도 |
|:--------|:------|:-----|:------|
| PubTables-1M | 과학 | 영문 | Medium |
| FinTabNet | 금융 | 영문 | High |
| TabRecSet | 일반 | 영/중 | Mixed |
| KorWikiTabular | 백과사전 | 한글 | Low-Medium |
| WTW | 실제환경 | 영/중 | Extreme |

#### 분석 질문

- 한국어 표에서 HeaderRAG가 더 효과적인가?
- 금융 도메인처럼 복잡한 표에서 성능 향상이 더 큰가?
- 극단 케이스(WTW)에서 강건성이 유지되는가?

---

## 📈 Phase 4: 결과 분석 및 시각화 (2주)

### 4.1 성능 비교 테이블 ✅

#### 실험 1 결과 예시

| Method | GriTS ↑ | TEDS ↑ | Header F1 ↑ | Cell Acc ↑ |
|:-------|:--------|:-------|:------------|:-----------|
| Naive Parsing | 0.72 | 0.68 | 0.65 | 0.78 |
| TATR | 0.89 | 0.86 | 0.82 | 0.91 |
| Sato | 0.85 | 0.83 | 0.88 | 0.87 |
| **HeaderRAG (Ours)** | **0.93** | **0.91** | **0.94** | **0.95** |

#### 실험 2 결과 예시

| Method | Faithfulness ↑ | Answer Rel ↑ | Context Prec ↑ | F1 ↑ |
|:-------|:---------------|:-------------|:---------------|:-----|
| Naive RAG | 0.68 | 0.72 | 0.65 | 0.71 |
| TableRAG | 0.79 | 0.81 | 0.78 | 0.82 |
| Tab2KG | 0.82 | 0.83 | 0.81 | 0.84 |
| **KG-RAG (Ours)** | **0.88** | **0.89** | **0.87** | **0.91** |

#### 구현 상태

✅ `experiments/result_analyzer.py`에 구현 완료
- `ResultAnalyzer` 클래스
- `create_performance_table()`: 성능 비교 테이블 생성

---

### 4.2 복잡도별 성능 분석 ✅

#### 차트 생성

```python
# 복잡도별 성능 곡선
plt.plot(complexity_levels, your_method_scores, label='HeaderRAG')
plt.plot(complexity_levels, baseline_scores, label='Baseline')
plt.xlabel('Table Complexity')
plt.ylabel('GriTS Score')
plt.legend()
```

**기대 결과**: 복잡도가 높을수록 HeaderRAG와 베이스라인 간 격차 증가

#### 구현 상태

✅ `experiments/result_analyzer.py`에 구현 완료
- `plot_complexity_analysis()`: 복잡도별 성능 차트 생성

---

### 4.3 오류 분석 (Error Analysis) ✅

#### 정성 분석

- HeaderRAG가 실패한 케이스 100개 샘플링
- 실패 원인 분류:
  - OCR 오류
  - 극단적 병합 셀
  - Nested sub-table 복잡성
  - 도메인 특화 용어

#### 개선 방향 도출

- 각 오류 유형별 비율 계산
- 향후 연구에서 해결 가능한 방향 제시

#### 구현 상태

✅ `experiments/result_analyzer.py`에 구현 완료
- `analyze_errors()`: 오류 분석 및 시각화

---

## 🚀 빠른 시작

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

추가 설치 필요:
```bash
pip install zss ragas datasets scipy
```

### 2. 파일럿 실험 실행

```bash
# 전체 실험
python experiments/run_pilot_experiments.py --experiment all --dataset pubtables1m --max_tables 100

# 파싱 실험만
python experiments/run_pilot_experiments.py --experiment 1a --dataset korwiki_tabular

# Ablation Study
python experiments/run_pilot_experiments.py --experiment ablation --dataset pubtables1m
```

### 3. 결과 분석

```python
from experiments.result_analyzer import ResultAnalyzer

analyzer = ResultAnalyzer()

# 성능 비교 테이블 생성
results = {...}  # 실험 결과 로드
performance_table = analyzer.create_performance_table(
    results,
    metrics=['grits_overall', 'header_f1', 'teds'],
    output_path='results/performance_table.csv'
)

# 복잡도별 분석 차트
analyzer.plot_complexity_analysis(
    complexity_results,
    metric='grits_overall',
    output_path='results/complexity_analysis.png'
)
```

---

## 📝 체크리스트

### Phase 1: 메트릭 구현 ✅
- [x] TEDS 구현
- [x] GriTS 구현
- [x] 헤더 감지 정확도
- [x] RAGAS 메트릭 구현
- [x] 복잡도 메트릭 정의

### Phase 2: 파일럿 실험 ✅
- [x] 소규모 데이터셋 실험 프레임워크
- [x] 계층적 샘플링
- [x] Ablation Study 설계
- [x] 통계 분석 통합

### Phase 3: 전체 규모 실험
- [ ] 대규모 데이터셋 확장
- [x] 교차 검증 도구
- [x] 통계 분석 도구
- [ ] 도메인별 분석 스크립트

### Phase 4: 결과 분석
- [x] 성능 비교 테이블 생성
- [x] 복잡도별 분석 차트
- [x] 오류 분석 도구
- [x] 종합 리포트 생성

---

## 📚 참고 자료

- **TEDS**: [논문 링크]
- **GriTS**: [논문 링크]
- **RAGAS**: https://github.com/explodinggradients/ragas
- **Cohen's d**: 효과 크기 해석 가이드

---

## 🎯 다음 단계

1. **파일럿 실험 실행**: 소규모 데이터로 파이프라인 검증
2. **메트릭 검증**: 계산된 메트릭이 예상과 일치하는지 확인
3. **베이스라인 통합**: TATR, Sato, TableRAG 등 실제 통합 테스트
4. **대규모 실험 준비**: 데이터셋 확장 및 실행 계획 수립



