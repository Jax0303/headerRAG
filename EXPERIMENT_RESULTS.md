# 전체 데이터셋 실험 결과 리포트

## 📊 실험 개요

- **실험 시간**: 2025-11-05 19:26:27
- **총 테이블 수**: 26개
- **총 쿼리 수**: 300개
- **데이터셋**: RAG-Evaluation-Dataset-KO (전체)

---

## 실험 1: 파싱 성능 비교

### 데이터셋 정보
- **처리된 테이블 수**: 26개
- **베이스라인 포함**: 예 (TATR, Sato)

### 메트릭 요약
파싱 메트릭은 실험 실행 중 계산되었으나, 일부 메트릭이 정답 데이터(Ground Truth) 없이 실행되어 정량적 비교가 제한적입니다.

**참고**: GriTS, TEDS 등의 메트릭은 Ground Truth 구조가 필요합니다. 실제 평가를 위해서는 정답 데이터셋이 필요합니다.

---

## 실험 2: RAG 성능 비교

### 데이터셋 정보
- **처리된 테이블 수**: 26개
- **처리된 쿼리 수**: 300개
- **Ground Truth 항목 수**: 300개

### 성능 결과

#### KG-RAG (HeaderRAG)

| 메트릭 | 평균 | 표준편차 |
|:-------|:-----|:--------|
| **Precision** | 0.1471 | ±0.3502 |
| **Recall** | 0.0963 | ±0.2192 |
| **F1 Score** | 0.1146 | ±0.2682 |
| **MRR** | 0.0148 | ±0.0336 |
| Faithfulness | 0.0000 | ±0.0000 |
| Answer Relevancy | 0.0000 | ±0.0000 |

#### Naive RAG

| 메트릭 | 평균 | 표준편차 |
|:-------|:-----|:--------|
| **Precision** | 0.1480 | ±0.3500 |
| **Recall** | 0.0981 | ±0.2195 |
| **F1 Score** | 0.1158 | ±0.2681 |
| **MRR** | 0.0145 | ±0.0325 |
| Faithfulness | 0.0000 | ±0.0000 |
| Answer Relevancy | 0.0000 | ±0.0000 |

### 성능 비교 분석

| 메트릭 | 차이 | 상대적 변화 |
|:-------|:-----|:-----------|
| Precision | -0.0009 | -0.62% |
| Recall | -0.0018 | -1.87% |
| F1 Score | -0.0012 | -1.06% |
| MRR | +0.0003 | +2.31% |

### 주요 발견사항

1. **검색 성능 (Precision/Recall/F1)**
   - KG-RAG와 Naive RAG의 성능이 매우 유사함
   - Precision: 약 14.7-14.8% (낮은 수준)
   - Recall: 약 9.6-9.8% (낮은 수준)
   - F1 Score: 약 11.5% (낮은 수준)

2. **MRR (Mean Reciprocal Rank)**
   - KG-RAG가 약간 우수함 (+2.31%)
   - 절대값은 매우 낮음 (0.0148)

3. **RAGAS 메트릭**
   - Faithfulness와 Answer Relevancy가 0으로 계산됨
   - 원인: 답변 생성이 구현되지 않아 빈 문자열로 평가됨
   - 컨텍스트만 평가된 상태

### 결과 해석

#### 낮은 검색 성능의 원인

1. **데이터셋 크기 제한**
   - 26개 테이블만 사용하여 검색 공간이 제한적
   - 실제 사용 시에는 더 많은 테이블이 필요

2. **Ground Truth 매칭 문제**
   - 일부 쿼리에서 Ground Truth 테이블이 검색되지 않을 수 있음
   - 300개 쿼리 중 많은 쿼리가 관련 테이블을 찾지 못함

3. **답변 생성 미구현**
   - 현재는 검색만 수행하고 답변 생성은 미구현
   - RAGAS 메트릭의 Faithfulness와 Answer Relevancy는 답변이 있어야 계산 가능

#### 개선 방향

1. **답변 생성 모듈 추가**
   - LLM을 사용한 답변 생성 구현 필요
   - 생성된 답변에 대해 RAGAS 메트릭 정상 계산 가능

2. **데이터셋 확장**
   - 더 많은 테이블로 실험 (목표: 100개 이상)
   - 다양한 도메인 데이터셋 추가

3. **임베딩 모델 개선**
   - 더 강력한 임베딩 모델 사용
   - 도메인 특화 파인튜닝 고려

4. **검색 전략 개선**
   - 하이브리드 검색 (키워드 + 벡터)
   - 리랭킹 모델 추가

---

## 시각화 결과

다음 시각화 파일이 생성되었습니다:

- `results/visualizations/rag_precision_comparison.png`
- `results/visualizations/rag_recall_comparison.png`
- `results/visualizations/rag_f1_comparison.png`
- `results/visualizations/rag_mrr_comparison.png`
- `results/visualizations/baseline_rag_rag.png`

---

## 결과 파일 위치

- **실험 요약**: `results/full_experiment_summary.json`
- **파싱 성능 테이블**: `results/analysis/parsing_performance_table.csv`
- **RAG 성능 테이블**: `results/analysis/rag_performance_table.csv`
- **전체 로그**: `results/full_experiment_log.txt`

---

## 결론

1. **파싱 실험**: 26개 테이블 처리 완료, Ground Truth 없이 구조 분석 수행
2. **RAG 실험**: 300개 쿼리 평가 완료, 검색 성능은 낮지만 두 방법 간 차이는 미미함
3. **다음 단계**: 답변 생성 모듈 추가 및 데이터셋 확장 필요

---

## 향후 계획

1. ✅ 전체 데이터셋 실험 완료
2. ⏳ 답변 생성 모듈 구현
3. ⏳ 대규모 데이터셋 실험 (100개 이상 테이블)
4. ⏳ 도메인별 성능 분석
5. ⏳ 베이스라인 모델과의 상세 비교

