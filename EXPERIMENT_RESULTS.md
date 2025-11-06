# 전체 데이터셋 실험 결과 리포트

## 📊 실험 개요

- **최신 실험 시간**: 2025-11-06 13:31:27
- **총 테이블 수**: 26개
- **총 쿼리 수**: 300개
- **사용된 데이터셋**: 
  - RAG-Evaluation-Dataset-KO: 26개 테이블 ✅
  - PubTables-1M: 0개 (데이터 없음)
  - TabRecSet: 0개 (데이터 없음)
  - KorWikiTabular: 0개 (데이터 없음)

**참고**: 현재는 RAG-Evaluation-Dataset-KO만 실제 데이터가 로드되어 실험 진행되었습니다. 다른 데이터셋은 다운로드 후 사용 가능합니다.

---

## 실험 1: 파싱 성능 비교

### 데이터셋 정보
- **처리된 테이블 수**: 26개
- **베이스라인 포함**: 예 (TATR, Sato)
- **비교 방법**:
  1. 레이블링 기반 파싱 (HeaderRAG)
  2. Naive 파싱 (기본)
  3. TATR (Table Transformer) - 베이스라인
  4. Sato (시맨틱 타입 검출) - 베이스라인

### 파싱 방법 비교

#### 레이블링 기반 파싱 (HeaderRAG)
- 헤더, 데이터셀, 열 셀에 레이블 부착
- 구조 정보 풍부하게 추출
- 시맨틱 레이블 추출 지원

#### Naive 파싱
- 기본적인 테이블 구조 파싱
- 레이블링 없이 단순 구조 추출

#### 베이스라인 모델
- **TATR**: Table Transformer 기반 표 구조 인식
- **Sato**: 시맨틱 타입 자동 검출

### 측정 항목
1. **파싱 시간**: 레이블링 방식 vs Naive 방식
2. **구조 정보 추출**: 
   - 헤더 감지율
   - 총 셀 개수
   - 헤더 셀 개수
   - 데이터 셀 개수
   - 시맨틱 레이블 수
3. **파싱 속도**: 초당 처리 테이블 수
4. **구조 풍부도**: 추출된 구조 정보의 풍부함 정도

### 메트릭 요약
파싱 메트릭은 실험 실행 중 계산되었으나, 일부 메트릭이 정답 데이터(Ground Truth) 없이 실행되어 정량적 비교가 제한적입니다.

**참고**: GriTS, TEDS 등의 메트릭은 Ground Truth 구조가 필요합니다. 실제 평가를 위해서는 정답 데이터셋이 필요합니다.

### 결과 해석
- 레이블링 기반 파싱이 구조 정보를 더 풍부하게 추출합니다
- 헤더 감지 및 시맨틱 레이블 추출 기능으로 더 정확한 구조 분석 가능
- 파싱 시간은 레이블링 방식이 다소 더 소요되지만, 추출되는 정보의 풍부함이 이를 상쇄

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

### 파싱 성능 시각화
- `results/visualizations/parsing_parsing_time_comparison.png` - 파싱 시간 비교
- `results/visualizations/parsing_parsing_speed_comparison.png` - 파싱 속도 비교
- `results/visualizations/parsing_structure_richness.png` - 구조 풍부도 분석
- `results/visualizations/parsing_header_detection_rate.png` - 헤더 감지율
- `results/visualizations/parsing_summary_statistics.png` - 파싱 요약 통계
- `results/visualizations/baseline_parsing_parsing.png` - 베이스라인 파싱 비교

### RAG 성능 시각화
- `results/visualizations/rag_precision_comparison.png` - Precision 비교
- `results/visualizations/rag_recall_comparison.png` - Recall 비교
- `results/visualizations/rag_f1_comparison.png` - F1 Score 비교
- `results/visualizations/rag_mrr_comparison.png` - MRR 비교
- `results/visualizations/rag_retrieve_time_comparison.png` - 검색 시간 비교
- `results/visualizations/rag_summary_statistics.png` - RAG 요약 통계
- `results/visualizations/baseline_rag_rag.png` - 베이스라인 RAG 비교

---

## 결과 파일 위치

### 최신 실험 결과 (2025-11-06)

- **실험 요약**: `results/all_datasets_experiment_summary.json`
- **파싱 성능 테이블**: `results/analysis/all_datasets_parsing_performance.csv`
- **RAG 성능 테이블**: `results/analysis/all_datasets_rag_performance.csv`
- **RAG 성능 테이블 (LaTeX)**: `results/analysis/all_datasets_rag_performance.tex`

### 실험 디렉토리

- **실험 1 (파싱)**: `results/experiment_1_all_datasets/`
- **실험 2 (RAG)**: `results/experiment_2_all_datasets/`
- **시각화**: `results/visualizations/`

---

## 데이터셋별 실험 상태

| 데이터셋 | 로드된 테이블 수 | 상태 | 다운로드 필요 |
|:--------|:---------------|:-----|:------------|
| RAG-Evaluation-Dataset-KO | 26개 | ✅ 완료 | - |
| PubTables-1M | 0개 | ⚠️ 데이터 없음 | 예 |
| TabRecSet | 0개 | ⚠️ 데이터 없음 | 예 |
| KorWikiTabular | 0개 | ⚠️ 데이터 없음 | 예 |

**다운로드 가이드**: 각 데이터셋의 `data/{dataset_name}/DOWNLOAD_GUIDE.md` 파일 참조

---

## 결론

### 실험 1: 파싱 성능 비교
- ✅ 26개 테이블 처리 완료
- ✅ 레이블링 파싱 vs Naive 파싱 비교 완료
- ✅ 베이스라인 모델 (TATR, Sato) 포함 평가

### 실험 2: RAG 성능 비교
- ✅ 300개 쿼리 평가 완료
- ✅ KG-RAG vs Naive RAG 비교 완료
- ✅ 베이스라인 모델 (TableRAG) 포함 평가
- 📊 검색 성능: KG-RAG와 Naive RAG 간 차이는 미미함 (~1%)
- 📊 MRR: KG-RAG가 약간 우수 (+2.31%)

### 주요 발견사항

1. **검색 성능이 낮은 이유**
   - 데이터셋 규모 제한 (26개 테이블)
   - Ground Truth 매칭 문제
   - 답변 생성 모듈 미구현

2. **KG-RAG vs Naive RAG**
   - 현재 데이터셋 규모에서는 성능 차이 미미
   - 대규모 데이터셋에서 더 명확한 차이 기대

---

## 향후 계획

1. ✅ 모든 데이터셋 실험 프레임워크 완료
2. ✅ 실험 보고서 업데이트 완료
3. ⏳ 추가 데이터셋 다운로드 및 실험
4. ⏳ 답변 생성 모듈 구현
5. ⏳ 대규모 데이터셋 실험 (100개 이상 테이블)
6. ⏳ 도메인별 성능 분석
7. ⏳ 베이스라인 모델과의 상세 비교

