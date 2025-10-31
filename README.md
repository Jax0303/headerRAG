# HeaderRAG: 테이블 파싱 및 RAG 실험 프레임워크

## 프로젝트 개요

한국 기업에서 사용하는 복잡한 표 데이터를 대상으로 한 RAG(Retrieval-Augmented Generation) 실험 프레임워크입니다.

## 주요 실험

1. **레이블링 기반 파싱 vs Naive 파싱 성능 비교**
   - 각 데이터셀, 헤더셀, 열 셀에 레이블을 부착한 파싱 방식
   - 레이블 없이 단순 파싱하는 방식

2. **KG 기반 RAG vs Naive 파싱 RAG 비교**
   - 테이블을 Knowledge Graph로 변환 후 RAG
   - Naive하게 테이블 파싱 후 RAG

## 데이터셋 추천

### 1. KOSIS 국가통계포털
- **장점**: 다양한 산업/경제 통계, 복잡한 표 구조, 대규모 데이터
- **데이터 형식**: Excel, CSV
- **링크**: https://kosis.kr/

### 2. 공공데이터포털 (data.go.kr)
- **장점**: 한국 정부 기관의 공개 데이터, 표/차트 포함
- **주요 데이터**: 기업 재무제표, 통계 데이터, 정책 데이터
- **링크**: https://www.data.go.kr/

### 3. 금융감독원 전자공시시스템 (DART)
- **장점**: 실제 기업 재무제표, 매우 복잡한 표 구조, 거대한 규모
- **데이터 형식**: XBRL, Excel
- **링크**: https://dart.fss.or.kr/

### 4. 한국은행 경제통계시스템 (ECOS)
- **장점**: 경제 지표, 복잡한 시계열 표 데이터
- **링크**: https://ecos.bok.or.kr/

## 프로젝트 구조

```
headerRAG/
├── data/                    # 데이터셋 저장
├── models/                  # 모델 저장
├── src/
│   ├── parsing/            # 파싱 모듈
│   │   ├── labeled_parser.py
│   │   └── naive_parser.py
│   ├── kg/                 # Knowledge Graph 변환
│   │   └── table_to_kg.py
│   ├── rag/                # RAG 시스템
│   │   ├── kg_rag.py
│   │   └── naive_rag.py
│   └── evaluation/         # 평가 모듈
│       └── metrics.py
├── experiments/            # 실험 스크립트
├── requirements.txt
└── README.md
```

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

실험 실행:
```bash
python experiments/run_experiments.py
```

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

