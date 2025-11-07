# 라벨링 시스템 개선 로드맵

## 현재 시스템 한계점

### 1. 헤더 감지
- **현재**: 첫 행/열의 텍스트 비율만 확인 (50% 기준)
- **문제**: 
  - 다중 헤더 행/열 감지 불가 (예: 2-3줄 헤더)
  - 중첩 헤더 처리 불가 (예: "2023년 | 매출 | 국내/해외")
  - 비정형 헤더 구조 처리 불가

### 2. 시맨틱 레이블
- **현재**: 단순 패턴 매칭 (연도, 금액, 비율)
- **문제**:
  - 도메인 특화 용어 인식 불가 ("EBITDA", "ROE" 등)
  - 문맥 이해 없음 ("Q1" vs "1분기")
  - 다국어 지원 제한적

### 3. 병합 셀 감지
- **현재**: 같은 값 연속만 확인
- **문제**:
  - NaN으로 병합된 셀만 감지 가능
  - 실제 병합 정보 없으면 실패
  - 불규칙한 병합 구조 처리 불가

### 4. 복잡한 표 구조
- **현재**: 단순 2차원 배열 처리
- **문제**:
  - 스팬 헤더 처리 불가
  - 셀 정렬 정보 미활용 (시각적 힌트)
  - 계층적 구조 처리 불가

---

## 개선 우선순위 및 방안

### 🎯 Phase 1: 즉시 개선 가능 (규칙 기반 강화)

#### 1-1. 정규표현식 확장
```python
# 현재: 연도, 금액, 비율만
# 개선: 더 많은 패턴 추가

패턴_사전 = {
    '날짜': r'\d{4}[.-]\d{1,2}[.-]\d{1,2}',
    '시간': r'\d{1,2}:\d{2}',
    '백분율': r'\d+\.?\d*%',
    '통화': r'[₩$€£¥]\s*\d+',
    '단위': r'\d+\s*(kg|g|m|cm|km|l|ml)',
    # 한국어 특화
    '연도_한국어': r'\d{4}년',
    '분기': r'[1-4]분기|Q[1-4]',
    '월': r'\d{1,2}월',
    # 도메인별
    '재무지표': r'(ROE|ROA|EBITDA|PER|PBR)',
    '통계': r'(평균|중앙값|표준편차)',
}
```

**구현 포인트**:
- 정규표현식 사전 구축
- 우선순위 기반 매칭 (더 구체적인 것 먼저)
- 한국어 특화 패턴 추가

#### 1-2. 통계 기반 헤더 감지 개선
```python
# 현재: 텍스트 비율만 확인
# 개선: 다중 신호 통합

def improved_header_detection(table):
    신호들 = {
        '텍스트_비율': calculate_text_ratio(row),
        '값_일관성': calculate_value_consistency(row),
        '데이터_타입_분포': analyze_dtype_distribution(row),
        '위치_가중치': calculate_position_weight(row_index),
        '주변_셀_패턴': analyze_context_pattern(row)
    }
    
    점수 = 가중_합계(신호들)
    return 점수 > threshold
```

**구현 포인트**:
- 여러 힌트 통합 (텍스트 비율 + 데이터 타입 + 위치)
- 행/열별 통계 분석
- 이상값 탐지 활용

#### 1-3. 컨텍스트 기반 레이블링
```python
# 현재: 셀 하나만 봄
# 개선: 주변 셀 고려

def contextual_labeling(cell, row, col, table):
    # 같은 행/열 패턴 분석
    같은_행_패턴 = analyze_row_pattern(table[row, :])
    같은_열_패턴 = analyze_col_pattern(table[:, col])
    
    # 주변 셀 값 고려
    주변_셀 = get_neighbors(row, col, table)
    
    # 전체 테이블 구조 고려
    구조_패턴 = analyze_table_structure(table)
    
    return integrated_label(cell, 같은_행_패턴, 같은_열_패턴, 주변_셀, 구조_패턴)
```

**구현 포인트**:
- 행/열별 패턴 분석
- 주변 셀 값 활용
- 전체 구조 먼저 파악 후 개별 셀 레이블링

---

### 🚀 Phase 2: 머신러닝 통합

#### 2-1. Transformer 기반 헤더 분류
```python
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression

class MLHeaderDetector:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
        self.model = AutoModel.from_pretrained('klue/bert-base')
        self.classifier = LogisticRegression()
    
    def extract_features(self, cell, context):
        # 셀 값 임베딩
        cell_embedding = self.model.encode(cell.value)
        
        # 컨텍스트 임베딩 (주변 셀들)
        context_embedding = self.model.encode(context)
        
        # 위치 정보
        position_features = [cell.row, cell.col]
        
        return np.concatenate([cell_embedding, context_embedding, position_features])
    
    def predict_header(self, cell, context):
        features = self.extract_features(cell, context)
        return self.classifier.predict(features)
```

**필요 라이브러리**: `transformers`, `torch`, `sklearn`

**구현 포인트**:
- 한국어 BERT 모델 사용 (klue/bert-base 등)
- 셀 임베딩 + 컨텍스트 임베딩 결합
- 간단한 분류기 (LogisticRegression)로 헤더/데이터 분류
- 학습 데이터: 기존 파싱 결과 활용

#### 2-2. LLM 기반 시맨틱 레이블링
```python
from openai import OpenAI

class LLMSemanticLabeler:
    def __init__(self):
        self.client = OpenAI()
        
    def label_cell(self, cell, table_context):
        prompt = f"""
다음 테이블 셀의 의미를 분류하세요:
셀 값: {cell.value}
컨텍스트: {table_context}

가능한 레이블:
- 연도, 날짜, 시간
- 금액, 통화, 비율
- 재무지표 (ROE, ROA, EBITDA 등)
- 통계 (평균, 합계, 개수 등)
- 도메인 특화 용어

답변 형식: 레이블 이름만
"""
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
```

**필요 라이브러리**: `openai`, `langchain`

**구현 포인트**:
- Few-shot learning으로 예시 제공
- 도메인별 프롬프트 템플릿
- 캐싱으로 비용 절감 (같은 패턴 재사용)
- Fallback: LLM 실패 시 규칙 기반 사용

#### 2-3. 그래프 신경망(GNN) 기반 구조 분석
```python
import torch
from torch_geometric.nn import GCNConv

class TableStructureGNN:
    def __init__(self):
        # 셀을 노드로, 인접 관계를 엣지로
        self.gnn = GCNConv(...)
    
    def build_graph(self, table):
        # 각 셀을 노드로
        nodes = []
        edges = []
        
        for i in range(rows):
            for j in range(cols):
                node_id = i * cols + j
                nodes.append(cell_features(i, j))
                
                # 인접 셀과 연결
                if j > 0:
                    edges.append((node_id, node_id - 1))  # 왼쪽
                if i > 0:
                    edges.append((node_id, node_id - cols))  # 위쪽
        
        return nodes, edges
    
    def detect_structure(self, table):
        nodes, edges = self.build_graph(table)
        structure = self.gnn(nodes, edges)
        return structure
```

**필요 라이브러리**: `torch_geometric`, `torch`

**구현 포인트**:
- 셀 간 관계를 그래프로 모델링
- GNN으로 구조 패턴 학습
- 커뮤니티 탐지로 헤더 그룹 찾기

---

### 🎨 Phase 3: 컴퓨터 비전 통합

#### 3-1. Table Transformer 모델 활용
```python
from transformers import TableTransformerModel
import torchvision

class CVTableParser:
    def __init__(self):
        self.model = TableTransformerModel.from_pretrained('microsoft/table-transformer-structure-recognition')
    
    def parse_structure(self, table_image):
        # 이미지에서 표 구조 추출
        result = self.model(table_image)
        
        # 셀 위치, 크기, 병합 정보 추출
        cells = result['cells']
        
        for cell in cells:
            cell['bbox'] = cell.bounding_box
            cell['merged'] = detect_merge_from_bbox(cell, cells)
        
        return cells
```

**필요 라이브러리**: `detectron2`, `torchvision`, `transformers`

**구현 포인트**:
- Microsoft Table Transformer 사용
- 셀 경계 상자(bbox)로 병합 셀 감지
- 정렬 정보로 구조 이해

---

### 🔄 Phase 4: 통합 접근법

#### 4-1. 앙상블 메서드
```python
class EnsembleLabeler:
    def __init__(self):
        self.rule_based = LabeledTableParser()
        self.ml_detector = MLHeaderDetector()
        self.llm_labeler = LLMSemanticLabeler()
        self.cv_parser = CVTableParser()
    
    def label_table(self, table):
        results = {}
        
        # 1. 규칙 기반
        results['rule'] = self.rule_based.parse(table)
        
        # 2. ML 기반
        if has_gpu():
            results['ml'] = self.ml_detector.predict(table)
        
        # 3. LLM 기반 (일부 셀만)
        results['llm'] = self.llm_labeler.label_uncertain_cells(table, results['rule'])
        
        # 4. CV 기반 (이미지가 있는 경우)
        if has_image(table):
            results['cv'] = self.cv_parser.parse(table_image)
        
        # 결과 통합 (신뢰도 기반 가중 평균)
        return self.aggregate_results(results)
    
    def aggregate_results(self, results):
        # 각 방법의 신뢰도 점수 계산
        confidences = calculate_confidence(results)
        
        # 가중 평균
        final_labels = {}
        for method, labels in results.items():
            weight = confidences[method]
            for cell_id, label in labels.items():
                final_labels[cell_id] = weighted_average(
                    final_labels.get(cell_id, {}), 
                    label, 
                    weight
                )
        
        return final_labels
```

**구현 포인트**:
- 다양한 방법 조합
- 신뢰도 기반 가중 평균
- 불확실한 경우 여러 방법 재확인

---

## 단계별 구현 순서

### 단기 (1-2주)
1. ✅ 정규표현식 패턴 확장
2. ✅ 통계 기반 헤더 감지 개선
3. ✅ 컨텍스트 기반 레이블링 추가

### 중기 (1-2개월)
1. Transformer 기반 헤더 분류 모델 학습
2. LLM 기반 시맨틱 레이블링 통합
3. 평가 데이터셋 구축 및 성능 측정

### 장기 (3-6개월)
1. GNN 기반 구조 분석 모델 개발
2. CV 모델 통합 (이미지 기반 표 처리)
3. 앙상블 시스템 구축 및 최적화

---

## 구체적 개선 포인트 요약

### 즉시 개선 가능한 부분
1. **패턴 매칭 확장**
   - 날짜, 시간, 단위, 재무지표 등 패턴 추가
   - 한국어 특화 패턴 (년, 월, 일, 분기 등)

2. **헤더 감지 개선**
   - 텍스트 비율 + 데이터 타입 + 위치 통합
   - 다중 헤더 행 감지 (2-3줄까지)

3. **컨텍스트 활용**
   - 행/열 패턴 분석
   - 주변 셀 값 고려

### ML/딥러닝 추가 시
1. **한국어 BERT 활용**
   - klue/bert-base로 셀 임베딩
   - 헤더/데이터 분류 모델 학습

2. **LLM 활용**
   - GPT-4/Claude로 시맨틱 레이블링
   - Few-shot learning + 도메인 프롬프트

3. **CV 모델 활용**
   - Table Transformer로 구조 추출
   - 시각적 정보 활용

### 통합 시
1. **앙상블 접근**
   - 규칙 + ML + LLM + CV 통합
   - 신뢰도 기반 가중 평균

2. **점진적 개선**
   - 간단한 표는 규칙 기반 빠르게 처리
   - 복잡한 표만 ML/LLM 활용

---

## 참고 논문/모델

- **Table Transformer**: Microsoft, 구조 인식 전용 모델
- **TATR**: Table Transformer 기반 표 인식
- **RGPT**: 테이블 구조 이해용 GPT 모델
- **TAPAS**: Google, 테이블 QA 모델

---

## 예상 효과

### 정확도
- 현재: 간단한 표 70-80%, 복잡한 표 40-50%
- 개선 후: 간단한 표 95%+, 복잡한 표 75-85%

### 처리 속도
- 규칙 기반: 1000 tables/sec
- ML 추가: 100 tables/sec (GPU)
- LLM 추가: 10 tables/sec (API 지연)
- 통합: 단순 표는 빠르게, 복잡한 표만 느리게 처리

### 비용
- 규칙 기반: 무료
- ML: GPU 필요 (로컬/클라우드)
- LLM: API 비용 (셀당 $0.001 정도)

