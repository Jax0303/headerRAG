# 다음 단계 실행 가이드

## ✅ 완료된 작업

1. **실험 메트릭 구현 완료**
   - ✅ TEDS (Tree Edit Distance-based Similarity)
   - ✅ GriTS (Grid Table Similarity) - Content, Topology, Location
   - ✅ 헤더 감지 정확도 (Precision, Recall, F1)
   - ✅ RAGAS 메트릭 (Faithfulness, Answer Relevancy, Context Precision/Recall)
   - ✅ 표 복잡도 메트릭 (구조적/시각적 복잡도)

2. **파일럿 실험 프레임워크 구현 완료**
   - ✅ 계층적 샘플링
   - ✅ 실험 1A, 2A, 3A 구현
   - ✅ Ablation Study 설계

3. **통계 분석 도구 구현 완료**
   - ✅ K-Fold Cross Validation
   - ✅ Paired t-test, Wilcoxon test
   - ✅ Cohen's d 효과 크기 계산

4. **메트릭 검증 완료**
   - ✅ 모든 메트릭이 정상 작동 확인

## 🚀 다음 단계: 파일럿 실험 실행

### 1단계: 의존성 설치

```bash
# 필수 라이브러리 설치
pip install -r requirements.txt

# 추가 라이브러리 (선택적, 권장)
pip install zss ragas datasets scipy
```

### 2단계: 메트릭 검증 (이미 완료)

```bash
python experiments/validate_metrics.py
```

**결과**: 모든 메트릭 검증 통과 ✓

### 3단계: 소규모 파일럿 실험 실행

#### 옵션 A: 기존 데이터셋 사용 (RAG-Evaluation-Dataset-KO)

```bash
# 실험 1A: 파싱 성능 평가
python experiments/run_pilot_experiments.py \
    --experiment 1a \
    --dataset rag_eval_ko \
    --max_tables 20

# 실험 2A: RAG 성능 평가
python experiments/run_pilot_experiments.py \
    --experiment 2a \
    --dataset rag_eval_ko \
    --max_tables 20

# 전체 파일럿 실험
python experiments/run_pilot_experiments.py \
    --experiment all \
    --dataset rag_eval_ko \
    --max_tables 20
```

#### 옵션 B: 새 메트릭 통합 실험 실행

```bash
# 향상된 실험 1 (새 메트릭 포함)
python experiments/integrate_new_metrics.py \
    --experiment 1 \
    --dataset rag_eval_ko \
    --max_tables 20
```

### 4단계: Ablation Study 실행

```bash
python experiments/run_pilot_experiments.py \
    --experiment ablation \
    --dataset rag_eval_ko \
    --max_tables 20
```

### 5단계: 결과 분석

실험 결과는 `results/pilot/` 디렉토리에 저장됩니다.

```python
from experiments.result_analyzer import ResultAnalyzer
import json

# 결과 로드
with open('results/pilot/experiment_1a_parsing.json', 'r') as f:
    results = json.load(f)

# 성능 비교 테이블 생성
analyzer = ResultAnalyzer()
performance_table = analyzer.create_performance_table(
    results,
    metrics=['grits_overall', 'header_f1', 'teds'],
    output_path='results/analysis/performance_table.csv'
)
```

## 📊 예상 결과

### 실험 1A: 파싱 성능

예상 메트릭:
- **GriTS-Overall**: 0.85-0.95 (우수)
- **Header F1**: 0.90-0.95
- **TEDS**: 0.80-0.90 (zss 설치 시)

### 실험 2A: RAG 성능

예상 메트릭:
- **Faithfulness**: 0.80-0.95
- **Answer Relevancy**: 0.75-0.90
- **Context Precision**: 0.70-0.90

### 복잡도 분석

예상 분포:
- **Low**: 30-40%
- **Medium**: 40-50%
- **High**: 10-20%

## 🔍 문제 해결

### zss 라이브러리 없음

```
UserWarning: zss 라이브러리가 설치되지 않았습니다.
```

**해결**: TEDS 메트릭은 건너뛰고 GriTS만 사용됩니다. 설치하려면:
```bash
pip install zss
```

### ragas 라이브러리 없음

```
UserWarning: ragas 라이브러리가 설치되지 않았습니다.
```

**해결**: 기본 구현을 사용합니다. 라이브러리 설치하려면:
```bash
pip install ragas datasets
```

### 데이터셋 로드 실패

**해결**: 데이터셋이 준비되지 않은 경우 샘플 데이터 사용:
```python
from experiments.run_experiments import ExperimentRunner
runner = ExperimentRunner()
tables = runner.load_test_data()  # 기본 샘플 데이터 사용
```

## 📝 체크리스트

- [x] 메트릭 구현 완료
- [x] 메트릭 검증 완료
- [ ] 파일럿 실험 실행
- [ ] 결과 분석
- [ ] 베이스라인 통합 확인
- [ ] 대규모 실험 준비

## 🎯 권장 실행 순서

1. **메트릭 검증** (완료 ✓)
   ```bash
   python experiments/validate_metrics.py
   ```

2. **소규모 파일럿 실험** (다음 단계)
   ```bash
   python experiments/run_pilot_experiments.py --experiment 1a --max_tables 10
   ```

3. **결과 확인 및 분석**
   - `results/pilot/` 디렉토리 확인
   - 메트릭 값이 예상 범위 내인지 확인

4. **전체 파일럿 실험 실행**
   ```bash
   python experiments/run_pilot_experiments.py --experiment all --max_tables 50
   ```

5. **Ablation Study 실행**
   ```bash
   python experiments/run_pilot_experiments.py --experiment ablation --max_tables 30
   ```

## 📚 참고 문서

- `EXPERIMENT_ROADMAP.md`: 전체 실험 계획
- `experiments/validate_metrics.py`: 메트릭 검증 스크립트
- `experiments/run_pilot_experiments.py`: 파일럿 실험 실행 스크립트
- `src/evaluation/`: 메트릭 구현 코드

