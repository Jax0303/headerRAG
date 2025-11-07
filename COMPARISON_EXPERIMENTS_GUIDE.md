# 비교 실험 실행 가이드

이 문서는 HeaderRAG 프로젝트에서 베이스라인 모델과의 비교 실험을 실행하는 방법을 설명합니다.

## 목차

1. [빠른 시작](#빠른-시작)
2. [실험 1: 파싱 성능 비교](#실험-1-파싱-성능-비교)
3. [실험 2: RAG 성능 비교](#실험-2-rag-성능-비교)
4. [전체 실험 실행](#전체-실험-실행)
5. [결과 확인](#결과-확인)

---

## 빠른 시작

### 기본 비교 실험 실행

```python
from experiments.run_experiments import ExperimentRunner
import pandas as pd

# 실험 러너 초기화
runner = ExperimentRunner(output_dir="results")

# 테스트 데이터 로드
tables = runner.load_test_data("data/sample_tables")

# 실험 1: 파싱 성능 비교 (베이스라인 포함)
results = runner.experiment_1_parsing_comparison(
    tables=tables,
    include_baselines=True  # TATR, Sato 포함
)

# 결과 저장
runner.save_results(results, "experiment_1_parsing")
```

---

## 실험 1: 파싱 성능 비교

### 비교 대상

- **레이블링 기반 파싱** (HeaderRAG)
- **Naive 파싱** (기본)
- **TATR** (베이스라인) - Table Transformer
- **Sato** (베이스라인) - 시맨틱 타입 검출

### 실행 방법

```python
from experiments.run_experiments import ExperimentRunner

runner = ExperimentRunner()

# 테이블 데이터 로드
tables = runner.load_test_data("data/sample_tables")
# 또는 실제 데이터셋 사용
# tables = runner.load_test_data("", use_dataset=True)

# 비교 실험 실행
results = runner.experiment_1_parsing_comparison(
    tables=tables,
    include_baselines=True  # 베이스라인 모델 포함
)

# 결과 저장
runner.save_results(results, "experiment_1_parsing")
```

### 결과 확인

```python
# 결과 구조 확인
print("레이블링 파싱 결과:", len(results['labeled_parsing']))
print("Naive 파싱 결과:", len(results['naive_parsing']))

if 'tatr_parsing' in results:
    print("TATR 파싱 결과:", len(results['tatr_parsing']))

if 'sato_semantic' in results:
    print("Sato 검출 결과:", len(results['sato_semantic']))

# 요약 통계 확인
summary = results.get('summary', {})
print("\n=== 파싱 성능 요약 ===")
print(f"평균 구조 풍부도: {summary.get('avg_structure_richness', 0):.3f}")
print(f"헤더 감지율: {summary.get('header_detection_rate', 0):.2%}")
print(f"레이블링 파싱 평균 시간: {summary['labeled_parsing_stats']['avg_parsing_time_ms']:.2f}ms")
print(f"Naive 파싱 평균 시간: {summary['naive_parsing_stats']['avg_parsing_time_ms']:.2f}ms")
```

---

## 실험 2: RAG 성능 비교

### 비교 대상

- **KG 기반 RAG** (HeaderRAG)
- **Naive RAG** (기본)
- **TableRAG** (베이스라인)

### 실행 방법

```python
from experiments.run_experiments import ExperimentRunner

runner = ExperimentRunner()

# 테이블 데이터 로드
tables = runner.load_test_data("data/sample_tables")

# 테스트 쿼리 준비
queries = [
    "2023년 매출액은 얼마인가요?",
    "직원 수가 가장 많은 연도는?",
    "순이익이 가장 높은 연도는?"
]

# 정답 테이블 ID 매핑 (각 쿼리에 대한 정답 테이블)
ground_truth = {
    "2023년 매출액은 얼마인가요?": ["table_0"],
    "직원 수가 가장 많은 연도는?": ["table_0"],
    "순이익이 가장 높은 연도는?": ["table_0"]
}

# 비교 실험 실행
results = runner.experiment_2_rag_comparison(
    tables=tables,
    queries=queries,
    ground_truth=ground_truth,
    include_baselines=True  # TableRAG 포함
)

# 결과 저장
runner.save_results(results, "experiment_2_rag")
```

### 결과 확인

```python
# 결과 구조 확인
print("KG RAG 결과:", len(results['kg_rag']))
print("Naive RAG 결과:", len(results['naive_rag']))

if 'tablerag_baseline' in results:
    print("TableRAG 베이스라인 결과:", len(results['tablerag_baseline']))

# 요약 통계 확인
summary = results.get('summary', {})
print("\n=== RAG 성능 요약 ===")

# KG RAG 메트릭
kg_avg = summary.get('kg_rag_avg', {})
print(f"\nKG RAG:")
print(f"  Precision: {kg_avg.get('precision', 0):.3f}")
print(f"  Recall: {kg_avg.get('recall', 0):.3f}")
print(f"  F1: {kg_avg.get('f1', 0):.3f}")
print(f"  MRR: {kg_avg.get('mrr', 0):.3f}")

# Naive RAG 메트릭
naive_avg = summary.get('naive_rag_avg', {})
print(f"\nNaive RAG:")
print(f"  Precision: {naive_avg.get('precision', 0):.3f}")
print(f"  Recall: {naive_avg.get('recall', 0):.3f}")
print(f"  F1: {naive_avg.get('f1', 0):.3f}")
print(f"  MRR: {naive_avg.get('mrr', 0):.3f}")

# TableRAG 메트릭 (있는 경우)
if 'tablerag_baseline' in results and results['tablerag_baseline']:
    tablerag_metrics = [r['metrics'] for r in results['tablerag_baseline']]
    if tablerag_metrics:
        print(f"\nTableRAG:")
        print(f"  Precision: {np.mean([m.get('precision', 0) for m in tablerag_metrics]):.3f}")
        print(f"  Recall: {np.mean([m.get('recall', 0) for m in tablerag_metrics]):.3f}")
        print(f"  F1: {np.mean([m.get('f1', 0) for m in tablerag_metrics]):.3f}")
```

---

## 전체 실험 실행

### 방법 1: 개별 실험 실행

```python
from experiments.run_experiments import ExperimentRunner

runner = ExperimentRunner(output_dir="results")

# 테이블 데이터 로드
tables = runner.load_test_data("data/sample_tables")

# 실험 1: 파싱 성능 비교
print("=== 실험 1 시작 ===")
results_1 = runner.experiment_1_parsing_comparison(
    tables=tables,
    include_baselines=True
)
runner.save_results(results_1, "experiment_1_parsing")

# 실험 2: RAG 성능 비교
print("\n=== 실험 2 시작 ===")
queries = ["2023년 매출액은?", "직원 수는?"]
ground_truth = {
    "2023년 매출액은?": ["table_0"],
    "직원 수는?": ["table_0"]
}
results_2 = runner.experiment_2_rag_comparison(
    tables=tables,
    queries=queries,
    ground_truth=ground_truth,
    include_baselines=True
)
runner.save_results(results_2, "experiment_2_rag")

# 실험 3: 복잡도 분석
print("\n=== 실험 3 시작 ===")
results_3 = runner.experiment_3_complexity_analysis(tables)
runner.save_results(results_3, "experiment_3_complexity")
```

### 방법 2: run_all_experiments 사용

```python
from experiments.run_experiments import ExperimentRunner

runner = ExperimentRunner()

# 전체 실험 실행
results = runner.run_all_experiments(
    data_path="data/sample_tables",
    use_dataset=False,
    include_baselines=True,  # 베이스라인 포함
    queries=None,  # 기본 쿼리 사용
    ground_truth=None  # 기본 ground truth 사용
)

print("모든 실험 완료!")
```

### 방법 3: 명령줄에서 실행

```bash
# 기본 실행 (베이스라인 없이)
python experiments/run_experiments.py

# 베이스라인 포함 실행하려면 코드 수정 필요
# 또는 Python 스크립트로 실행
python -c "
from experiments.run_experiments import ExperimentRunner
runner = ExperimentRunner()
tables = runner.load_test_data('data/sample_tables')
results = runner.experiment_1_parsing_comparison(tables, include_baselines=True)
runner.save_results(results, 'experiment_1_parsing')
"
```

---

## 결과 확인

### 결과 파일 위치

```
results/
├── experiment_1/
│   ├── cycle_001.json      # 사이클별 결과
│   ├── cycle_002.json
│   └── logs/
│       └── experiment_results.txt  # 상세 로그
├── experiment_2/
│   ├── cycle_001.json
│   └── logs/
└── experiment_3/
    ├── cycle_001.json
    └── logs/
```

### 결과 분석 예제

```python
import json
import pandas as pd

# 결과 파일 로드
with open("results/experiment_1/cycle_001.json", "r", encoding="utf-8") as f:
    results = json.load(f)

# 레이블링 파싱 통계
labeled_stats = pd.DataFrame([
    r['stats'] for r in results['labeled_parsing']
])
print("레이블링 파싱 통계:")
print(labeled_stats.describe())

# Naive 파싱 통계
naive_stats = pd.DataFrame([
    r['stats'] for r in results['naive_parsing']
])
print("\nNaive 파싱 통계:")
print(naive_stats.describe())

# TATR 베이스라인 통계 (있는 경우)
if 'tatr_parsing' in results:
    tatr_stats = pd.DataFrame([
        r['stats'] for r in results['tatr_parsing']
    ])
    print("\nTATR 파싱 통계:")
    print(tatr_stats.describe())

# 비교 결과
comparison = pd.DataFrame(results['comparison'])
print("\n비교 결과:")
print(comparison.describe())
```

### 시각화

```python
from experiments.visualize_results import ResultVisualizer

visualizer = ResultVisualizer(results_dir="results")

# 파싱 결과 시각화
visualizer.visualize_parsing_results()

# RAG 결과 시각화
visualizer.visualize_rag_results()
```

---

## 베이스라인 모델 설정

### TATR 모델 경로 설정

```python
import os

# 환경변수로 설정
os.environ['TATR_REPO_PATH'] = '/path/to/table-transformer'

# 또는 직접 지정
from src.baselines import TATRParser
tatr = TATRParser(repo_path='/path/to/table-transformer')
```

### Sato 모델 경로 설정

```python
import os

os.environ['SATO_REPO_PATH'] = '/path/to/sato'

# 또는 직접 지정
from src.baselines import SatoSemanticTypeDetector
sato = SatoSemanticTypeDetector(repo_path='/path/to/sato')
```

### TableRAG 설정

```python
import os

os.environ['TABLERAG_REPO_PATH'] = '/path/to/tablerag'

# 또는 직접 지정
from src.baselines import TableRAGBaseline
rag = TableRAGBaseline(
    repo_path='/path/to/tablerag',
    use_colbert=False  # ColBERT 사용하려면 True
)
```

---

## 베이스라인 없이 실행

베이스라인 모델이 설치되지 않았거나 사용하지 않으려면:

```python
# include_baselines=False로 설정
results = runner.experiment_1_parsing_comparison(
    tables=tables,
    include_baselines=False  # 베이스라인 제외
)
```

베이스라인 모델이 없어도 실험은 정상적으로 실행되며, 경고 메시지만 표시됩니다.

---

## 문제 해결

### 베이스라인 모델이 로드되지 않음

```python
# 베이스라인 모델 import 확인
try:
    from src.baselines import TATRParser
    print("TATR 로드 성공")
except ImportError as e:
    print(f"TATR 로드 실패: {e}")
```

### 베이스라인 모델이 시뮬레이션 모드로 동작

베이스라인 모델의 실제 저장소가 없으면 시뮬레이션 모드로 동작합니다. 실제 모델을 사용하려면 해당 저장소를 설치하고 경로를 설정하세요.

자세한 내용은 [BASELINES_GUIDE.md](BASELINES_GUIDE.md)를 참조하세요.

---

## 예제 스크립트

전체 비교 실험을 실행하는 예제 스크립트:

```python
#!/usr/bin/env python
# comparison_experiments.py

from experiments.run_experiments import ExperimentRunner
import pandas as pd

def main():
    # 실험 러너 초기화
    runner = ExperimentRunner(output_dir="results")
    
    # 테이블 데이터 로드
    print("테이블 데이터 로드 중...")
    tables = runner.load_test_data("data/sample_tables")
    print(f"로드된 테이블 수: {len(tables)}")
    
    # 실험 1: 파싱 성능 비교
    print("\n" + "="*50)
    print("실험 1: 파싱 성능 비교 (베이스라인 포함)")
    print("="*50)
    results_1 = runner.experiment_1_parsing_comparison(
        tables=tables,
        include_baselines=True
    )
    runner.save_results(results_1, "experiment_1_parsing")
    
    # 실험 2: RAG 성능 비교
    print("\n" + "="*50)
    print("실험 2: RAG 성능 비교 (베이스라인 포함)")
    print("="*50)
    
    # 샘플 쿼리 생성
    queries = [
        "2023년 매출액은 얼마인가요?",
        "직원 수가 가장 많은 연도는?",
        "순이익이 가장 높은 연도는?"
    ]
    
    # Ground truth (실제로는 데이터셋에서 가져와야 함)
    ground_truth = {
        q: ["table_0"] for q in queries
    }
    
    results_2 = runner.experiment_2_rag_comparison(
        tables=tables,
        queries=queries,
        ground_truth=ground_truth,
        include_baselines=True
    )
    runner.save_results(results_2, "experiment_2_rag")
    
    print("\n" + "="*50)
    print("모든 비교 실험 완료!")
    print("="*50)
    print(f"결과 저장 위치: results/")

if __name__ == "__main__":
    main()
```

실행:

```bash
python comparison_experiments.py
```




