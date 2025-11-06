# 사이클 기능 사용 가이드

## 개요

실험 결과를 더 체계적으로 관리하기 위해 사이클 기능을 추가했습니다. 여러 번 실험을 실행할 때 결과를 묶어서 저장할 수 있습니다.

## 디렉토리 구조

실험 결과는 실험별로 디렉토리가 분리되어 저장됩니다:

```
results/
├── experiment_1/          # 실험 1: 파싱 성능 비교
│   ├── cycle_001.json     # 1번째 사이클 결과 (10회 실행)
│   ├── cycle_002.json     # 2번째 사이클 결과 (10회 실행)
│   └── logs/              # 상세 로그 파일
│       └── experiment_results.txt
├── experiment_2/          # 실험 2: RAG 성능 비교
│   ├── cycle_001.json
│   ├── cycle_002.json
│   └── logs/
│       └── experiment_results.txt
└── experiment_3/          # 실험 3: 복잡도 분석
    ├── cycle_001.json
    └── logs/
        └── experiment_results.txt
```

## 사용 방법

### 기본 사용 (사이클당 10회 실행)

```python
from experiments.run_experiments import ExperimentRunner

# 사이클당 10회 실행 (기본값)
runner = ExperimentRunner(output_dir="results", cycle_runs=10)

# 실험 실행
tables = runner.load_test_data("", use_dataset=True)
parsing_results = runner.experiment_1_parsing_comparison(tables)
runner.save_results(parsing_results, "experiment_1_parsing")
```

### 사이클 횟수 변경

```python
# 사이클당 5회 실행
runner = ExperimentRunner(output_dir="results", cycle_runs=5)

# 사이클당 20회 실행
runner = ExperimentRunner(output_dir="results", cycle_runs=20)
```

## 결과 파일 구조

각 사이클 결과 파일(`cycle_XXX.json`)은 다음 구조를 가집니다:

```json
{
  "cycle_number": 1,
  "cycle_runs": 10,
  "timestamp": "20251104_143811",
  "individual_results": [
    // 각 실행별 상세 결과 (10개)
  ],
  "aggregated_summary": {
    // 10회 실행 결과의 집계 요약
    "total_runs": 10,
    "avg_total_tables": 150.5,
    "avg_structure_richness": 0.85,
    // ... 기타 집계된 메트릭
  }
}
```

## 장점

1. **체계적인 관리**: 실험별로 디렉토리가 분리되어 결과를 찾기 쉬움
2. **효율적인 저장**: 여러 번 실행 결과를 하나의 파일로 묶어서 저장
3. **집계된 분석**: 사이클별로 자동 집계된 요약 데이터 제공
4. **개별 결과 보존**: 각 실행의 상세 결과도 `individual_results`에 보존

## 예시: 여러 사이클 실행

```python
runner = ExperimentRunner(output_dir="results", cycle_runs=10)

tables = runner.load_test_data("", use_dataset=True)

# 10번 실행 (1번째 사이클)
for i in range(10):
    parsing_results = runner.experiment_1_parsing_comparison(tables)
    runner.save_results(parsing_results, "experiment_1_parsing")
    # 10번째 실행 후 자동으로 cycle_001.json 생성

# 다시 10번 실행 (2번째 사이클)
for i in range(10):
    parsing_results = runner.experiment_1_parsing_comparison(tables)
    runner.save_results(parsing_results, "experiment_1_parsing")
    # 10번째 실행 후 자동으로 cycle_002.json 생성
```

## 진행 상황 확인

실험 실행 중 사이클 진행 상황이 출력됩니다:

```
사이클 1 진행 중: 3/10회 완료
사이클 1 진행 중: 4/10회 완료
...
사이클 1 결과 저장: results/experiment_1/cycle_001.json (10회 실행)
```




