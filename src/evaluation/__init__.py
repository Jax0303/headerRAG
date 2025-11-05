# Evaluation module
from .logger import EvaluationLogger

# RAG 관련 모듈은 조건부 import
try:
    from .metrics import RAGEvaluator
    from .dataset_metrics import DatasetEvaluator
    __all__ = ['RAGEvaluator', 'DatasetEvaluator', 'EvaluationLogger']
except ImportError:
    __all__ = ['EvaluationLogger']

