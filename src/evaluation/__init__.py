"""Stage 11: Evaluation Benchmark & Error Analysis package."""

from src.evaluation.evaluator import BenchmarkEvaluator
from src.evaluation.models import (
    BenchmarkCase,
    BenchmarkReport,
    CaseEvaluationResult,
    ConfusionMatrix,
    ErrorType,
    EvaluationMetrics,
    PipelineStage,
    StageFailureDiagnosis,
)

__all__ = [
    "BenchmarkCase",
    "BenchmarkEvaluator",
    "BenchmarkReport",
    "CaseEvaluationResult",
    "ConfusionMatrix",
    "ErrorType",
    "EvaluationMetrics",
    "PipelineStage",
    "StageFailureDiagnosis",
]
