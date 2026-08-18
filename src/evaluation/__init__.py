"""
Evaluation package for Phase 6.
Provides performance evaluation, Information Coefficient calculation,
Newey-West HAC statistical significance testing, regime breakdown, and market correlation analysis.
"""

from src.evaluation.metrics import PerformanceEvaluator

__all__ = ["PerformanceEvaluator"]
