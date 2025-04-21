"""
Evaluation Package

This package provides components for evaluating research results and determining
when tasks are complete for the Advanced Autonomous Research Agent system.
"""

from .base_evaluator import BaseEvaluator
from .completion_criteria import CompletionCriteria, CriteriaLevel
from .completion_evaluator import CompletionEvaluator

# Make key classes available at package level
__all__ = [
    'BaseEvaluator',
    'CompletionCriteria',
    'CriteriaLevel',
    'CompletionEvaluator',
]