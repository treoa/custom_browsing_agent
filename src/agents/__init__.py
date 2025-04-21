"""
Agent Package

This package provides the various specialized agents for the multi-agent research system.
"""

from .base_agent import BaseAgent
from .executive_agent import ExecutiveAgent
from .research_agent import ResearchAgent
from .analysis_agent import AnalysisAgent
from .critique_agent import CritiqueAgent
from .synthesis_agent import SynthesisAgent

__all__ = [
    'BaseAgent',
    'ExecutiveAgent',
    'ResearchAgent',
    'AnalysisAgent',
    'CritiqueAgent',
    'SynthesisAgent',
]