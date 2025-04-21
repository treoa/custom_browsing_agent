"""
Base Evaluator Module

This module provides the BaseEvaluator abstract class which defines the common
interface for all evaluator components in the Advanced Autonomous Research Agent.
"""

from typing import Dict, List, Any, Optional, Union, Callable
import abc
import time


class BaseEvaluator(abc.ABC):
    """
    Abstract base class for evaluator components.
    
    This class defines the common interface that all evaluator components must
    implement, ensuring consistency across different evaluation aspects.
    """
    
    def __init__(self, name: str = "BaseEvaluator"):
        """
        Initialize a BaseEvaluator instance.
        
        Args:
            name: The name/identifier for this evaluator
        """
        self.name = name
        self.creation_time = time.time()
    
    @abc.abstractmethod
    async def evaluate(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Evaluate the given data according to this evaluator's criteria.
        
        All evaluator components must implement this method with appropriate
        parameters for their specific evaluation type.
        
        Returns:
            Dictionary of evaluation results
        """
        pass