"""
Base Agent Module

This module provides the BaseAgent class which is the foundation for all agent types
in the advanced autonomous research system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from langchain_core.language_models import BaseChatModel


class BaseAgent(ABC):
    """
    Abstract base class for all agent types in the multi-agent research system.
    
    This class defines the common interface that all agents must implement.
    """
    
    def __init__(
        self,
        name: str,
        llm: BaseChatModel,
        system_prompt: str,
        memory: Optional[Any] = None,
    ):
        """
        Initialize a BaseAgent instance.
        
        Args:
            name: The name/identifier for this agent
            llm: The language model to use for this agent
            system_prompt: The system prompt that defines the agent's role and behavior
            memory: Optional memory system for the agent
        """
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.memory = memory
        self.task_history = []
        
    @abstractmethod
    async def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a task and return the result.
        
        Args:
            task: The task to execute
            context: Additional context for the task execution
            
        Returns:
            A dictionary containing the execution results
        """
        pass
    
    def add_to_history(self, task: str, result: Dict[str, Any]) -> None:
        """
        Add a task and its result to the agent's history.
        
        Args:
            task: The task that was executed
            result: The result of the task execution
        """
        self.task_history.append({"task": task, "result": result})
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get the agent's task execution history.
        
        Returns:
            A list of task-result pairs
        """
        return self.task_history