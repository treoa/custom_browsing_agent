"""
Base Planner Module

This module provides the BasePlanner class which is the foundation for all planning
components in the advanced autonomous research system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from langchain_core.language_models import BaseChatModel


class BasePlanner(ABC):
    """
    Abstract base class for all planners in the research system.
    
    This class defines the common interface that all planners must implement.
    """
    
    def __init__(
        self,
        name: str,
        llm: BaseChatModel,
        system_prompt: str,
    ):
        """
        Initialize a BasePlanner instance.
        
        Args:
            name: The name/identifier for this planner
            llm: The language model to use for this planner
            system_prompt: The system prompt that defines the planner's role
        """
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.plan_history = []
    
    @abstractmethod
    async def create_plan(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a plan for a given task.
        
        Args:
            task: The task to plan for
            context: Additional context for planning
            
        Returns:
            A structured plan
        """
        pass
    
    @abstractmethod
    async def update_plan(self, plan_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing plan.
        
        Args:
            plan_id: The ID of the plan to update
            updates: The updates to apply to the plan
            
        Returns:
            The updated plan
        """
        pass
    
    @abstractmethod
    async def evaluate_plan(self, plan_id: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a plan based on execution results.
        
        Args:
            plan_id: The ID of the plan to evaluate
            results: The results of executing the plan
            
        Returns:
            Evaluation results
        """
        pass
    
    def add_to_history(self, plan: Dict[str, Any]) -> None:
        """
        Add a plan to the planner's history.
        
        Args:
            plan: The plan to add to history
        """
        self.plan_history.append(plan)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get the planner's planning history.
        
        Returns:
            A list of plans
        """
        return self.plan_history
    
    def get_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific plan by ID.
        
        Args:
            plan_id: The ID of the plan to retrieve
            
        Returns:
            The plan, or None if not found
        """
        for plan in self.plan_history:
            if plan.get("id") == plan_id:
                return plan
        return None