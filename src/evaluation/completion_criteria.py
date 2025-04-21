"""
Completion Criteria Module

This module provides the CompletionCriteria class which defines the criteria for
determining when a research task is considered complete, addressing the issue of
premature task completion in research agents.
"""

from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import time
import json
from enum import Enum


class CriteriaLevel(Enum):
    """Enum defining the level of importance for completion criteria."""
    REQUIRED = 1    # Must be satisfied for completion
    IMPORTANT = 2   # Should be satisfied but not mandatory
    DESIRED = 3     # Nice to have but not critical


class CompletionCriteria:
    """
    CompletionCriteria defines explicit thresholds and metrics for determining
    when research is adequately thorough.
    
    This class encapsulates both quantitative metrics (like source diversity,
    depth coverage, etc.) and qualitative assessments for research completion.
    """
    
    def __init__(
        self,
        # Quantitative Metrics
        min_sources: int = 5,
        min_source_diversity_score: float = 0.6,
        min_depth_coverage_score: float = 0.7,
        min_breadth_coverage_percentage: float = 80.0,
        min_cross_verification_rate: float = 0.5,
        min_strategic_objective_coverage: float = 0.8,
        
        # Research Time Allocation
        min_exploration_time_percentage: float = 20.0,
        min_deep_investigation_time_percentage: float = 40.0,
        
        # Content Type Requirements
        content_type_distribution: Optional[Dict[str, float]] = None,
        required_content_types: Optional[List[str]] = None,
        
        # Custom Criteria
        custom_criteria: Optional[List[Dict[str, Any]]] = None,
        
        # Task-specific Configuration
        task_specific_thresholds: Optional[Dict[str, Any]] = None,
        
        # Override Controls
        allow_user_override: bool = True,
        allow_time_based_relaxation: bool = True,
        max_research_time_minutes: Optional[int] = None,
    ):
        """
        Initialize a CompletionCriteria instance.
        
        Args:
            min_sources: Minimum number of quality sources required
            min_source_diversity_score: Minimum score for diversity of sources (0-1)
            min_depth_coverage_score: Minimum score for depth of coverage (0-1)
            min_breadth_coverage_percentage: Minimum percentage of subtopics covered
            min_cross_verification_rate: Minimum rate of facts verified across sources
            min_strategic_objective_coverage: Minimum coverage of strategic objectives
            min_exploration_time_percentage: Minimum percentage of time spent on exploration
            min_deep_investigation_time_percentage: Minimum percentage of time on deep investigation
            content_type_distribution: Target distribution of content types (e.g., factual, analytical)
            required_content_types: Content types that must be included
            custom_criteria: List of custom criteria dictionaries
            task_specific_thresholds: Task-specific thresholds that override defaults
            allow_user_override: Whether user can override completion determination
            allow_time_based_relaxation: Whether to relax criteria based on research time
            max_research_time_minutes: Maximum research time in minutes
        """
        # Quantitative Metrics
        self.min_sources = min_sources
        self.min_source_diversity_score = min_source_diversity_score
        self.min_depth_coverage_score = min_depth_coverage_score
        self.min_breadth_coverage_percentage = min_breadth_coverage_percentage
        self.min_cross_verification_rate = min_cross_verification_rate
        self.min_strategic_objective_coverage = min_strategic_objective_coverage
        
        # Research Time Allocation
        self.min_exploration_time_percentage = min_exploration_time_percentage
        self.min_deep_investigation_time_percentage = min_deep_investigation_time_percentage
        
        # Content Type Requirements
        self.content_type_distribution = content_type_distribution or {
            "factual": 0.4,
            "conceptual": 0.3,
            "analytical": 0.2,
            "opinion": 0.1
        }
        self.required_content_types = required_content_types or ["factual", "analytical"]
        
        # Custom Criteria
        self.custom_criteria = custom_criteria or []
        
        # Task-specific Configuration
        self.task_specific_thresholds = task_specific_thresholds or {}
        
        # Override Controls
        self.allow_user_override = allow_user_override
        self.allow_time_based_relaxation = allow_time_based_relaxation
        self.max_research_time_minutes = max_research_time_minutes
        
        # Criteria Levels (which criteria are required vs. desired)
        self.criteria_levels = {
            "min_sources": CriteriaLevel.REQUIRED,
            "min_source_diversity_score": CriteriaLevel.IMPORTANT,
            "min_depth_coverage_score": CriteriaLevel.REQUIRED,
            "min_breadth_coverage_percentage": CriteriaLevel.IMPORTANT,
            "min_cross_verification_rate": CriteriaLevel.IMPORTANT,
            "min_strategic_objective_coverage": CriteriaLevel.REQUIRED,
            "min_exploration_time_percentage": CriteriaLevel.DESIRED,
            "min_deep_investigation_time_percentage": CriteriaLevel.DESIRED,
            "content_type_requirements": CriteriaLevel.IMPORTANT
        }
        
        # Apply task-specific overrides
        self._apply_task_specific_overrides()
    
    def _apply_task_specific_overrides(self) -> None:
        """Apply task-specific threshold overrides to the default criteria."""
        for key, value in self.task_specific_thresholds.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def evaluate(self, research_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate research results against completion criteria.
        
        Args:
            research_results: Dictionary of research results and metrics
            
        Returns:
            Evaluation results with completion status and details
        """
        evaluation = {
            "timestamp": time.time(),
            "metrics": {},
            "criteria_satisfaction": {},
            "overall_complete": False,
            "required_criteria_met": False,
            "important_criteria_met": False,
            "recommendation": ""
        }
        
        # Track which levels of criteria are met
        required_criteria_met = True
        important_criteria_met = True
        desired_criteria_met = True
        
        # Evaluate Source Metrics
        sources_count = len(research_results.get("sources", []))
        source_diversity = research_results.get("source_diversity_score", 0.0)
        
        evaluation["metrics"]["sources_count"] = sources_count
        evaluation["metrics"]["source_diversity"] = source_diversity
        
        # Check against thresholds
        evaluation["criteria_satisfaction"]["min_sources"] = sources_count >= self.min_sources
        evaluation["criteria_satisfaction"]["min_source_diversity"] = source_diversity >= self.min_source_diversity_score
        
        # Update criteria level tracking
        if not evaluation["criteria_satisfaction"]["min_sources"] and self.criteria_levels["min_sources"] == CriteriaLevel.REQUIRED:
            required_criteria_met = False
        elif not evaluation["criteria_satisfaction"]["min_sources"] and self.criteria_levels["min_sources"] == CriteriaLevel.IMPORTANT:
            important_criteria_met = False
        elif not evaluation["criteria_satisfaction"]["min_sources"] and self.criteria_levels["min_sources"] == CriteriaLevel.DESIRED:
            desired_criteria_met = False
            
        if not evaluation["criteria_satisfaction"]["min_source_diversity"] and self.criteria_levels["min_source_diversity_score"] == CriteriaLevel.REQUIRED:
            required_criteria_met = False
        elif not evaluation["criteria_satisfaction"]["min_source_diversity"] and self.criteria_levels["min_source_diversity_score"] == CriteriaLevel.IMPORTANT:
            important_criteria_met = False
        elif not evaluation["criteria_satisfaction"]["min_source_diversity"] and self.criteria_levels["min_source_diversity_score"] == CriteriaLevel.DESIRED:
            desired_criteria_met = False
        
        # Evaluate Coverage Metrics
        depth_coverage = research_results.get("depth_coverage_score", 0.0)
        breadth_coverage = research_results.get("breadth_coverage_percentage", 0.0)
        cross_verification = research_results.get("cross_verification_rate", 0.0)
        objective_coverage = research_results.get("strategic_objective_coverage", 0.0)
        
        evaluation["metrics"]["depth_coverage"] = depth_coverage
        evaluation["metrics"]["breadth_coverage"] = breadth_coverage
        evaluation["metrics"]["cross_verification"] = cross_verification
        evaluation["metrics"]["objective_coverage"] = objective_coverage
        
        # Check against thresholds
        evaluation["criteria_satisfaction"]["min_depth_coverage"] = depth_coverage >= self.min_depth_coverage_score
        evaluation["criteria_satisfaction"]["min_breadth_coverage"] = breadth_coverage >= self.min_breadth_coverage_percentage
        evaluation["criteria_satisfaction"]["min_cross_verification"] = cross_verification >= self.min_cross_verification_rate
        evaluation["criteria_satisfaction"]["min_objective_coverage"] = objective_coverage >= self.min_strategic_objective_coverage
        
        # Update criteria level tracking for coverage metrics
        self._update_criteria_tracking(evaluation["criteria_satisfaction"]["min_depth_coverage"], "min_depth_coverage_score", required_criteria_met, important_criteria_met, desired_criteria_met)
        self._update_criteria_tracking(evaluation["criteria_satisfaction"]["min_breadth_coverage"], "min_breadth_coverage_percentage", required_criteria_met, important_criteria_met, desired_criteria_met)
        self._update_criteria_tracking(evaluation["criteria_satisfaction"]["min_cross_verification"], "min_cross_verification_rate", required_criteria_met, important_criteria_met, desired_criteria_met)
        self._update_criteria_tracking(evaluation["criteria_satisfaction"]["min_objective_coverage"], "min_strategic_objective_coverage", required_criteria_met, important_criteria_met, desired_criteria_met)
        
        # Evaluate Time Allocation
        time_allocation = research_results.get("time_allocation", {})
        exploration_time_pct = time_allocation.get("exploration", 0.0)
        deep_investigation_time_pct = time_allocation.get("deep_investigation", 0.0)
        
        evaluation["metrics"]["exploration_time_percentage"] = exploration_time_pct
        evaluation["metrics"]["deep_investigation_time_percentage"] = deep_investigation_time_pct
        
        # Check against thresholds
        evaluation["criteria_satisfaction"]["min_exploration_time"] = exploration_time_pct >= self.min_exploration_time_percentage
        evaluation["criteria_satisfaction"]["min_deep_investigation_time"] = deep_investigation_time_pct >= self.min_deep_investigation_time_percentage
        
        # Update criteria level tracking for time allocation
        self._update_criteria_tracking(evaluation["criteria_satisfaction"]["min_exploration_time"], "min_exploration_time_percentage", required_criteria_met, important_criteria_met, desired_criteria_met)
        self._update_criteria_tracking(evaluation["criteria_satisfaction"]["min_deep_investigation_time"], "min_deep_investigation_time_percentage", required_criteria_met, important_criteria_met, desired_criteria_met)
        
        # Evaluate Content Type Distribution
        content_types = research_results.get("content_type_distribution", {})
        required_types_present = True
        
        for required_type in self.required_content_types:
            if required_type not in content_types or content_types.get(required_type, 0.0) < 0.05:
                required_types_present = False
                break
                
        evaluation["metrics"]["content_types"] = content_types
        evaluation["criteria_satisfaction"]["required_content_types"] = required_types_present
        
        # Update criteria level tracking for content types
        self._update_criteria_tracking(evaluation["criteria_satisfaction"]["required_content_types"], "content_type_requirements", required_criteria_met, important_criteria_met, desired_criteria_met)
        
        # Evaluate custom criteria
        for i, criterion in enumerate(self.custom_criteria):
            criterion_name = criterion.get("name", f"custom_criterion_{i}")
            criterion_threshold = criterion.get("threshold", 0.5)
            criterion_value = research_results.get(criterion_name, 0.0)
            criterion_level = CriteriaLevel[criterion.get("level", "IMPORTANT").upper()]
            
            evaluation["metrics"][criterion_name] = criterion_value
            criterion_satisfied = criterion_value >= criterion_threshold
            evaluation["criteria_satisfaction"][criterion_name] = criterion_satisfied
            
            # Update criteria level tracking
            if not criterion_satisfied:
                if criterion_level == CriteriaLevel.REQUIRED:
                    required_criteria_met = False
                elif criterion_level == CriteriaLevel.IMPORTANT:
                    important_criteria_met = False
                elif criterion_level == CriteriaLevel.DESIRED:
                    desired_criteria_met = False
        
        # Check if time-based relaxation should be applied
        research_time = research_results.get("research_time_minutes", 0)
        time_relaxation_applied = False
        
        if self.allow_time_based_relaxation and self.max_research_time_minutes and research_time >= self.max_research_time_minutes * 0.8:
            # If we've spent 80% of max time, relax importance of non-required criteria
            time_relaxation_applied = True
            important_criteria_met = True  # Ignore IMPORTANT criteria failures
            
            evaluation["time_relaxation_applied"] = True
            evaluation["time_relaxation_reason"] = f"Research time ({research_time} min) approaches max time limit ({self.max_research_time_minutes} min)"
        
        # Set overall completion status
        evaluation["required_criteria_met"] = required_criteria_met
        evaluation["important_criteria_met"] = important_criteria_met
        evaluation["desired_criteria_met"] = desired_criteria_met
        
        # Complete only if all required and important criteria are met
        evaluation["overall_complete"] = required_criteria_met and important_criteria_met
        
        # Generate recommendation
        if evaluation["overall_complete"]:
            if desired_criteria_met:
                evaluation["recommendation"] = "Research is complete and exceeds all criteria."
            else:
                evaluation["recommendation"] = "Research meets all required and important criteria. Consider additional research to address desired criteria."
        else:
            if not required_criteria_met:
                # List all required criteria that aren't met
                unmet_required = [key for key, value in evaluation["criteria_satisfaction"].items() 
                                 if not value and getattr(self.criteria_levels, key, CriteriaLevel.IMPORTANT) == CriteriaLevel.REQUIRED]
                
                evaluation["recommendation"] = f"Research incomplete. Must address these required criteria: {', '.join(unmet_required)}"
            elif not important_criteria_met:
                # List all important criteria that aren't met
                unmet_important = [key for key, value in evaluation["criteria_satisfaction"].items() 
                                  if not value and getattr(self.criteria_levels, key, CriteriaLevel.DESIRED) == CriteriaLevel.IMPORTANT]
                
                evaluation["recommendation"] = f"Research partially complete. Should address these important criteria: {', '.join(unmet_important)}"
        
        # Add improvements needed section
        evaluation["improvements_needed"] = self._generate_improvements_list(evaluation["criteria_satisfaction"], research_results)
        
        return evaluation
    
    def _update_criteria_tracking(self, criterion_satisfied: bool, criterion_name: str, 
                                 required_criteria_met: bool, important_criteria_met: bool, 
                                 desired_criteria_met: bool) -> Tuple[bool, bool, bool]:
        """
        Update the tracking of which level of criteria are met.
        
        Args:
            criterion_satisfied: Whether the criterion is satisfied
            criterion_name: Name of the criterion
            required_criteria_met: Current state of required criteria satisfaction
            important_criteria_met: Current state of important criteria satisfaction
            desired_criteria_met: Current state of desired criteria satisfaction
            
        Returns:
            Updated values for required_criteria_met, important_criteria_met, desired_criteria_met
        """
        if not criterion_satisfied:
            if self.criteria_levels.get(criterion_name, CriteriaLevel.IMPORTANT) == CriteriaLevel.REQUIRED:
                required_criteria_met = False
            elif self.criteria_levels.get(criterion_name, CriteriaLevel.IMPORTANT) == CriteriaLevel.IMPORTANT:
                important_criteria_met = False
            elif self.criteria_levels.get(criterion_name, CriteriaLevel.IMPORTANT) == CriteriaLevel.DESIRED:
                desired_criteria_met = False
                
        return required_criteria_met, important_criteria_met, desired_criteria_met
    
    def _generate_improvements_list(self, criteria_satisfaction: Dict[str, bool], 
                                  research_results: Dict[str, Any]) -> List[str]:
        """
        Generate a list of specific improvements needed based on unmet criteria.
        
        Args:
            criteria_satisfaction: Dictionary of criteria satisfaction status
            research_results: Dictionary of research results and metrics
            
        Returns:
            List of specific improvement recommendations
        """
        improvements = []
        
        # Check sources
        if not criteria_satisfaction.get("min_sources", True):
            current = len(research_results.get("sources", []))
            improvements.append(f"Increase number of sources from {current} to at least {self.min_sources}")
            
        # Check source diversity
        if not criteria_satisfaction.get("min_source_diversity", True):
            current = research_results.get("source_diversity_score", 0.0)
            improvements.append(f"Improve source diversity from {current:.2f} to at least {self.min_source_diversity_score:.2f} by including more diverse source types")
            
        # Check depth coverage
        if not criteria_satisfaction.get("min_depth_coverage", True):
            current = research_results.get("depth_coverage_score", 0.0)
            improvements.append(f"Increase depth of coverage from {current:.2f} to at least {self.min_depth_coverage_score:.2f} by exploring key topics in more detail")
            
        # Check breadth coverage
        if not criteria_satisfaction.get("min_breadth_coverage", True):
            current = research_results.get("breadth_coverage_percentage", 0.0)
            improvements.append(f"Expand breadth of coverage from {current:.2f}% to at least {self.min_breadth_coverage_percentage:.2f}% by addressing more subtopics")
            
        # Check cross-verification
        if not criteria_satisfaction.get("min_cross_verification", True):
            current = research_results.get("cross_verification_rate", 0.0)
            improvements.append(f"Increase cross-verification rate from {current:.2f} to at least {self.min_cross_verification_rate:.2f} by verifying facts across multiple sources")
            
        # Check strategic objective coverage
        if not criteria_satisfaction.get("min_objective_coverage", True):
            current = research_results.get("strategic_objective_coverage", 0.0)
            improvements.append(f"Improve strategic objective coverage from {current:.2f} to at least {self.min_strategic_objective_coverage:.2f} by addressing key research objectives")
            
        # Check required content types
        if not criteria_satisfaction.get("required_content_types", True):
            content_types = research_results.get("content_type_distribution", {})
            missing_types = [t for t in self.required_content_types if t not in content_types or content_types.get(t, 0.0) < 0.05]
            improvements.append(f"Include these missing content types: {', '.join(missing_types)}")
            
        return improvements
        
    def get_criteria_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the completion criteria.
        
        Returns:
            Dictionary summarizing the criteria
        """
        return {
            "quantitative_metrics": {
                "min_sources": self.min_sources,
                "min_source_diversity_score": self.min_source_diversity_score,
                "min_depth_coverage_score": self.min_depth_coverage_score,
                "min_breadth_coverage_percentage": self.min_breadth_coverage_percentage,
                "min_cross_verification_rate": self.min_cross_verification_rate,
                "min_strategic_objective_coverage": self.min_strategic_objective_coverage
            },
            "time_allocation": {
                "min_exploration_time_percentage": self.min_exploration_time_percentage,
                "min_deep_investigation_time_percentage": self.min_deep_investigation_time_percentage,
                "max_research_time_minutes": self.max_research_time_minutes
            },
            "content_requirements": {
                "content_type_distribution": self.content_type_distribution,
                "required_content_types": self.required_content_types
            },
            "criteria_levels": {
                key: level.name for key, level in self.criteria_levels.items()
            },
            "custom_criteria": self.custom_criteria,
            "task_specific_overrides": self.task_specific_thresholds,
            "override_controls": {
                "allow_user_override": self.allow_user_override,
                "allow_time_based_relaxation": self.allow_time_based_relaxation
            }
        }
    
    def to_json(self) -> str:
        """
        Convert completion criteria to JSON.
        
        Returns:
            JSON string representation of criteria
        """
        summary = self.get_criteria_summary()
        
        # Convert enum values to strings
        for key, level in summary["criteria_levels"].items():
            if isinstance(level, CriteriaLevel):
                summary["criteria_levels"][key] = level.name
        
        return json.dumps(summary, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'CompletionCriteria':
        """
        Create CompletionCriteria from JSON string.
        
        Args:
            json_str: JSON string representation of criteria
            
        Returns:
            CompletionCriteria instance
        """
        data = json.loads(json_str)
        
        # Extract metric thresholds
        quantitative_metrics = data.get("quantitative_metrics", {})
        min_sources = quantitative_metrics.get("min_sources", 5)
        min_source_diversity_score = quantitative_metrics.get("min_source_diversity_score", 0.6)
        min_depth_coverage_score = quantitative_metrics.get("min_depth_coverage_score", 0.7)
        min_breadth_coverage_percentage = quantitative_metrics.get("min_breadth_coverage_percentage", 80.0)
        min_cross_verification_rate = quantitative_metrics.get("min_cross_verification_rate", 0.5)
        min_strategic_objective_coverage = quantitative_metrics.get("min_strategic_objective_coverage", 0.8)
        
        # Extract time allocation
        time_allocation = data.get("time_allocation", {})
        min_exploration_time_percentage = time_allocation.get("min_exploration_time_percentage", 20.0)
        min_deep_investigation_time_percentage = time_allocation.get("min_deep_investigation_time_percentage", 40.0)
        max_research_time_minutes = time_allocation.get("max_research_time_minutes")
        
        # Extract content requirements
        content_requirements = data.get("content_requirements", {})
        content_type_distribution = content_requirements.get("content_type_distribution")
        required_content_types = content_requirements.get("required_content_types")
        
        # Extract custom criteria and task-specific overrides
        custom_criteria = data.get("custom_criteria", [])
        task_specific_thresholds = data.get("task_specific_overrides", {})
        
        # Extract override controls
        override_controls = data.get("override_controls", {})
        allow_user_override = override_controls.get("allow_user_override", True)
        allow_time_based_relaxation = override_controls.get("allow_time_based_relaxation", True)
        
        # Create instance
        instance = cls(
            min_sources=min_sources,
            min_source_diversity_score=min_source_diversity_score,
            min_depth_coverage_score=min_depth_coverage_score,
            min_breadth_coverage_percentage=min_breadth_coverage_percentage,
            min_cross_verification_rate=min_cross_verification_rate,
            min_strategic_objective_coverage=min_strategic_objective_coverage,
            min_exploration_time_percentage=min_exploration_time_percentage,
            min_deep_investigation_time_percentage=min_deep_investigation_time_percentage,
            content_type_distribution=content_type_distribution,
            required_content_types=required_content_types,
            custom_criteria=custom_criteria,
            task_specific_thresholds=task_specific_thresholds,
            allow_user_override=allow_user_override,
            allow_time_based_relaxation=allow_time_based_relaxation,
            max_research_time_minutes=max_research_time_minutes
        )
        
        # Set criteria levels if provided
        if "criteria_levels" in data:
            for key, level_name in data["criteria_levels"].items():
                try:
                    instance.criteria_levels[key] = CriteriaLevel[level_name]
                except (KeyError, ValueError):
                    # Use default level if invalid
                    instance.criteria_levels[key] = CriteriaLevel.IMPORTANT
        
        return instance
    
    @classmethod
    def create_task_specific(cls, task_type: str, complexity: str = "medium") -> 'CompletionCriteria':
        """
        Create task-specific completion criteria based on task type and complexity.
        
        Args:
            task_type: Type of research task (e.g., "factual", "exploratory", "analytical")
            complexity: Task complexity (e.g., "low", "medium", "high")
            
        Returns:
            CompletionCriteria instance tailored to task type and complexity
        """
        # Base criteria
        base_criteria = cls()
        
        # Adjust based on task type
        if task_type == "factual":
            # Factual research prioritizes verification and source reliability
            base_criteria.min_cross_verification_rate = 0.7
            base_criteria.min_sources = 7
            base_criteria.criteria_levels["min_cross_verification_rate"] = CriteriaLevel.REQUIRED
            base_criteria.content_type_distribution = {
                "factual": 0.7,
                "conceptual": 0.1,
                "analytical": 0.1,
                "opinion": 0.1
            }
            base_criteria.required_content_types = ["factual"]
            
        elif task_type == "exploratory":
            # Exploratory research prioritizes breadth and diverse sources
            base_criteria.min_breadth_coverage_percentage = 90.0
            base_criteria.min_source_diversity_score = 0.8
            base_criteria.min_exploration_time_percentage = 40.0
            base_criteria.criteria_levels["min_breadth_coverage_percentage"] = CriteriaLevel.REQUIRED
            base_criteria.criteria_levels["min_source_diversity_score"] = CriteriaLevel.REQUIRED
            base_criteria.content_type_distribution = {
                "factual": 0.3,
                "conceptual": 0.4,
                "analytical": 0.2,
                "opinion": 0.1
            }
            
        elif task_type == "analytical":
            # Analytical research prioritizes depth and analysis
            base_criteria.min_depth_coverage_score = 0.8
            base_criteria.min_deep_investigation_time_percentage = 60.0
            base_criteria.criteria_levels["min_depth_coverage_score"] = CriteriaLevel.REQUIRED
            base_criteria.content_type_distribution = {
                "factual": 0.3,
                "conceptual": 0.2,
                "analytical": 0.5,
                "opinion": 0.0
            }
            base_criteria.required_content_types = ["factual", "analytical"]
            
        # Adjust based on complexity
        if complexity == "low":
            # Lower thresholds for simpler tasks
            base_criteria.min_sources -= 2
            base_criteria.min_breadth_coverage_percentage -= 10.0
            base_criteria.min_depth_coverage_score -= 0.1
            base_criteria.max_research_time_minutes = 15
            
        elif complexity == "high":
            # Higher thresholds for complex tasks
            base_criteria.min_sources += 3
            base_criteria.min_breadth_coverage_percentage += 5.0
            base_criteria.min_depth_coverage_score += 0.1
            base_criteria.min_strategic_objective_coverage += 0.1
            base_criteria.max_research_time_minutes = 60
            
        else:  # medium complexity
            base_criteria.max_research_time_minutes = 30
            
        return base_criteria