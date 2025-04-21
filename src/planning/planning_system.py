"""
Planning System Module

This module provides the PlanningSystem class which coordinates between strategic
and tactical planners to manage the overall research planning process.
"""

from typing import Dict, List, Any, Optional
import time
import uuid

from langchain_core.language_models import BaseChatModel

from .strategic_planner import StrategicPlanner
from .tactical_planner import TacticalPlanner


class PlanningSystem:
    """
    Planning System coordinates between strategic and tactical planners.
    
    This system manages the overall research planning process, ensuring alignment
    between high-level objectives and specific execution tasks.
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        strategic_planner: Optional[StrategicPlanner] = None,
        tactical_planner: Optional[TacticalPlanner] = None,
    ):
        """
        Initialize a PlanningSystem instance.
        
        Args:
            llm: The language model to use
            strategic_planner: Optional custom strategic planner (creates default if None)
            tactical_planner: Optional custom tactical planner (creates default if None)
        """
        self.llm = llm
        self.strategic_planner = strategic_planner or StrategicPlanner(llm=llm)
        self.tactical_planner = tactical_planner or TacticalPlanner(llm=llm)
        
        # Store active plans
        self.active_strategic_plan = None
        self.active_tactical_plans = {}  # Map of task IDs to tactical plans
    
    async def create_research_plan(
        self,
        research_request: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a comprehensive research plan for a request.
        
        This creates both strategic and initial tactical plans.
        
        Args:
            research_request: The research request to plan for
            context: Optional additional context
            
        Returns:
            Complete research plan with both strategic and tactical components
        """
        # Create strategic plan
        strategic_plan = await self.strategic_planner.create_plan(
            task=research_request,
            context=context
        )
        
        if strategic_plan.get("status") == "error":
            return {
                "status": "error",
                "error": "Failed to create strategic plan",
                "strategic_plan_error": strategic_plan
            }
        
        # Set as active strategic plan
        self.active_strategic_plan = strategic_plan
        
        # Create tactical plans for each phase of the strategic plan
        tactical_plans = []
        if "research_strategy" in strategic_plan:
            for phase in strategic_plan["research_strategy"]:
                phase_name = phase.get("phase", "")
                phase_focus = phase.get("focus", "")
                
                # Create tactical plan for this phase
                tactical_plan = await self.tactical_planner.create_plan(
                    task=f"Execute research phase: {phase_name} - {phase_focus}",
                    context={
                        "strategic_plan": strategic_plan,
                        "phase": phase
                    }
                )
                
                # Add to tactical plans
                tactical_plans.append(tactical_plan)
                
                # Add to active tactical plans
                task_id = tactical_plan.get("id")
                if task_id:
                    self.active_tactical_plans[task_id] = tactical_plan
        
        # Build combined research plan
        complete_plan = {
            "id": str(uuid.uuid4()),
            "type": "complete_research_plan",
            "research_request": research_request,
            "created_at": time.time(),
            "strategic_plan": strategic_plan,
            "tactical_plans": tactical_plans,
            "status": "created"
        }
        
        return complete_plan
    
    async def refine_strategic_plan(
        self,
        plan_id: str,
        feedback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Refine an existing strategic plan based on feedback.
        
        Args:
            plan_id: The ID of the strategic plan to refine
            feedback: Feedback for refinement
            
        Returns:
            The refined strategic plan
        """
        # Create update context from feedback
        updates = {
            "refinement_feedback": feedback
        }
        
        # Call strategic planner to update
        refined_plan = await self.strategic_planner.update_plan(plan_id, updates)
        
        # Update active strategic plan if successful
        if refined_plan.get("status") != "error":
            self.active_strategic_plan = refined_plan
        
        return refined_plan
    
    async def create_tactical_plan(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a tactical plan for a specific task.
        
        Args:
            task: The task to plan for
            context: Optional additional context
            
        Returns:
            Tactical plan
        """
        # Add active strategic plan to context if available
        full_context = context or {}
        if self.active_strategic_plan and "strategic_plan" not in full_context:
            full_context["strategic_plan"] = self.active_strategic_plan
        
        # Create tactical plan
        tactical_plan = await self.tactical_planner.create_plan(task, full_context)
        
        # Add to active tactical plans if successful
        if tactical_plan.get("status") != "error":
            task_id = tactical_plan.get("id")
            if task_id:
                self.active_tactical_plans[task_id] = tactical_plan
        
        return tactical_plan
    
    async def adapt_tactical_plan(
        self,
        plan_id: str,
        feedback: Dict[str, Any],
        execution_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adapt a tactical plan based on execution feedback.
        
        Args:
            plan_id: The ID of the tactical plan to adapt
            feedback: Feedback on execution
            execution_state: Current execution state
            
        Returns:
            The adapted tactical plan
        """
        # Call tactical planner to adapt
        adapted_plan = await self.tactical_planner.adapt_to_feedback(
            plan_id, feedback, execution_state
        )
        
        # Update active tactical plans if successful
        if adapted_plan.get("status") != "error":
            self.active_tactical_plans[plan_id] = adapted_plan
        
        return adapted_plan
    
    async def evaluate_research_progress(
        self,
        execution_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate overall research progress based on execution results.
        
        Args:
            execution_results: Results from executing the research plans
            
        Returns:
            Comprehensive evaluation of research progress
        """
        # Create container for all evaluations
        evaluations = {
            "id": str(uuid.uuid4()),
            "type": "research_progress_evaluation",
            "evaluated_at": time.time(),
            "strategic_evaluation": None,
            "tactical_evaluations": [],
            "overall_assessment": None
        }
        
        # Evaluate strategic plan if available
        if self.active_strategic_plan:
            strategic_eval = await self.strategic_planner.evaluate_plan(
                self.active_strategic_plan.get("id", ""),
                execution_results
            )
            evaluations["strategic_evaluation"] = strategic_eval
        
        # Evaluate tactical plans
        for plan_id, plan in self.active_tactical_plans.items():
            # Extract results relevant to this tactical plan
            plan_results = {}
            if "tactical_results" in execution_results:
                plan_results = execution_results["tactical_results"].get(plan_id, {})
            
            # Evaluate tactical plan
            tactical_eval = await self.tactical_planner.evaluate_plan(
                plan_id,
                plan_results
            )
            
            evaluations["tactical_evaluations"].append(tactical_eval)
        
        # Calculate overall research progress
        progress_percentage = self._calculate_overall_progress(evaluations)
        
        # Create overall assessment
        overall_assessment = {
            "progress_percentage": progress_percentage,
            "strategic_alignment": self._assess_strategic_alignment(evaluations),
            "tactical_effectiveness": self._assess_tactical_effectiveness(evaluations),
            "remaining_work": self._identify_remaining_work(evaluations),
            "next_steps": self._recommend_next_steps(evaluations)
        }
        
        evaluations["overall_assessment"] = overall_assessment
        
        return evaluations
    
    async def get_execution_tasks(
        self,
        tactical_plan_id: Optional[str] = None,
        execution_state: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get execution tasks from tactical plans.
        
        Args:
            tactical_plan_id: Optional specific tactical plan to get tasks from
            execution_state: Optional current execution state to consider
            
        Returns:
            List of execution tasks
        """
        tasks = []
        
        # Function to extract and process tasks from a plan
        def extract_tasks(plan):
            if "tasks" not in plan:
                return []
            
            plan_tasks = []
            for task in plan["tasks"]:
                # Copy task
                execution_task = dict(task)
                
                # Add plan reference
                execution_task["plan_id"] = plan.get("id", "")
                execution_task["plan_type"] = plan.get("type", "")
                
                # Check dependencies if execution state is provided
                if execution_state and "dependencies" in task:
                    # Check if all dependencies are completed
                    dependencies_met = True
                    for dep_id in task["dependencies"]:
                        if not execution_state.get("completed_tasks", {}).get(dep_id, False):
                            dependencies_met = False
                            break
                    
                    execution_task["dependencies_met"] = dependencies_met
                
                plan_tasks.append(execution_task)
            
            return plan_tasks
        
        # If specific tactical plan requested
        if tactical_plan_id:
            if tactical_plan_id in self.active_tactical_plans:
                tasks.extend(extract_tasks(self.active_tactical_plans[tactical_plan_id]))
        else:
            # Get tasks from all active tactical plans
            for plan_id, plan in self.active_tactical_plans.items():
                tasks.extend(extract_tasks(plan))
        
        # Sort tasks by priority (highest first)
        tasks.sort(key=lambda x: -(x.get("priority", 0)))
        
        return tasks
    
    def get_active_plans(self) -> Dict[str, Any]:
        """
        Get all active plans.
        
        Returns:
            Dictionary with active strategic and tactical plans
        """
        return {
            "strategic_plan": self.active_strategic_plan,
            "tactical_plans": self.active_tactical_plans
        }
    
    def _calculate_overall_progress(self, evaluations: Dict[str, Any]) -> float:
        """
        Calculate overall research progress percentage.
        
        Args:
            evaluations: Evaluation results
            
        Returns:
            Progress percentage (0-100)
        """
        # Initialize variables
        total_weight = 0
        weighted_progress = 0
        
        # Consider strategic evaluation (weight: 40%)
        if evaluations.get("strategic_evaluation"):
            strategic_eval = evaluations["strategic_evaluation"]
            if "overall_assessment" in strategic_eval and "success_rating" in strategic_eval["overall_assessment"]:
                # Convert 0-10 rating to 0-100 percentage
                strategic_progress = strategic_eval["overall_assessment"]["success_rating"] * 10
                weighted_progress += strategic_progress * 0.4
                total_weight += 0.4
        
        # Consider tactical evaluations (combined weight: 60%)
        tactical_evals = evaluations.get("tactical_evaluations", [])
        if tactical_evals:
            tactical_weight_per_plan = 0.6 / len(tactical_evals)
            
            for tactical_eval in tactical_evals:
                if "overall_assessment" in tactical_eval and "success_rating" in tactical_eval["overall_assessment"]:
                    # Convert 0-10 rating to 0-100 percentage
                    tactical_progress = tactical_eval["overall_assessment"]["success_rating"] * 10
                    weighted_progress += tactical_progress * tactical_weight_per_plan
                    total_weight += tactical_weight_per_plan
        
        # Calculate final percentage
        if total_weight > 0:
            return weighted_progress / total_weight
        else:
            return 0.0
    
    def _assess_strategic_alignment(self, evaluations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess alignment between strategic objectives and tactical execution.
        
        Args:
            evaluations: Evaluation results
            
        Returns:
            Strategic alignment assessment
        """
        # Default alignment assessment
        alignment = {
            "overall_alignment": "medium",
            "well_aligned_areas": [],
            "misaligned_areas": [],
            "assessment": "Insufficient data to assess strategic alignment"
        }
        
        # Extract strategic objectives
        strategic_objectives = []
        if self.active_strategic_plan and "objectives" in self.active_strategic_plan:
            strategic_objectives = self.active_strategic_plan["objectives"]
        
        # If no strategic objectives, return default
        if not strategic_objectives:
            return alignment
        
        # Map objectives to tactical tasks
        objective_to_tasks = {}
        for obj in strategic_objectives:
            obj_id = obj.get("id", "")
            if obj_id:
                objective_to_tasks[obj_id] = []
        
        # Populate tasks for each objective
        for plan_id, plan in self.active_tactical_plans.items():
            if "tasks" in plan:
                for task in plan["tasks"]:
                    obj_id = task.get("objective_id", "")
                    if obj_id in objective_to_tasks:
                        objective_to_tasks[obj_id].append(task)
        
        # Analyze alignment
        well_aligned = []
        misaligned = []
        
        for obj in strategic_objectives:
            obj_id = obj.get("id", "")
            obj_description = obj.get("description", "")
            
            if obj_id:
                tasks = objective_to_tasks.get(obj_id, [])
                
                # Check task count and priority
                if len(tasks) == 0:
                    misaligned.append({
                        "objective_id": obj_id,
                        "objective_description": obj_description,
                        "issue": "No tactical tasks addressing this objective"
                    })
                elif all(task.get("priority", 0) < 5 for task in tasks):
                    misaligned.append({
                        "objective_id": obj_id,
                        "objective_description": obj_description,
                        "issue": "All tasks for this objective have low priority"
                    })
                else:
                    well_aligned.append({
                        "objective_id": obj_id,
                        "objective_description": obj_description,
                        "task_count": len(tasks)
                    })
        
        # Determine overall alignment
        if len(misaligned) == 0:
            overall_alignment = "high"
        elif len(misaligned) <= len(strategic_objectives) * 0.25:
            overall_alignment = "medium-high"
        elif len(misaligned) <= len(strategic_objectives) * 0.5:
            overall_alignment = "medium"
        elif len(misaligned) <= len(strategic_objectives) * 0.75:
            overall_alignment = "medium-low"
        else:
            overall_alignment = "low"
        
        # Create assessment text
        assessment = f"Strategic alignment is {overall_alignment}. "
        if well_aligned:
            assessment += f"{len(well_aligned)} objectives are well-addressed by tactical plans. "
        if misaligned:
            assessment += f"{len(misaligned)} objectives have alignment issues. "
        
        return {
            "overall_alignment": overall_alignment,
            "well_aligned_areas": well_aligned,
            "misaligned_areas": misaligned,
            "assessment": assessment
        }
    
    def _assess_tactical_effectiveness(self, evaluations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess effectiveness of tactical execution.
        
        Args:
            evaluations: Evaluation results
            
        Returns:
            Tactical effectiveness assessment
        """
        # Extract tactical evaluations
        tactical_evals = evaluations.get("tactical_evaluations", [])
        
        # Default effectiveness assessment
        effectiveness = {
            "overall_effectiveness": "medium",
            "strengths": [],
            "weaknesses": [],
            "assessment": "Insufficient data to assess tactical effectiveness"
        }
        
        if not tactical_evals:
            return effectiveness
        
        # Analyze tactical evaluations
        strengths = []
        weaknesses = []
        
        # Aggregate ratings across aspects
        aspect_ratings = {
            "task_sequencing": [],
            "search_strategy": [],
            "information_extraction": [],
            "time_efficiency": []
        }
        
        for eval_data in tactical_evals:
            if "tactical_effectiveness" in eval_data:
                effectiveness_data = eval_data["tactical_effectiveness"]
                
                # Collect ratings for each aspect
                for aspect in aspect_ratings:
                    if aspect in effectiveness_data:
                        aspect_ratings[aspect].append(effectiveness_data[aspect])
                
                # Extract improvement suggestions
                if "improvement_suggestions" in eval_data:
                    for suggestion in eval_data["improvement_suggestions"]:
                        if "aspect" in suggestion and "suggestion" in suggestion:
                            weaknesses.append({
                                "aspect": suggestion["aspect"],
                                "issue": suggestion["suggestion"]
                            })
        
        # Calculate average ratings and identify strengths/weaknesses
        for aspect, ratings in aspect_ratings.items():
            if ratings:
                avg_rating = sum(ratings) / len(ratings)
                
                if avg_rating >= 8:
                    strengths.append({
                        "aspect": aspect,
                        "rating": avg_rating,
                        "assessment": f"Strong performance in {aspect}"
                    })
                elif avg_rating <= 5:
                    if not any(w["aspect"] == aspect for w in weaknesses):
                        weaknesses.append({
                            "aspect": aspect,
                            "rating": avg_rating,
                            "assessment": f"Needs improvement in {aspect}"
                        })
        
        # Determine overall effectiveness
        overall_ratings = []
        for ratings in aspect_ratings.values():
            if ratings:
                overall_ratings.extend(ratings)
        
        if overall_ratings:
            avg_overall = sum(overall_ratings) / len(overall_ratings)
            
            if avg_overall >= 8:
                overall_effectiveness = "high"
            elif avg_overall >= 6:
                overall_effectiveness = "medium-high"
            elif avg_overall >= 4:
                overall_effectiveness = "medium"
            elif avg_overall >= 2:
                overall_effectiveness = "medium-low"
            else:
                overall_effectiveness = "low"
        else:
            overall_effectiveness = "medium"  # Default if no ratings available
        
        # Create assessment text
        assessment = f"Tactical effectiveness is {overall_effectiveness}. "
        if strengths:
            assessment += f"Key strengths include {', '.join(s['aspect'] for s in strengths[:3])}. "
        if weaknesses:
            assessment += f"Areas for improvement include {', '.join(w['aspect'] for w in weaknesses[:3])}. "
        
        return {
            "overall_effectiveness": overall_effectiveness,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "assessment": assessment
        }
    
    def _identify_remaining_work(self, evaluations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify remaining work to complete the research.
        
        Args:
            evaluations: Evaluation results
            
        Returns:
            List of remaining work items
        """
        remaining_work = []
        
        # Check strategic evaluation for unfulfilled objectives
        if evaluations.get("strategic_evaluation") and "objectives_assessment" in evaluations["strategic_evaluation"]:
            for obj_assessment in evaluations["strategic_evaluation"]["objectives_assessment"]:
                completion = obj_assessment.get("completion_percentage", 0)
                
                if completion < 90:
                    remaining_work.append({
                        "type": "strategic_objective",
                        "id": obj_assessment.get("objective_id", ""),
                        "description": f"Complete objective: {obj_assessment.get('objective', '')}",
                        "completion_percentage": completion,
                        "priority": "high" if completion < 50 else "medium"
                    })
        
        # Check tactical evaluations for incomplete tasks
        for eval_data in evaluations.get("tactical_evaluations", []):
            if "task_assessments" in eval_data:
                for task_assessment in eval_data["task_assessments"]:
                    completion = task_assessment.get("completion_percentage", 0)
                    
                    if completion < 100 and task_assessment.get("completion_status") != "completed":
                        remaining_work.append({
                            "type": "tactical_task",
                            "id": task_assessment.get("task_id", ""),
                            "description": f"Complete task: {task_assessment.get('task_id', '')}",
                            "completion_percentage": completion,
                            "priority": "high" if completion < 50 else "medium"
                        })
            
            # Check for unmet criteria
            if "completion_criteria_assessment" in eval_data and "criteria_unmet" in eval_data["completion_criteria_assessment"]:
                for criterion in eval_data["completion_criteria_assessment"]["criteria_unmet"]:
                    remaining_work.append({
                        "type": "completion_criterion",
                        "description": f"Satisfy criterion: {criterion}",
                        "completion_percentage": 0,
                        "priority": "high"
                    })
        
        # Sort by priority and completion percentage
        def priority_value(item):
            if item.get("priority") == "high":
                return 3
            elif item.get("priority") == "medium":
                return 2
            else:
                return 1
        
        remaining_work.sort(key=lambda x: (priority_value(x), -x.get("completion_percentage", 0)), reverse=True)
        
        return remaining_work
    
    def _recommend_next_steps(self, evaluations: Dict[str, Any]) -> List[str]:
        """
        Recommend next steps based on evaluation results.
        
        Args:
            evaluations: Evaluation results
            
        Returns:
            List of recommended next steps
        """
        next_steps = []
        
        # Get remaining work
        remaining_work = self._identify_remaining_work(evaluations)
        
        # Prioritize high-priority remaining work
        high_priority_work = [w for w in remaining_work if w.get("priority") == "high"]
        for work_item in high_priority_work[:3]:  # Top 3 high-priority items
            next_steps.append(f"Address {work_item['type']}: {work_item['description']}")
        
        # Check strategic alignment
        strategic_alignment = self._assess_strategic_alignment(evaluations)
        if strategic_alignment.get("overall_alignment") in ["low", "medium-low"]:
            next_steps.append("Revise tactical plans to better align with strategic objectives")
        
        # Check tactical effectiveness
        tactical_effectiveness = self._assess_tactical_effectiveness(evaluations)
        if tactical_effectiveness.get("overall_effectiveness") in ["low", "medium-low"]:
            weakness_aspects = [w["aspect"] for w in tactical_effectiveness.get("weaknesses", [])]
            if weakness_aspects:
                next_steps.append(f"Improve tactical execution in: {', '.join(weakness_aspects[:3])}")
        
        # If overall progress is low, consider refinement
        overall_progress = evaluations.get("overall_assessment", {}).get("progress_percentage", 0)
        if overall_progress < 30:
            next_steps.append("Consider refining the strategic plan to better address the research request")
        
        # Ensure we have at least one next step
        if not next_steps:
            if remaining_work:
                next_steps.append(f"Continue working on remaining tasks ({len(remaining_work)} remaining)")
            else:
                next_steps.append("Finalize research and prepare comprehensive output")
        
        return next_steps