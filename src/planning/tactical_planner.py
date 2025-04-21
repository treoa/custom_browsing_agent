"""
Tactical Planner Module

This module provides the TacticalPlanner class which creates detailed action plans
for executing strategic research objectives.
"""

from typing import Dict, List, Any, Optional
import json
import time
import uuid

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from .base_planner import BasePlanner


class TacticalPlanner(BasePlanner):
    """
    Tactical Planner creates detailed action plans for research execution.
    
    This planner converts strategic objectives into concrete, actionable steps,
    specifying the exact operations that agents should perform.
    """
    
    TACTICAL_PLANNER_PROMPT = """You are a Tactical Planner in a multi-agent research system.
Your role is to translate strategic research objectives into concrete, detailed action plans. You should:
1. Break down high-level objectives into specific, actionable tasks
2. Sequence tasks in optimal order with clear dependencies
3. Specify exact operations for research agents to perform
4. Define precise criteria for task completion
5. Allocate appropriate time and resources to each task
6. Create detailed search strategies and queries for web research
7. Specify data collection and extraction methodologies
8. Design verification and validation steps

Your tactical plans should be detailed, concrete, and immediately actionable.
They should leave no ambiguity about what actions to take or how to determine success.
"""
    
    def __init__(
        self,
        llm: BaseChatModel,
        name: str = "TacticalPlanner",
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize a TacticalPlanner instance.
        
        Args:
            llm: The language model to use
            name: The name/identifier for this planner (default: "TacticalPlanner")
            system_prompt: Optional custom system prompt (uses default if None)
        """
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt or self.TACTICAL_PLANNER_PROMPT,
        )
    
    async def create_plan(
        self, 
        task: str, 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create a tactical plan for a specific research task.
        
        Args:
            task: The specific research task to plan for
            context: Additional context including strategic plan
            
        Returns:
            A structured tactical plan
        """
        context = context or {}
        strategic_plan = context.get("strategic_plan", {})
        
        # Create plan request prompt
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
Create a detailed tactical plan for the following research task:
"{task}"

{self._format_strategic_plan(strategic_plan)}
{self._format_additional_context(context)}

Your tactical plan should include:
1. Specific, actionable tasks to accomplish the research objectives
2. Detailed search queries and web research strategies
3. Precise data extraction and organization methods
4. Verification and validation steps
5. Task dependencies and optimal sequence
6. Completion criteria for each task

Format your response as a JSON object with the following structure:
{{
    "tasks": [
        {{
            "id": "task_1",
            "description": "Detailed description of the task",
            "objective_id": "Related strategic objective ID",
            "search_queries": ["query 1", "query 2", ...],
            "information_extraction": {{
                "data_points": ["data point 1", "data point 2", ...],
                "methodology": "How to extract the information"
            }},
            "verification_steps": ["step 1", "step 2", ...],
            "completion_criteria": ["criterion 1", "criterion 2", ...],
            "estimated_time_minutes": time_estimate,
            "dependencies": ["task_2", "task_3", ...],
            "priority": 1-10
        }},
        ...
    ],
    "execution_sequence": [
        ["task_1"],  // Tasks that can be executed in parallel
        ["task_2", "task_3"],  // Next set of tasks after task_1 completes
        ...
    ],
    "contingency_plans": [
        {{
            "scenario": "Description of potential issue",
            "detection": "How to detect this issue",
            "actions": ["action 1", "action 2", ...]
        }},
        ...
    ],
    "success_criteria": {{
        "all_tasks_completed": true/false,
        "minimum_tasks_required": ["task_1", "task_2", ...],
        "information_completeness": "Description of what complete information looks like"
    }}
}}
""")
        ]
        
        # Get response from LLM
        response = await self.llm.ainvoke(messages)
        
        try:
            # Extract JSON from the response content
            plan_json = response.content
            if "```json" in plan_json:
                plan_json = plan_json.split("```json")[1].split("```")[0].strip()
            elif "```" in plan_json:
                plan_json = plan_json.split("```")[1].split("```")[0].strip()
                
            tactical_plan = json.loads(plan_json)
            
            # Add metadata
            plan_id = str(uuid.uuid4())
            tactical_plan["id"] = plan_id
            tactical_plan["type"] = "tactical"
            tactical_plan["task"] = task
            if strategic_plan:
                tactical_plan["strategic_plan_id"] = strategic_plan.get("id")
            tactical_plan["created_at"] = time.time()
            tactical_plan["status"] = "created"
            
            # Add to history
            self.add_to_history(tactical_plan)
            
            return tactical_plan
        except Exception as e:
            # Handle parsing errors
            error_plan = {
                "id": str(uuid.uuid4()),
                "type": "tactical",
                "task": task,
                "created_at": time.time(),
                "status": "error",
                "error": str(e),
                "raw_response": response.content
            }
            
            # Add to history
            self.add_to_history(error_plan)
            
            return error_plan
    
    async def update_plan(self, plan_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing tactical plan.
        
        Args:
            plan_id: The ID of the plan to update
            updates: The updates to apply to the plan
            
        Returns:
            The updated plan
        """
        # Get the existing plan
        existing_plan = self.get_plan(plan_id)
        if not existing_plan:
            return {
                "status": "error",
                "error": f"Plan with ID {plan_id} not found"
            }
        
        # Create update prompt
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
Update the following tactical research plan:

CURRENT PLAN:
{json.dumps(existing_plan, indent=2)}

REQUESTED UPDATES:
{json.dumps(updates, indent=2)}

Maintain the overall structure while incorporating these updates.
Keep any parts of the original plan that don't need to be changed.

Format your response as a complete JSON object with the same structure as the original plan.
""")
        ]
        
        # Get response from LLM
        response = await self.llm.ainvoke(messages)
        
        try:
            # Extract JSON from the response content
            plan_json = response.content
            if "```json" in plan_json:
                plan_json = plan_json.split("```json")[1].split("```")[0].strip()
            elif "```" in plan_json:
                plan_json = plan_json.split("```")[1].split("```")[0].strip()
                
            updated_plan = json.loads(plan_json)
            
            # Ensure ID and metadata are preserved
            updated_plan["id"] = existing_plan["id"]
            updated_plan["type"] = existing_plan["type"]
            updated_plan["task"] = existing_plan["task"]
            if "strategic_plan_id" in existing_plan:
                updated_plan["strategic_plan_id"] = existing_plan["strategic_plan_id"]
            updated_plan["created_at"] = existing_plan["created_at"]
            updated_plan["updated_at"] = time.time()
            updated_plan["status"] = "updated"
            
            # Update history
            self.plan_history = [updated_plan if p["id"] == plan_id else p for p in self.plan_history]
            
            return updated_plan
        except Exception as e:
            # Handle parsing errors
            error_update = {
                "id": existing_plan["id"],
                "type": existing_plan["type"],
                "task": existing_plan["task"],
                "created_at": existing_plan["created_at"],
                "updated_at": time.time(),
                "status": "error",
                "error": str(e),
                "raw_response": response.content
            }
            
            # Update history
            self.plan_history = [error_update if p["id"] == plan_id else p for p in self.plan_history]
            
            return error_update
    
    async def evaluate_plan(self, plan_id: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a tactical plan based on execution results.
        
        Args:
            plan_id: The ID of the plan to evaluate
            results: The results of executing the plan
            
        Returns:
            Evaluation results
        """
        # Get the existing plan
        existing_plan = self.get_plan(plan_id)
        if not existing_plan:
            return {
                "status": "error",
                "error": f"Plan with ID {plan_id} not found"
            }
        
        # Create evaluation prompt
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
Evaluate the following tactical research plan based on execution results:

PLAN:
{json.dumps(existing_plan, indent=2)}

EXECUTION RESULTS:
{json.dumps(results, indent=2)}

Assess:
1. Completion status of each task
2. Quality and relevance of information gathered
3. Whether verification steps were successful
4. If completion criteria were met
5. Overall effectiveness of the tactical approach
6. Areas for improvement in future tactical planning

Format your response as a JSON object with the following structure:
{{
    "task_assessments": [
        {{
            "task_id": "task_1",
            "completion_status": "completed/partial/failed",
            "completion_percentage": 0-100,
            "information_quality": "high/medium/low",
            "assessment": "Detailed assessment of task execution"
        }},
        ...
    ],
    "verification_assessment": {{
        "steps_completed": ["step 1", "step 2", ...],
        "steps_skipped": ["step 3", ...],
        "verification_success": true/false,
        "assessment": "Assessment of verification process"
    }},
    "completion_criteria_assessment": {{
        "criteria_met": ["criterion 1", "criterion 2", ...],
        "criteria_unmet": ["criterion 3", ...],
        "overall_completion": 0-100,
        "assessment": "Assessment of completion criteria satisfaction"
    }},
    "tactical_effectiveness": {{
        "task_sequencing": 0-10,
        "search_strategy": 0-10,
        "information_extraction": 0-10,
        "time_efficiency": 0-10,
        "assessment": "Overall assessment of tactical effectiveness"
    }},
    "improvement_suggestions": [
        {{
            "aspect": "Aspect to improve",
            "suggestion": "Specific improvement suggestion",
            "expected_benefit": "Expected benefit of this improvement"
        }},
        ...
    ],
    "overall_assessment": {{
        "success_rating": 0-10,
        "summary": "Summary assessment of the plan's execution"
    }}
}}
""")
        ]
        
        # Get response from LLM
        response = await self.llm.ainvoke(messages)
        
        try:
            # Extract JSON from the response content
            evaluation_json = response.content
            if "```json" in evaluation_json:
                evaluation_json = evaluation_json.split("```json")[1].split("```")[0].strip()
            elif "```" in evaluation_json:
                evaluation_json = evaluation_json.split("```")[1].split("```")[0].strip()
                
            evaluation = json.loads(evaluation_json)
            
            # Add metadata
            evaluation["plan_id"] = plan_id
            evaluation["evaluated_at"] = time.time()
            evaluation["type"] = "tactical_evaluation"
            
            # Update plan status
            for plan in self.plan_history:
                if plan["id"] == plan_id:
                    plan["status"] = "evaluated"
                    plan["evaluation"] = evaluation
            
            return evaluation
        except Exception as e:
            # Handle parsing errors
            error_evaluation = {
                "plan_id": plan_id,
                "evaluated_at": time.time(),
                "type": "tactical_evaluation",
                "status": "error",
                "error": str(e),
                "raw_response": response.content
            }
            
            return error_evaluation
    
    async def adapt_to_feedback(
        self, 
        plan_id: str, 
        feedback: Dict[str, Any], 
        execution_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adapt a tactical plan based on execution feedback.
        
        Args:
            plan_id: The ID of the plan to adapt
            feedback: Feedback on execution so far
            execution_state: Current state of execution
            
        Returns:
            The adapted plan
        """
        # Get the existing plan
        existing_plan = self.get_plan(plan_id)
        if not existing_plan:
            return {
                "status": "error",
                "error": f"Plan with ID {plan_id} not found"
            }
        
        # Create adaptation prompt
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
Adapt the following tactical research plan based on execution feedback:

CURRENT PLAN:
{json.dumps(existing_plan, indent=2)}

CURRENT EXECUTION STATE:
{json.dumps(execution_state, indent=2)}

FEEDBACK:
{json.dumps(feedback, indent=2)}

You should:
1. Modify tasks that need improvement
2. Add new tasks to address gaps
3. Remove or deprioritize tasks that are unproductive
4. Adjust search strategies based on what's working
5. Refine information extraction methods
6. Update task dependencies and sequence

Format your response as a complete JSON object with the same structure as the original plan.
""")
        ]
        
        # Get response from LLM
        response = await self.llm.ainvoke(messages)
        
        try:
            # Extract JSON from the response content
            plan_json = response.content
            if "```json" in plan_json:
                plan_json = plan_json.split("```json")[1].split("```")[0].strip()
            elif "```" in plan_json:
                plan_json = plan_json.split("```")[1].split("```")[0].strip()
                
            adapted_plan = json.loads(plan_json)
            
            # Ensure ID and metadata are preserved
            adapted_plan["id"] = existing_plan["id"]
            adapted_plan["type"] = existing_plan["type"]
            adapted_plan["task"] = existing_plan["task"]
            if "strategic_plan_id" in existing_plan:
                adapted_plan["strategic_plan_id"] = existing_plan["strategic_plan_id"]
            adapted_plan["created_at"] = existing_plan["created_at"]
            adapted_plan["adapted_at"] = time.time()
            adapted_plan["status"] = "adapted"
            adapted_plan["adaptation_feedback"] = feedback
            
            # Update history
            self.plan_history = [adapted_plan if p["id"] == plan_id else p for p in self.plan_history]
            
            return adapted_plan
        except Exception as e:
            # Handle parsing errors
            error_adaptation = {
                "id": existing_plan["id"],
                "type": existing_plan["type"],
                "task": existing_plan["task"],
                "created_at": existing_plan["created_at"],
                "adapted_at": time.time(),
                "status": "error",
                "error": str(e),
                "raw_response": response.content
            }
            
            # Update history
            self.plan_history = [error_adaptation if p["id"] == plan_id else p for p in self.plan_history]
            
            return error_adaptation
    
    def _format_strategic_plan(self, strategic_plan: Dict[str, Any]) -> str:
        """
        Format strategic plan for inclusion in tactical planning prompts.
        
        Args:
            strategic_plan: Strategic plan dictionary
            
        Returns:
            Formatted strategic plan string
        """
        if not strategic_plan:
            return ""
        
        strategic_plan_str = "STRATEGIC PLAN:\n"
        
        # Add objectives
        if "objectives" in strategic_plan:
            strategic_plan_str += "Objectives:\n"
            for obj in strategic_plan["objectives"]:
                obj_id = obj.get("id", "")
                description = obj.get("description", "")
                strategic_plan_str += f"- {obj_id}: {description}\n"
            strategic_plan_str += "\n"
        
        # Add key questions
        if "key_questions" in strategic_plan:
            strategic_plan_str += "Key Questions:\n"
            for q in strategic_plan["key_questions"]:
                q_id = q.get("id", "")
                question = q.get("question", "")
                strategic_plan_str += f"- {q_id}: {question}\n"
            strategic_plan_str += "\n"
        
        # Add success criteria
        if "success_criteria" in strategic_plan:
            strategic_plan_str += "Success Criteria:\n"
            for criterion in strategic_plan["success_criteria"]:
                description = criterion.get("criterion", "")
                measurement = criterion.get("measurement", "")
                strategic_plan_str += f"- {description} (Measured by: {measurement})\n"
            strategic_plan_str += "\n"
        
        return strategic_plan_str
    
    def _format_additional_context(self, context: Dict[str, Any]) -> str:
        """
        Format additional context for inclusion in planning prompts.
        
        Args:
            context: Context dictionary
            
        Returns:
            Formatted context string
        """
        if not context:
            return ""
        
        # Skip strategic_plan as it's handled separately
        context_without_strategic = {k: v for k, v in context.items() if k != "strategic_plan"}
        
        if not context_without_strategic:
            return ""
        
        context_str = "ADDITIONAL CONTEXT:\n"
        
        for key, value in context_without_strategic.items():
            if isinstance(value, (dict, list)):
                context_str += f"{key.upper()}:\n{json.dumps(value, indent=2)}\n\n"
            else:
                context_str += f"{key.upper()}: {value}\n\n"
        
        return context_str