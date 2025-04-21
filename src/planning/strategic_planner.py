"""
Strategic Planner Module

This module provides the StrategicPlanner class which creates high-level research
plans and strategies for the autonomous research system.
"""

from typing import Dict, List, Any, Optional
import json
import time
import uuid

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from .base_planner import BasePlanner


class StrategicPlanner(BasePlanner):
    """
    Strategic Planner creates high-level research plans and strategies.
    
    This planner focuses on defining overall research objectives, key questions,
    and the general approach for complex research tasks.
    """
    
    STRATEGIC_PLANNER_PROMPT = """You are a Strategic Planner in a multi-agent research system.
Your role is to create high-level research plans and strategies. You should:
1. Analyze research requests to identify core objectives
2. Break down complex topics into key research questions
3. Define clear success criteria for research completion
4. Design overall research strategies that are thorough and effective
5. Structure research to ensure comprehensive coverage of the topic
6. Balance breadth and depth of investigation
7. Anticipate potential challenges and plan contingencies
8. Provide clear guidance for tactical planning

Your plans should be well-structured, comprehensive, and actionable. They should emphasize 
thoroughness and depth in research while maintaining a coherent overall strategy.
"""
    
    def __init__(
        self,
        llm: BaseChatModel,
        name: str = "StrategicPlanner",
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize a StrategicPlanner instance.
        
        Args:
            llm: The language model to use
            name: The name/identifier for this planner (default: "StrategicPlanner")
            system_prompt: Optional custom system prompt (uses default if None)
        """
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt or self.STRATEGIC_PLANNER_PROMPT,
        )
    
    async def create_plan(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a strategic research plan for a given task.
        
        Args:
            task: The research task to plan for
            context: Additional context for planning
            
        Returns:
            A structured strategic plan
        """
        context = context or {}
        
        # Create plan request prompt
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
Create a comprehensive strategic research plan for the following task:
"{task}"

{self._format_context(context)}

Your plan should include:
1. Overall research objectives
2. Key research questions to answer
3. Information types and sources to investigate
4. Success criteria for determining when research is complete
5. A high-level research strategy
6. Risk factors and contingency approaches

Format your response as a JSON object with the following structure:
{{
    "objectives": [
        {{
            "id": "obj_1",
            "description": "First research objective",
            "rationale": "Why this objective matters"
        }},
        ...
    ],
    "key_questions": [
        {{
            "id": "q_1",
            "question": "First key question",
            "related_objectives": ["obj_1", ...],
            "priority": 1-10
        }},
        ...
    ],
    "information_sources": [
        {{
            "type": "source type (e.g., academic, news, primary data)",
            "relevance": "Why this source type is relevant",
            "reliability": "Assessment of reliability"
        }},
        ...
    ],
    "success_criteria": [
        {{
            "criterion": "Description of success criterion",
            "measurement": "How to measure if this criterion is met"
        }},
        ...
    ],
    "research_strategy": [
        {{
            "phase": "Phase name/number",
            "focus": "What this phase focuses on",
            "approaches": ["approach 1", "approach 2", ...],
            "deliverables": ["deliverable 1", "deliverable 2", ...]
        }},
        ...
    ],
    "risk_factors": [
        {{
            "risk": "Description of potential risk",
            "impact": "high/medium/low",
            "mitigation": "How to mitigate this risk"
        }},
        ...
    ]
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
                
            strategic_plan = json.loads(plan_json)
            
            # Add metadata
            plan_id = str(uuid.uuid4())
            strategic_plan["id"] = plan_id
            strategic_plan["type"] = "strategic"
            strategic_plan["task"] = task
            strategic_plan["created_at"] = time.time()
            strategic_plan["status"] = "created"
            
            # Add to history
            self.add_to_history(strategic_plan)
            
            return strategic_plan
        except Exception as e:
            # Handle parsing errors
            error_plan = {
                "id": str(uuid.uuid4()),
                "type": "strategic",
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
        Update an existing strategic plan.
        
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
Update the following strategic research plan:

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
        Evaluate a strategic plan based on execution results.
        
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
Evaluate the following strategic research plan based on execution results:

PLAN:
{json.dumps(existing_plan, indent=2)}

EXECUTION RESULTS:
{json.dumps(results, indent=2)}

Assess:
1. How well each objective was met
2. Whether key questions were answered
3. If success criteria were satisfied
4. Overall effectiveness of the research strategy
5. How well risks were mitigated
6. Areas for improvement in future planning

Format your response as a JSON object with the following structure:
{{
    "objectives_assessment": [
        {{
            "objective_id": "obj_1",
            "completion_percentage": 0-100,
            "assessment": "Detailed assessment of objective achievement"
        }},
        ...
    ],
    "questions_assessment": [
        {{
            "question_id": "q_1",
            "answered": true/false,
            "quality": "high/medium/low",
            "assessment": "Detailed assessment of answer quality"
        }},
        ...
    ],
    "success_criteria_assessment": [
        {{
            "criterion": "Description of criterion",
            "met": true/false,
            "assessment": "Detailed assessment of criterion satisfaction"
        }},
        ...
    ],
    "strategy_effectiveness": {{
        "overall_rating": 0-10,
        "strengths": ["strength 1", "strength 2", ...],
        "weaknesses": ["weakness 1", "weakness 2", ...]
    }},
    "risk_management": {{
        "anticipated_risks_handled": 0-100,
        "unanticipated_risks": ["risk 1", "risk 2", ...],
        "assessment": "Overall assessment of risk management"
    }},
    "improvement_suggestions": [
        {{
            "area": "Area for improvement",
            "suggestion": "Specific improvement suggestion",
            "expected_benefit": "Expected benefit of this improvement"
        }},
        ...
    ],
    "overall_assessment": {{
        "success_rating": 0-10,
        "summary": "Summary assessment of the plan's success"
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
            evaluation["type"] = "strategic_evaluation"
            
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
                "type": "strategic_evaluation",
                "status": "error",
                "error": str(e),
                "raw_response": response.content
            }
            
            return error_evaluation
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """
        Format context information for inclusion in prompts.
        
        Args:
            context: Context dictionary
            
        Returns:
            Formatted context string
        """
        if not context:
            return ""
        
        context_str = "ADDITIONAL CONTEXT:\n"
        
        for key, value in context.items():
            if isinstance(value, (dict, list)):
                context_str += f"{key.upper()}:\n{json.dumps(value, indent=2)}\n\n"
            else:
                context_str += f"{key.upper()}: {value}\n\n"
        
        return context_str