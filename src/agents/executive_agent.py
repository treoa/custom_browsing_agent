"""
Executive Agent Module

This module provides the ExecutiveAgent class which is responsible for overall coordination,
goal setting, and research strategy development in the multi-agent system.
"""

from typing import Dict, List, Any, Optional
import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from .base_agent import BaseAgent


class ExecutiveAgent(BaseAgent):
    """
    The Executive Agent is responsible for coordinating the overall research process.
    
    This agent interprets user research requests, formulates research objectives,
    develops strategies, allocates tasks to specialized agents, and monitors progress.
    """
    
    EXECUTIVE_SYSTEM_PROMPT = """You are the Executive Agent in a multi-agent research system.
Your role is to coordinate the overall research process by:
1. Interpreting user research requests into clear research objectives
2. Developing comprehensive research strategies
3. Breaking down complex research tasks into manageable components
4. Allocating tasks to specialized agents
5. Monitoring overall progress
6. Ensuring research quality and completeness
7. Communicating findings to the user

You must establish explicit, measurable research completion criteria for each task.
You should maintain a global context of the research project and implement continuous
improvement through metacognitive feedback loops.
"""
    
    def __init__(
        self,
        llm: BaseChatModel,
        name: str = "Executive",
        system_prompt: Optional[str] = None,
        memory: Optional[Any] = None,
    ):
        """
        Initialize an ExecutiveAgent instance.
        
        Args:
            llm: The language model to use
            name: The name/identifier for this agent (default: "Executive")
            system_prompt: Optional custom system prompt (uses default if None)
            memory: Optional memory system
        """
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt or self.EXECUTIVE_SYSTEM_PROMPT,
            memory=memory,
        )
        self.research_plan = None
        self.task_queue = []
        self.agent_assignments = {}
    
    async def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute an executive function and return the result.
        
        Args:
            task: The task to execute (e.g., "create_research_plan", "assign_tasks")
            context: Additional context for task execution
            
        Returns:
            A dictionary containing the execution results
        """
        context = context or {}
        
        if task == "create_research_plan":
            return await self._create_research_plan(context.get("user_request", ""))
        elif task == "assign_tasks":
            return await self._assign_tasks(context.get("research_plan", {}), 
                                           context.get("available_agents", []))
        elif task == "evaluate_progress":
            return await self._evaluate_progress(context.get("task_results", {}))
        elif task == "synthesize_findings":
            return await self._synthesize_findings(context.get("research_results", {}))
        else:
            return {"error": f"Unknown task: {task}"}
    
    async def _create_research_plan(self, user_request: str) -> Dict[str, Any]:
        """
        Create a comprehensive research plan based on the user's request.
        
        Args:
            user_request: The user's research request
            
        Returns:
            A structured research plan
        """
        if not user_request:
            return {"error": "No user request provided"}
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
Create a comprehensive research plan for the following user request:
\"{user_request}\"

Your plan should include:
1. Clear research objectives
2. Key research questions to answer
3. A step-by-step research strategy
4. Required information sources
5. Explicit completion criteria for determining when the research is adequate

Format your response as a JSON object with the following structure:
{{
    "objectives": ["objective1", "objective2", ...],
    "key_questions": ["question1", "question2", ...],
    "strategy": [
        {{"step": 1, "description": "...", "rationale": "..."}},
        {{"step": 2, "description": "...", "rationale": "..."}}
    ],
    "information_sources": ["source1", "source2", ...],
    "completion_criteria": [
        {{"criterion": "...", "measurement": "..."}}
    ]
}}
""")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        try:
            # Extract JSON from the response content
            plan_json = response.content
            if "```json" in plan_json:
                plan_json = plan_json.split("```json")[1].split("```")[0].strip()
            elif "```" in plan_json:
                plan_json = plan_json.split("```")[1].split("```")[0].strip()
                
            research_plan = json.loads(plan_json)
            self.research_plan = research_plan
            
            result = {
                "status": "success",
                "research_plan": research_plan
            }
        except Exception as e:
            result = {
                "status": "error",
                "error": f"Failed to parse research plan: {str(e)}",
                "raw_response": response.content
            }
        
        self.add_to_history("create_research_plan", result)
        return result
    
    async def _assign_tasks(self, research_plan: Dict[str, Any], 
                           available_agents: List[str]) -> Dict[str, Any]:
        """
        Assign research tasks to available agents based on the research plan.
        
        Args:
            research_plan: The structured research plan
            available_agents: List of available agent identifiers
            
        Returns:
            Task assignments for each agent
        """
        if not research_plan:
            return {"error": "No research plan provided"}
        if not available_agents:
            return {"error": "No available agents provided"}
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
Assign tasks to the available agents based on the research plan.

Research Plan:
{json.dumps(research_plan, indent=2)}

Available Agents:
{json.dumps(available_agents, indent=2)}

Format your response as a JSON object with agent assignments:
{{
    "agent_assignments": [
        {{
            "agent": "agent_name",
            "tasks": [
                {{
                    "task_id": "unique_id",
                    "description": "detailed task description",
                    "priority": 1-5,
                    "dependencies": ["task_id1", "task_id2"]
                }}
            ]
        }}
    ]
}}
""")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        try:
            # Extract JSON from the response content
            assignments_json = response.content
            if "```json" in assignments_json:
                assignments_json = assignments_json.split("```json")[1].split("```")[0].strip()
            elif "```" in assignments_json:
                assignments_json = assignments_json.split("```")[1].split("```")[0].strip()
                
            agent_assignments = json.loads(assignments_json)
            self.agent_assignments = agent_assignments
            
            # Process assignments into task queue
            self.task_queue = []
            for assignment in agent_assignments.get("agent_assignments", []):
                agent = assignment.get("agent")
                for task in assignment.get("tasks", []):
                    task["assigned_agent"] = agent
                    self.task_queue.append(task)
            
            # Sort task queue by priority and dependencies
            self.task_queue.sort(key=lambda x: x.get("priority", 5))
            
            result = {
                "status": "success",
                "agent_assignments": agent_assignments,
                "task_queue": self.task_queue
            }
        except Exception as e:
            result = {
                "status": "error",
                "error": f"Failed to parse agent assignments: {str(e)}",
                "raw_response": response.content
            }
        
        self.add_to_history("assign_tasks", result)
        return result
    
    async def _evaluate_progress(self, task_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate research progress based on task results.
        
        Args:
            task_results: Results from completed tasks
            
        Returns:
            Progress evaluation and next steps
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
Evaluate the research progress based on completed tasks and the original research plan.

Research Plan:
{json.dumps(self.research_plan, indent=2)}

Task Results:
{json.dumps(task_results, indent=2)}

Provide an evaluation of the current progress and recommendations for next steps.
Format your response as a JSON object:
{{
    "progress_percentage": 0-100,
    "completed_objectives": ["objective1", "objective2", ...],
    "pending_objectives": ["objective3", "objective4", ...],
    "gaps_identified": ["gap1", "gap2", ...],
    "recommendations": ["recommendation1", "recommendation2", ...],
    "is_research_complete": true/false
}}
""")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        try:
            # Extract JSON from the response content
            evaluation_json = response.content
            if "```json" in evaluation_json:
                evaluation_json = evaluation_json.split("```json")[1].split("```")[0].strip()
            elif "```" in evaluation_json:
                evaluation_json = evaluation_json.split("```")[1].split("```")[0].strip()
                
            progress_evaluation = json.loads(evaluation_json)
            
            result = {
                "status": "success",
                "progress_evaluation": progress_evaluation
            }
        except Exception as e:
            result = {
                "status": "error",
                "error": f"Failed to parse progress evaluation: {str(e)}",
                "raw_response": response.content
            }
        
        self.add_to_history("evaluate_progress", result)
        return result
    
    async def _synthesize_findings(self, research_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize research findings into a coherent output.
        
        Args:
            research_results: Collected research results
            
        Returns:
            Synthesized research findings
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
Synthesize the research findings into a comprehensive, coherent output.

Research Plan:
{json.dumps(self.research_plan, indent=2)}

Research Results:
{json.dumps(research_results, indent=2)}

Provide a well-structured synthesis of the findings that addresses all research objectives
and key questions. Ensure the synthesis is comprehensive, logically organized, and highlights
key insights and implications.
""")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        result = {
            "status": "success",
            "synthesis": response.content
        }
        
        self.add_to_history("synthesize_findings", result)
        return result