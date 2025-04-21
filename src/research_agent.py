"""
Research Agent Module

This module provides the main ResearchAgent class which coordinates the entire
multi-agent research system for conducting deep, methodical investigations.
"""

from typing import Dict, List, Any, Optional
import asyncio
import time
import uuid
import os
import json

from langchain_core.language_models import BaseChatModel
from browser_use import Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig

from .agents import ExecutiveAgent, ResearchAgent as WebResearchAgent, AnalysisAgent, CritiqueAgent, SynthesisAgent
from .memory import MemorySystem
from .planning import PlanningSystem
from .evaluation import CompletionCriteria, CompletionEvaluator
from .evaluation.completion_criteria import CriteriaLevel


class ResearchAgent:
    """
    Advanced Autonomous Research Agent for conducting deep, methodical investigations.
    
    This agent coordinates a multi-agent system with specialized components for planning,
    execution, evaluation, and synthesis to ensure thorough, high-quality research.
    """
    
    def __init__(
        self,
        basic_llm: BaseChatModel,
        advanced_llm: BaseChatModel,
        browser: Optional[Browser] = None,
        browser_context: Optional[Any] = None,
        max_steps: int = 50,
        storage_path: Optional[str] = None,
        logger: Optional[Any] = None,
        progress_callback: Optional[callable] = None,
        show_plan_approval: bool = True,
        completion_criteria: Optional[CompletionCriteria] = None,
        task_type: str = "general",
        task_complexity: str = "medium",
    ):
        """
        Initialize a ResearchAgent instance.
        
        Args:
            basic_llm: The language model to use for basic tasks
            advanced_llm: The language model to use for advanced tasks
            browser: Optional browser instance (creates a new one if None)
            browser_context: Optional browser context for reusing an existing session
            max_steps: Maximum number of research steps per session
            storage_path: Path for storing persistent data (memory, episodes, etc.)
            logger: Optional logger instance
            progress_callback: Optional callback function for progress updates
            show_plan_approval: Whether to show plan for approval
        """
        self.basic_llm = basic_llm
        self.advanced_llm = advanced_llm
        self.browser = browser or self._create_default_browser()
        self.browser_context = browser_context  # Store the browser context
        self.max_steps = max_steps
        self.storage_path = storage_path
        self.logger = logger
        self.progress_callback = progress_callback
        self.show_plan_approval = show_plan_approval
        
        # Initialize completion criteria and evaluator
        self.completion_criteria = completion_criteria or CompletionCriteria.create_task_specific(
            task_type=task_type,
            complexity=task_complexity
        )
        self.completion_evaluator = CompletionEvaluator(
            llm=advanced_llm,
            criteria=self.completion_criteria,
            task_type=task_type,
            task_complexity=task_complexity
        )
        
        # Create storage paths if specified
        if storage_path:
            os.makedirs(storage_path, exist_ok=True)
            memory_path = os.path.join(storage_path, "memory")
            os.makedirs(memory_path, exist_ok=True)
        else:
            memory_path = None
        
        # Initialize memory system
        self.memory_system = MemorySystem(
            llm=basic_llm,
            ltm_storage_path=os.path.join(memory_path, "ltm.json") if memory_path else None,
            episodic_storage_path=os.path.join(memory_path, "episodic.json") if memory_path else None,
        )
        
        # Initialize planning system
        self.planning_system = PlanningSystem(llm=advanced_llm)
        
        # Initialize specialized agents
        self.executive_agent = ExecutiveAgent(llm=advanced_llm, memory=self.memory_system)
        self.web_research_agent = WebResearchAgent(llm=basic_llm, browser=self.browser, browser_context=self.browser_context, memory=self.memory_system)
        self.analysis_agent = AnalysisAgent(llm=advanced_llm, memory=self.memory_system)
        self.critique_agent = CritiqueAgent(llm=advanced_llm, memory=self.memory_system) # here it can be changed to basic_llm if limites would be too low 
        self.synthesis_agent = SynthesisAgent(llm=basic_llm, memory=self.memory_system)
        
        # Research state
        self.current_research_id = None
        self.current_research_plan = None
        self.current_execution_state = {}
    
    async def _update_status_file(self, status: str, current_task: str, progress_percentage: int) -> None:
        """
        Update the status file with current progress information.
        
        Args:
            status: Current status of the research (e.g., "Planning", "Executing")
            current_task: Description of the current task
            progress_percentage: Overall progress percentage (0-100)
        """
        if not self.storage_path:
            return  # No storage path available
            
        status_file_path = os.path.join(self.storage_path, "status.md")
        
        try:
            with open(status_file_path, "w") as f:
                f.write(f"# Research Status\n\n")
                f.write(f"## Current Status: {status}\n\n")
                f.write(f"## Current Task\n{current_task}\n\n")
                f.write(f"## Progress\n{progress_percentage}%\n\n")
                f.write(f"## Last Updated\n{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
            # Log research progress
            log_file_path = os.path.join(self.storage_path, "logs", "research.log")
            if os.path.exists(os.path.dirname(log_file_path)):
                with open(log_file_path, "a") as f:
                    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {status}: {current_task} ({progress_percentage}%)\n")
                    
            # Call progress callback if available
            if self.progress_callback:
                try:
                    # Use integers for step and total to avoid the multiplication error
                    self.progress_callback(progress_percentage, 100, f"{status}: {current_task}")
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error in progress callback: {str(e)}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error updating status file: {str(e)}")
    
    async def research(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        show_plan_approval: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Conduct thorough research on a query.
        
        Args:
            query: The research query
            context: Optional additional context
            
        Returns:
            Research results and findings
        """
        # Initialize research
        research_id = str(uuid.uuid4())
        self.current_research_id = research_id
        
        # Create episode in episodic memory
        await self.memory_system.episodic.create_episode(
            label=f"Research: {query[:50]}",
            metadata={
                "research_id": research_id,
                "query": query,
                "context": context
            }
        )
        
        # Record start event
        await self.memory_system.episodic.add(
            key="research_start",
            value={
                "query": query,
                "context": context
            }
        )
        
        # Update status
        await self._update_status_file("Planning", "Creating research strategy", 5)
        
        # Create research plan
        self.current_research_plan = await self.planning_system.create_research_plan(
            research_request=query,
            context=context
        )
        
        if self.logger:
            self.logger.info("Research plan created")
        
        # Record planning event
        await self.memory_system.episodic.add(
            key="research_planning",
            value={
                "plan": self.current_research_plan
            }
        )
        
        # Show plan for approval if requested
        if show_plan_approval and self.current_research_plan:
            # Save plan to file for user review
            if self.storage_path:
                plan_path = os.path.join(self.storage_path, "research_plan.md")
                with open(plan_path, "w") as f:
                    f.write("# Research Plan\n\n")
                    f.write(f"Query: {query}\n\n")
                    
                    # Extract strategic plan
                    if "strategic_plan" in self.current_research_plan:
                        strategic_plan = self.current_research_plan["strategic_plan"]
                        
                        # Write objectives
                        if "objectives" in strategic_plan:
                            f.write("## Objectives\n\n")
                            for obj in strategic_plan["objectives"]:
                                f.write(f"- {obj.get('description', obj)}\n")
                            f.write("\n")
                        
                        # Write research strategy
                        if "research_strategy" in strategic_plan:
                            f.write("## Research Strategy\n\n")
                            for phase in strategic_plan["research_strategy"]:
                                phase_name = phase.get("phase", "")
                                phase_focus = phase.get("focus", "")
                                phase_desc = phase.get("description", "")
                                
                                f.write(f"### Phase: {phase_name}\n\n")
                                if phase_focus:
                                    f.write(f"Focus: {phase_focus}\n\n")
                                if phase_desc:
                                    f.write(f"{phase_desc}\n\n")
                    
                    # Write tactical plans
                    if "tactical_plans" in self.current_research_plan and self.current_research_plan["tactical_plans"]:
                        f.write("## Tactical Plans\n\n")
                        for i, plan in enumerate(self.current_research_plan["tactical_plans"]):
                            plan_name = plan.get("name", f"Plan {i+1}")
                            f.write(f"### {plan_name}\n\n")
                            
                            # Write tasks
                            if "tasks" in plan:
                                f.write("#### Tasks\n\n")
                                for task in plan["tasks"]:
                                    task_id = task.get("id", "")
                                    task_desc = task.get("description", "")
                                    f.write(f"- [{task_id}] {task_desc}\n")
                                f.write("\n")
                
                # Update todo.md file with plan tasks
                todo_path = os.path.join(self.storage_path, "todo.md")
                if os.path.exists(todo_path):
                    with open(todo_path, "w") as f:
                        f.write(f"# Research Plan for: {query}\n\n")
                        f.write("## Tasks\n\n")
                        
                        # Add all tasks from tactical plans
                        added_tasks = set()
                        for plan in self.current_research_plan.get("tactical_plans", []):
                            for task in plan.get("tasks", []):
                                task_id = task.get("id", "")
                                task_desc = task.get("description", "")
                                
                                if task_id and task_id not in added_tasks:
                                    f.write(f"- [ ] {task_id}: {task_desc}\n")
                                    added_tasks.add(task_id)
        
        # Update status
        await self._update_status_file("Executing", "Initializing execution", 10)
        
        # Initialize execution state
        self.current_execution_state = {
            "research_id": research_id,
            "query": query,
            "start_time": time.time(),
            "completed_tasks": {},
            "pending_tasks": [],
            "collected_information": {},
            "current_step": 0,
            "status": "in_progress"
        }
        
        # Get initial execution tasks
        execution_tasks = await self.planning_system.get_execution_tasks(
            execution_state=self.current_execution_state
        )
        
        self.current_execution_state["pending_tasks"] = [task["id"] for task in execution_tasks]
        
        # Execute research plan
        results = await self._execute_research_plan(self.current_research_plan, execution_tasks)
        
        # Analyze research results
        analysis_results = await self._analyze_research_results(results)
        
        # Critique research findings
        critique_results = await self._critique_research_findings(results, analysis_results)
        
        # Synthesize final output
        final_output = await self._synthesize_research_output(results, analysis_results, critique_results)
        
        # Update execution state
        self.current_execution_state["status"] = "completed"
        self.current_execution_state["end_time"] = time.time()
        self.current_execution_state["duration"] = self.current_execution_state["end_time"] - self.current_execution_state["start_time"]
        
        # Update status file with completion info
        await self._update_status_file("Completed", "Research completed successfully", 100)
        
        # Record completion event
        await self.memory_system.episodic.add(
            key="research_completion",
            value={
                "execution_state": self.current_execution_state,
                "final_output": final_output
            }
        )
        
        # End episode with outcome
        await self.memory_system.episodic.end_episode(
            outcome={
                "success": True,
                "output": final_output,
                "duration": self.current_execution_state["duration"]
            }
        )
        
        # Create comprehensive results object
        comprehensive_results = {
            "research_id": research_id,
            "query": query,
            "execution_state": self.current_execution_state,
            "results": results,
            "analysis": analysis_results,
            "critique": critique_results,
            "output": final_output
        }
        
        # Store in long-term memory
        await self.memory_system.ltm.add(
            key=f"research_{research_id}",
            value=comprehensive_results,
            metadata={
                "type": "research_results",
                "query": query,
                "timestamp": time.time()
            }
        )
        
        return comprehensive_results
    
    async def _execute_research_plan(
        self,
        research_plan: Dict[str, Any],
        execution_tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute a research plan by performing tasks through specialized agents.
        
        Args:
            research_plan: The research plan to execute
            execution_tasks: List of execution tasks from the plan
            
        Returns:
            Research results
        """
        results = {
            "collected_information": {},
            "completed_tasks": {},
            "task_results": {}
        }
        
        # Track research progress
        current_step = 0
        max_steps = self.max_steps
        
        # Execute tasks in order of priority and dependencies
        while current_step < max_steps and execution_tasks:
            current_step += 1
            self.current_execution_state["current_step"] = current_step
            
            # Get next task
            task = execution_tasks[0]
            task_id = task["id"]
            task_description = task.get("description", "")
            
            # Update progress percentage - adjust formula to provide better progression
            progress_percentage = int(min(10 + (current_step / max_steps * 80), 95))  # Cap at 95% until complete
            await self._update_status_file("Executing", f"Step {current_step}/{max_steps}: {task_description}", progress_percentage)
            
            # Log task execution
            if self.logger:
                self.logger.info(f"Step {current_step}/{max_steps}: Executing task {task_id} - {task_description}")
                
            print(f"Step {current_step}/{max_steps}: Executing task {task_id} - {task_description}")
            
            # Record task start
            await self.memory_system.episodic.add(
                key="task_start",
                value={
                    "task_id": task_id,
                    "task": task,
                    "step": current_step
                }
            )
            
            # Execute task based on type
            task_type = task.get("type", "web_research")
            
            if task_type == "web_research" or "search_queries" in task:
                # Use web research agent
                task_result = await self._execute_web_research_task(task)
            elif task_type == "analysis":
                # Use analysis agent
                task_result = await self._execute_analysis_task(task, results)
            elif task_type == "critique":
                # Use critique agent
                task_result = await self._execute_critique_task(task, results)
            else:
                # Default to executive agent for coordination
                task_result = await self._execute_executive_task(task, results)
            
            # Process task result
            if task_result.get("status") == "completed":
                # Mark task as completed
                self.current_execution_state["completed_tasks"][task_id] = True
                self.current_execution_state["pending_tasks"].remove(task_id)
                
                # Store task result
                results["task_results"][task_id] = task_result
                
                # Store collected information
                if "information" in task_result:
                    for key, value in task_result["information"].items():
                        results["collected_information"][key] = value
                
                # Record task completion
                await self.memory_system.episodic.add(
                    key="task_completion",
                    value={
                        "task_id": task_id,
                        "result": task_result,
                        "step": current_step
                    }
                )
            else:
                # Handle task failure
                results["task_results"][task_id] = task_result
                
                # Record task failure
                await self.memory_system.episodic.add(
                    key="task_failure",
                    value={
                        "task_id": task_id,
                        "result": task_result,
                        "step": current_step
                    }
                )
                
                # Check if we need to retry or skip
                if task_result.get("should_retry", False) and current_step < max_steps - 1:
                    # Leave task in execution queue for retry
                    print(f"Will retry task {task_id} in next step")
                    
                    # Record retry decision
                    await self.memory_system.episodic.add(
                        key="task_retry_decision",
                        value={
                            "task_id": task_id,
                            "step": current_step
                        }
                    )
                else:
                    # Skip task
                    self.current_execution_state["pending_tasks"].remove(task_id)
                    
                    # Record skip decision
                    await self.memory_system.episodic.add(
                        key="task_skip_decision",
                        value={
                            "task_id": task_id,
                            "step": current_step
                        }
                    )
            
            # Remove completed task from execution list
            execution_tasks.pop(0)
            
            # Check progress and adapt plan if needed
            if current_step % 5 == 0 or len(execution_tasks) == 0:
                # Evaluate progress
                progress_evaluation = await self.planning_system.evaluate_research_progress(
                    execution_results=results
                )
                
                # Record progress evaluation
                await self.memory_system.episodic.add(
                    key="progress_evaluation",
                    value={
                        "evaluation": progress_evaluation,
                        "step": current_step
                    }
                )
                
                # Check if we need to adapt plan
                if progress_evaluation["overall_assessment"]["progress_percentage"] < 50 and current_step < max_steps / 2:
                    # Adapt tactical plans based on evaluation
                    for tactical_plan in self.current_research_plan.get("tactical_plans", []):
                        plan_id = tactical_plan.get("id")
                        if plan_id:
                            # Adapt plan
                            adapted_plan = await self.planning_system.adapt_tactical_plan(
                                plan_id=plan_id,
                                feedback=progress_evaluation,
                                execution_state=self.current_execution_state
                            )
                            
                            # Record plan adaptation
                            await self.memory_system.episodic.add(
                                key="plan_adaptation",
                                value={
                                    "plan_id": plan_id,
                                    "adapted_plan": adapted_plan,
                                    "step": current_step
                                }
                            )
                    
                    # Get updated execution tasks
                    updated_tasks = await self.planning_system.get_execution_tasks(
                        execution_state=self.current_execution_state
                    )
                    
                    # Add new tasks to execution queue
                    for task in updated_tasks:
                        task_id = task["id"]
                        if task_id not in self.current_execution_state["completed_tasks"] and task_id not in self.current_execution_state["pending_tasks"]:
                            execution_tasks.append(task)
                            self.current_execution_state["pending_tasks"].append(task_id)
                    
                    # Sort tasks by priority
                    execution_tasks.sort(key=lambda x: -(x.get("priority", 0)))
        
        # Check for research completion
        complete = len(self.current_execution_state["pending_tasks"]) == 0
        
        # If research wasn't completed due to step limit, record that
        if not complete:
            await self.memory_system.episodic.add(
                key="research_incomplete",
                value={
                    "reason": "max_steps_reached",
                    "completed_steps": current_step,
                    "max_steps": max_steps,
                    "pending_tasks_count": len(self.current_execution_state["pending_tasks"])
                }
            )
        
        return results
    
    async def _execute_web_research_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a web research task using the web research agent.
        
        Args:
            task: The task to execute
            
        Returns:
            Task execution result
        """
        # Extract search queries and other parameters
        search_queries = task.get("search_queries", [])
        objective_id = task.get("objective_id", "")
        task_id = task["id"]
        task_description = task.get("description", "")
        
        # Create context for research agent
        context = {
            "task_id": task_id,
            "objective_id": objective_id,
            "task_description": task_description,
            "search_queries": search_queries,
            "information_extraction": task.get("information_extraction", {}),
            "verification_steps": task.get("verification_steps", []),
            "completion_criteria": task.get("completion_criteria", [])
        }
        
        # Create search task incorporating all queries
        search_task = f"Research: {task_description}\n\nSearch for information using these queries:\n"
        for query in search_queries:
            search_task += f"- {query}\n"
        
        # Add extraction guidance if available
        if "information_extraction" in task and "data_points" in task["information_extraction"]:
            search_task += "\n\nExtract the following information:\n"
            for data_point in task["information_extraction"]["data_points"]:
                search_task += f"- {data_point}\n"
        
        # Execute research task
        try:
            research_result = await self.web_research_agent.execute(
                task=search_task,
                context=context
            )
            
            # Process research results
            if research_result.get("status") == "completed":
                # Extract information from research results
                extracted_info = {}
                
                # Check if there's a research_summary with key_findings
                if "research_summary" in research_result and "key_findings" in research_result["research_summary"]:
                    for i, finding in enumerate(research_result["research_summary"]["key_findings"]):
                        extracted_info[f"finding_{i+1}"] = finding
                
                # Check if there are subtopics in the research summary
                if "research_summary" in research_result and "subtopics" in research_result["research_summary"]:
                    for subtopic in research_result["research_summary"]["subtopics"]:
                        topic_name = subtopic.get("topic", f"topic_{len(extracted_info)+1}")
                        extracted_info[topic_name] = subtopic.get("content", "")
                
                # If no structured data found, use the raw research summary
                if not extracted_info and "research_summary" in research_result and "summary" in research_result["research_summary"]:
                    extracted_info["summary"] = research_result["research_summary"]["summary"]
                
                # Create result object
                result = {
                    "status": "completed",
                    "task_id": task_id,
                    "information": extracted_info,
                    "sources": research_result.get("sources_visited", []),
                    "research_summary": research_result.get("research_summary", {})
                }
                
                return result
            else:
                # Handle research failure
                return {
                    "status": "failed",
                    "task_id": task_id,
                    "error": "Research execution failed",
                    "research_result": research_result,
                    "should_retry": True
                }
        except Exception as e:
            # Handle exceptions
            return {
                "status": "failed",
                "task_id": task_id,
                "error": f"Exception during research: {str(e)}",
                "should_retry": True
            }
    
    async def _execute_analysis_task(self, task: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an analysis task using the analysis agent.
        
        Args:
            task: The task to execute
            results: Current research results
            
        Returns:
            Task execution result
        """
        # Extract task parameters
        task_id = task["id"]
        task_description = task.get("description", "")
        analysis_type = task.get("analysis_type", "general")
        
        # Create context for analysis agent
        context = {
            "task_id": task_id,
            "task_description": task_description,
            "analysis_type": analysis_type,
            "research_findings": results["collected_information"]
        }
        
        # Execute analysis task
        try:
            analysis_result = await self.analysis_agent.execute(
                task=task_description,
                context=context
            )
            
            # Process analysis results
            if analysis_result.get("status") == "success":
                # Create result object
                result = {
                    "status": "completed",
                    "task_id": task_id,
                    "information": {
                        f"analysis_{analysis_type}": analysis_result.get(f"{analysis_type}_analysis", 
                                                    analysis_result.get("analysis", {}))
                    },
                    "analysis_result": analysis_result
                }
                
                return result
            else:
                # Handle analysis failure
                return {
                    "status": "failed",
                    "task_id": task_id,
                    "error": "Analysis execution failed",
                    "analysis_result": analysis_result,
                    "should_retry": False
                }
        except Exception as e:
            # Handle exceptions
            return {
                "status": "failed",
                "task_id": task_id,
                "error": f"Exception during analysis: {str(e)}",
                "should_retry": False
            }
    
    async def _execute_critique_task(self, task: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a critique task using the critique agent.
        
        Args:
            task: The task to execute
            results: Current research results
            
        Returns:
            Task execution result
        """
        # Extract task parameters
        task_id = task["id"]
        task_description = task.get("description", "")
        critique_type = task.get("critique_type", "completeness")
        
        # Get research plan for context
        strategic_plan = self.current_research_plan.get("strategic_plan", {}) if self.current_research_plan else {}
        
        # Create context for critique agent
        context = {
            "task_id": task_id,
            "task_description": task_description,
            "critique_type": critique_type,
            "research_findings": results["collected_information"],
            "research_plan": strategic_plan
        }
        
        # Execute critique task
        try:
            critique_result = await self.critique_agent.execute(
                task=task_description,
                context=context
            )
            
            # Process critique results
            if critique_result.get("status") == "success":
                # Create result object
                result = {
                    "status": "completed",
                    "task_id": task_id,
                    "information": {
                        f"critique_{critique_type}": critique_result.get(f"{critique_type}_evaluation", 
                                                   critique_result.get("evaluation", {}))
                    },
                    "critique_result": critique_result
                }
                
                return result
            else:
                # Handle critique failure
                return {
                    "status": "failed",
                    "task_id": task_id,
                    "error": "Critique execution failed",
                    "critique_result": critique_result,
                    "should_retry": False
                }
        except Exception as e:
            # Handle exceptions
            return {
                "status": "failed",
                "task_id": task_id,
                "error": f"Exception during critique: {str(e)}",
                "should_retry": False
            }
    
    async def _execute_executive_task(self, task: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an executive coordination task using the executive agent.
        
        Args:
            task: The task to execute
            results: Current research results
            
        Returns:
            Task execution result
        """
        # Extract task parameters
        task_id = task["id"]
        task_description = task.get("description", "")
        executive_task = task.get("executive_task", "create_research_plan")
        
        # Create context for executive agent
        context = {
            "task_id": task_id,
            "task_description": task_description,
            "research_findings": results["collected_information"],
            "completed_tasks": self.current_execution_state["completed_tasks"],
            "current_step": self.current_execution_state["current_step"]
        }
        
        # Execute executive task
        try:
            executive_result = await self.executive_agent.execute(
                task=executive_task,
                context=context
            )
            
            # Process executive results
            if executive_result.get("status") == "success":
                # Create result object
                result = {
                    "status": "completed",
                    "task_id": task_id,
                    "information": {
                        f"executive_{executive_task}": executive_result
                    },
                    "executive_result": executive_result
                }
                
                return result
            else:
                # Handle executive task failure
                return {
                    "status": "failed",
                    "task_id": task_id,
                    "error": "Executive task execution failed",
                    "executive_result": executive_result,
                    "should_retry": False
                }
        except Exception as e:
            # Handle exceptions
            return {
                "status": "failed",
                "task_id": task_id,
                "error": f"Exception during executive task: {str(e)}",
                "should_retry": False
            }
    
    async def _analyze_research_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the collected research results.
        
        Args:
            results: Research results to analyze
            
        Returns:
            Analysis results
        """
        # Get strategic plan for context
        strategic_plan = self.current_research_plan.get("strategic_plan", {}) if self.current_research_plan else {}
        
        # Create context for analysis agent
        context = {
            "research_findings": results["collected_information"],
            "strategic_plan": strategic_plan
        }
        
        # Execute analysis
        analysis_result = await self.analysis_agent.execute(
            task="Perform comprehensive analysis of research findings",
            context=context
        )
        
        # Record analysis event
        await self.memory_system.episodic.add(
            key="research_analysis",
            value={
                "analysis": analysis_result
            }
        )
        
        return analysis_result
    
    async def _critique_research_findings(
        self,
        results: Dict[str, Any],
        analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Critique the research findings and analysis.
        
        Args:
            results: Research results to critique
            analysis_results: Analysis results to critique
            
        Returns:
            Critique results
        """
        # Get strategic plan for context
        strategic_plan = self.current_research_plan.get("strategic_plan", {}) if self.current_research_plan else {}
        
        # Create context for critique agent
        context = {
            "research_findings": results["collected_information"],
            "analyses": analysis_results,
            "research_plan": strategic_plan,
            "critique_type": "comprehensive"
        }
        
        # Execute critique
        critique_result = await self.critique_agent.execute(
            task="Perform comprehensive critique of research findings and analysis",
            context=context
        )
        
        # Record critique event
        await self.memory_system.episodic.add(
            key="research_critique",
            value={
                "critique": critique_result
            }
        )
        
        return critique_result
    
    async def _synthesize_research_output(
        self,
        results: Dict[str, Any],
        analysis_results: Dict[str, Any],
        critique_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synthesize the final research output.
        
        Args:
            results: Research results
            analysis_results: Analysis results
            critique_results: Critique results
            
        Returns:
            Synthesized research output
        """
        # Get strategic plan for context
        strategic_plan = self.current_research_plan.get("strategic_plan", {}) if self.current_research_plan else {}
        
        # Create context for synthesis agent
        context = {
            "research_findings": results["collected_information"],
            "analyses": analysis_results,
            "critiques": critique_results,
            "research_plan": strategic_plan,
            "synthesis_type": "comprehensive"
        }
        
        # Execute synthesis
        synthesis_result = await self.synthesis_agent.execute(
            task="Create comprehensive synthesis of research findings, analysis, and critique",
            context=context
        )
        
        # Record synthesis event
        await self.memory_system.episodic.add(
            key="research_synthesis",
            value={
                "synthesis": synthesis_result
            }
        )
        
        return synthesis_result
    
    def _create_default_browser(self) -> Browser:
        """
        Create a default browser configuration.
        
        Returns:
            Configured Browser instance
        """
        browser_config = BrowserConfig(
            headless=True,  # Set to False for debugging
            disable_security=True,
            new_context_config=BrowserContextConfig(
                disable_security=True,
                minimum_wait_page_load_time=1,
                maximum_wait_page_load_time=15,
            )
        )
        
        return Browser(config=browser_config)
    
    async def save_state(self, path: Optional[str] = None) -> bool:
        """
        Save the current state of the research agent.
        
        Args:
            path: Path to save state (defaults to storage_path/state.json)
            
        Returns:
            Success status
        """
        # Determine save path
        save_path = path or (os.path.join(self.storage_path, "state.json") if self.storage_path else None)
        
        if not save_path:
            return False
        
        # Create state object
        state = {
            "current_research_id": self.current_research_id,
            "current_execution_state": self.current_execution_state,
            "timestamp": time.time()
        }
        
        # Save state to file
        try:
            with open(save_path, 'w') as f:
                json.dump(state, f, indent=2)
            return True
        except Exception:
            return False
    
    async def load_state(self, path: Optional[str] = None) -> bool:
        """
        Load the state of the research agent.
        
        Args:
            path: Path to load state from (defaults to storage_path/state.json)
            
        Returns:
            Success status
        """
        # Determine load path
        load_path = path or (os.path.join(self.storage_path, "state.json") if self.storage_path else None)
        
        if not load_path or not os.path.exists(load_path):
            return False
        
        # Load state from file
        try:
            with open(load_path, 'r') as f:
                state = json.load(f)
            
            # Restore state
            self.current_research_id = state.get("current_research_id")
            self.current_execution_state = state.get("current_execution_state", {})
            
            return True
        except Exception:
            return False