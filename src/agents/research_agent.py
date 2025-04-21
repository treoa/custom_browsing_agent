"""
Research Agent Module

This module provides the ResearchAgent class which is responsible for gathering
information from web sources using the browser-use library.
"""

from typing import Dict, List, Any, Optional
import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from browser_use.agent.service import Agent as BrowserAgent
from browser_use.browser.browser import Browser

from .base_agent import BaseAgent


class ResearchAgent(BaseAgent):
    """
    The Research Agent is responsible for gathering information from web sources.
    
    This agent uses browser-use to navigate and extract information from websites
    as part of the multi-agent research system.
    """
    
    RESEARCH_SYSTEM_PROMPT = """You are a Research Agent in a multi-agent research system.
Your role is to gather information from web sources using browser navigation. You should:
1. Follow the specific research tasks assigned to you
2. Navigate websites effectively to find relevant information
3. Extract and summarize key information related to the research objectives
4. Verify information across multiple sources when possible
5. Document your research process and sources for attribution
6. Focus on depth and thoroughness rather than breadth
7. Collect both factual information and conceptual understanding
8. Identify any contradictions or gaps in the information

When researching, you should progressively deepen your exploration of each topic,
avoiding shallow investigations. You should track source reliability and diversity.
"""
    
    def __init__(
        self,
        llm: BaseChatModel,
        browser: Browser,
        name: str = "WebResearcher",
        system_prompt: Optional[str] = None,
        memory: Optional[Any] = None,
        max_actions_per_step: int = 5,
        browser_context: Optional[Any] = None,
    ):
        """
        Initialize a ResearchAgent instance.
        
        Args:
            llm: The language model to use
            browser: The browser instance for web navigation
            name: The name/identifier for this agent (default: "WebResearcher")
            system_prompt: Optional custom system prompt (uses default if None)
            memory: Optional memory system
            max_actions_per_step: Maximum browser actions per step
            browser_context: Optional browser context to reuse an existing one
        """
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt or self.RESEARCH_SYSTEM_PROMPT,
            memory=memory,
        )
        self.browser = browser
        self.browser_context = browser_context  # Store the browser context
        self.max_actions_per_step = max_actions_per_step
        self.sources_visited = []
        self.research_findings = {}
    
    async def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a research task using browser navigation.
        
        Args:
            task: The research task description
            context: Additional context for the research task
            
        Returns:
            Dictionary containing research findings and metadata
        """
        context = context or {}
        
        # Create a browser-use agent for this specific task
        browser_agent = BrowserAgent(
            task=task,
            llm=self.llm,
            browser=self.browser,
            browser_context=self.browser_context,  # Pass the browser context if available
            max_actions_per_step=self.max_actions_per_step,
            system_prompt=self._generate_task_prompt(task, context),
        )
        
        # Execute the research task
        browser_result = await browser_agent.run(
            max_steps=context.get("max_steps", 20),
            validate_output=True
        )
        
        # Process browser agent output
        extracted_info = []
        sources = []
        
        for step in browser_result.steps:
            # Extract content from each step
            if hasattr(step, 'extracted_content') and step.extracted_content:
                extracted_info.append({
                    "content": step.extracted_content,
                    "url": step.current_url if hasattr(step, 'current_url') else "Unknown",
                    "step": step.step_number if hasattr(step, 'step_number') else 0
                })
            
            # Track sources
            if hasattr(step, 'current_url') and step.current_url:
                if step.current_url not in sources:
                    sources.append(step.current_url)
        
        # Consolidate research findings
        research_summary = await self._summarize_findings(extracted_info, task)
        
        result = {
            "task": task,
            "status": "completed",
            "sources_visited": sources,
            "extracted_information": extracted_info,
            "research_summary": research_summary,
            "browser_result": browser_result
        }
        
        # Update agent state
        self.sources_visited.extend(sources)
        self.research_findings[task] = research_summary
        
        # Add to history
        self.add_to_history(task, result)
        
        return result
    
    def _generate_task_prompt(self, task: str, context: Dict[str, Any]) -> str:
        """
        Generate a specific prompt for the browser agent based on the task and context.
        
        Args:
            task: The research task
            context: Additional context
            
        Returns:
            Customized system prompt for browser agent
        """
        # Add research objectives and any specific instructions to the browser agent
        objectives = context.get("objectives", [])
        objectives_str = "\n".join([f"- {obj}" for obj in objectives]) if objectives else "Not specified"
        
        key_questions = context.get("key_questions", [])
        questions_str = "\n".join([f"- {q}" for q in key_questions]) if key_questions else "Not specified"
        
        return f"""{self.system_prompt}

RESEARCH TASK: {task}

RESEARCH OBJECTIVES:
{objectives_str}

KEY QUESTIONS TO ANSWER:
{questions_str}

ADDITIONAL INSTRUCTIONS:
1. Focus on thorough, in-depth research rather than quick, shallow results
2. Document all sources carefully for later verification and attribution
3. Spend time understanding concepts deeply before moving on
4. Verify information across multiple sources when possible
5. Identify contradictions, gaps, or uncertainties in the information

Remember to avoid premature task completion. Continue researching until you have 
gathered comprehensive information that addresses all aspects of the task.
"""
    
    async def _summarize_findings(self, extracted_info: List[Dict[str, Any]], task: str) -> Dict[str, Any]:
        """
        Summarize collected research findings.
        
        Args:
            extracted_info: List of extracted information with sources
            task: The original research task
            
        Returns:
            Structured summary of research findings
        """
        # Combine all extracted content for summarization
        combined_content = "\n\n".join([
            f"FROM {item['url']}:\n{item['content']}" 
            for item in extracted_info
        ])
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
Summarize the following information collected during research on: "{task}"

COLLECTED INFORMATION:
{combined_content}

Create a comprehensive summary that:
1. Addresses the research task directly
2. Organizes information logically by subtopic
3. Highlights key findings and insights
4. Notes any contradictions or uncertainties
5. Identifies information gaps that may require further research

Format your response as a JSON object with the following structure:
{{
    "summary": "overall summary of findings",
    "key_findings": ["finding1", "finding2", ...],
    "subtopics": [
        {{
            "topic": "subtopic name",
            "content": "detailed information"
        }}
    ],
    "contradictions": ["contradiction1", "contradiction2", ...],
    "information_gaps": ["gap1", "gap2", ...],
    "source_assessment": "evaluation of source quality and diversity",
    "confidence_level": "high/medium/low",
    "recommended_next_steps": ["step1", "step2", ...]
}}
""")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        try:
            # Extract JSON from the response content
            summary_json = response.content
            if "```json" in summary_json:
                summary_json = summary_json.split("```json")[1].split("```")[0].strip()
            elif "```" in summary_json:
                summary_json = summary_json.split("```")[1].split("```")[0].strip()
                
            summary = json.loads(summary_json)
            return summary
        except Exception as e:
            return {
                "error": f"Failed to parse summary: {str(e)}",
                "raw_summary": response.content
            }