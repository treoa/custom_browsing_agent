"""
Synthesis Agent Module

This module provides the SynthesisAgent class which is responsible for compiling
research findings into coherent, comprehensive outputs.
"""

from typing import Dict, List, Any, Optional
import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from .base_agent import BaseAgent


class SynthesisAgent(BaseAgent):
    """
    The Synthesis Agent compiles research findings into coherent outputs.
    
    This agent integrates research findings, analyses, and critiques into
    well-structured, coherent outputs tailored to the research objectives.
    """
    
    SYNTHESIS_SYSTEM_PROMPT = """You are a Synthesis Agent in a multi-agent research system.
Your role is to compile research findings into coherent, comprehensive outputs. You should:
1. Integrate information from multiple sources and analyses
2. Organize content with logical structure and flow
3. Highlight key insights and their implications
4. Maintain appropriate detail level based on user needs
5. Ensure proper source attribution and citation
6. Create clear visual organization through formatting
7. Balance comprehensiveness with readability
8. Represent uncertainty and confidence levels accurately

You excel at narrative integration, clear communication, and information organization.
Your syntheses should be well-structured, cohesive, and closely aligned with research objectives.
"""
    
    def __init__(
        self,
        llm: BaseChatModel,
        name: str = "Synthesizer",
        system_prompt: Optional[str] = None,
        memory: Optional[Any] = None,
    ):
        """
        Initialize a SynthesisAgent instance.
        
        Args:
            llm: The language model to use
            name: The name/identifier for this agent (default: "Synthesizer")
            system_prompt: Optional custom system prompt (uses default if None)
            memory: Optional memory system
        """
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt or self.SYNTHESIS_SYSTEM_PROMPT,
            memory=memory,
        )
        self.synthesis_results = {}
    
    async def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a synthesis task and return the result.
        
        Args:
            task: The synthesis task to execute
            context: Additional context including research findings and analyses
            
        Returns:
            A dictionary containing the synthesis results
        """
        context = context or {}
        research_findings = context.get("research_findings", {})
        research_plan = context.get("research_plan", {})
        analyses = context.get("analyses", {})
        critiques = context.get("critiques", {})
        
        if not research_findings:
            return {
                "status": "error",
                "error": "No research findings provided for synthesis"
            }
        
        synthesis_type = context.get("synthesis_type", "comprehensive")
        output_format = context.get("output_format", "report")
        
        if synthesis_type == "executive_summary":
            result = await self._create_executive_summary(
                research_findings, research_plan, analyses, critiques
            )
        elif synthesis_type == "detailed_report":
            result = await self._create_detailed_report(
                research_findings, research_plan, analyses, critiques
            )
        elif synthesis_type == "key_insights":
            result = await self._extract_key_insights(research_findings, analyses)
        elif output_format == "presentation":
            result = await self._create_presentation_format(
                research_findings, research_plan, analyses, critiques
            )
        else:  # comprehensive synthesis
            result = await self._create_comprehensive_synthesis(
                research_findings, research_plan, analyses, critiques
            )
        
        # Store synthesis results
        task_id = context.get("task_id", task)
        self.synthesis_results[task_id] = result
        
        # Add to history
        self.add_to_history(task, result)
        
        return result
    
    async def _create_comprehensive_synthesis(
        self,
        research_findings: Dict[str, Any],
        research_plan: Dict[str, Any],
        analyses: Dict[str, Any],
        critiques: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a comprehensive synthesis of all research components.
        
        Args:
            research_findings: The research findings
            research_plan: The original research plan
            analyses: Analysis results
            critiques: Critique results
            
        Returns:
            Comprehensive synthesis
        """
        objectives = research_plan.get("objectives", [])
        objectives_str = "\n".join([f"- {obj}" for obj in objectives]) if objectives else "Not specified"
        
        key_questions = research_plan.get("key_questions", [])
        questions_str = "\n".join([f"- {q}" for q in key_questions]) if key_questions else "Not specified"
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
Create a comprehensive synthesis of the following research components.

RESEARCH OBJECTIVES:
{objectives_str}

KEY QUESTIONS:
{questions_str}

RESEARCH FINDINGS:
{json.dumps(research_findings, indent=2)}

ANALYSES:
{json.dumps(analyses, indent=2)}

CRITIQUES:
{json.dumps(critiques, indent=2)}

Create a comprehensive synthesis that:
1. Addresses all research objectives and key questions
2. Integrates findings, analyses, and critiques into a cohesive narrative
3. Organizes information with a clear, logical structure
4. Highlights key insights and their implications
5. Acknowledges limitations and areas of uncertainty
6. Provides proper attribution for information sources
7. Balances comprehensiveness with clarity and readability

Your synthesis should have the following sections:
1. Executive Summary
2. Introduction and Research Objectives
3. Methodology
4. Key Findings (organized by main themes/topics)
5. Analysis and Interpretation
6. Limitations and Caveats
7. Conclusions and Implications
8. References/Sources

Format your response as a markdown document with clear headings, subheadings, and formatting.
""")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        synthesis = response.content
        
        return {
            "status": "success",
            "synthesis_type": "comprehensive",
            "content": synthesis
        }
    
    async def _create_executive_summary(
        self,
        research_findings: Dict[str, Any],
        research_plan: Dict[str, Any],
        analyses: Dict[str, Any],
        critiques: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a concise executive summary of research findings.
        
        Args:
            research_findings: The research findings
            research_plan: The original research plan
            analyses: Analysis results
            critiques: Critique results
            
        Returns:
            Executive summary
        """
        objectives = research_plan.get("objectives", [])
        objectives_str = "\n".join([f"- {obj}" for obj in objectives]) if objectives else "Not specified"
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
Create a concise executive summary of the research findings.

RESEARCH OBJECTIVES:
{objectives_str}

RESEARCH FINDINGS:
{json.dumps(research_findings, indent=2)}

ANALYSES:
{json.dumps(analyses, indent=2)}

Create an executive summary that:
1. Is concise (500-750 words)
2. Highlights the most important findings related to research objectives
3. Presents key insights and their implications
4. Acknowledges major limitations or caveats
5. Uses clear, direct language suitable for busy executives

Format your response as a well-structured executive summary with appropriate paragraphs and formatting.
""")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        summary = response.content
        
        return {
            "status": "success",
            "synthesis_type": "executive_summary",
            "content": summary
        }
    
    async def _create_detailed_report(
        self,
        research_findings: Dict[str, Any],
        research_plan: Dict[str, Any],
        analyses: Dict[str, Any],
        critiques: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a detailed research report.
        
        Args:
            research_findings: The research findings
            research_plan: The original research plan
            analyses: Analysis results
            critiques: Critique results
            
        Returns:
            Detailed report
        """
        objectives = research_plan.get("objectives", [])
        objectives_str = "\n".join([f"- {obj}" for obj in objectives]) if objectives else "Not specified"
        
        key_questions = research_plan.get("key_questions", [])
        questions_str = "\n".join([f"- {q}" for q in key_questions]) if key_questions else "Not specified"
        
        methodology = research_plan.get("strategy", [])
        methodology_str = "\n".join([
            f"- Step {step.get('step', '')}: {step.get('description', '')}"
            for step in methodology
        ]) if methodology else "Not specified"
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
Create a detailed research report from the following components.

RESEARCH OBJECTIVES:
{objectives_str}

KEY QUESTIONS:
{questions_str}

METHODOLOGY:
{methodology_str}

RESEARCH FINDINGS:
{json.dumps(research_findings, indent=2)}

ANALYSES:
{json.dumps(analyses, indent=2)}

CRITIQUES:
{json.dumps(critiques, indent=2)}

Create a detailed research report that:
1. Thoroughly addresses all research objectives and key questions
2. Provides in-depth coverage of findings with supporting evidence
3. Includes detailed analysis and interpretation
4. Acknowledges limitations, contradictions, and uncertainties
5. Discusses implications and potential applications
6. Includes comprehensive attribution and citations

Your report should have the following sections:
1. Executive Summary
2. Introduction and Background
3. Research Objectives and Questions
4. Methodology
5. Findings (with detailed subsections for each major area)
6. Analysis and Discussion
7. Limitations and Areas for Further Research
8. Conclusions and Recommendations
9. References
10. Appendices (if applicable)

Format your response as a comprehensive markdown document with clear headings, subheadings, and formatting.
Use tables, lists, and other formatting elements to enhance readability where appropriate.
""")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        report = response.content
        
        return {
            "status": "success",
            "synthesis_type": "detailed_report",
            "content": report
        }
    
    async def _extract_key_insights(
        self,
        research_findings: Dict[str, Any],
        analyses: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract and synthesize key insights from research findings.
        
        Args:
            research_findings: The research findings
            analyses: Analysis results
            
        Returns:
            Key insights synthesis
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
Extract and synthesize key insights from the following research components.

RESEARCH FINDINGS:
{json.dumps(research_findings, indent=2)}

ANALYSES:
{json.dumps(analyses, indent=2)}

Extract the most important insights that:
1. Represent significant discoveries or conclusions
2. Have meaningful implications or applications
3. Challenge or confirm existing understanding
4. Suggest new patterns, relationships, or trends
5. Address critical gaps in knowledge

For each insight, include:
1. A clear, concise statement of the insight
2. Supporting evidence from the research
3. Implications or significance of the insight
4. Confidence level in the insight
5. Related questions or areas for further exploration

Format your response as a markdown document with clear headings for each key insight.
Prioritize quality over quantity - focus on the most valuable and well-supported insights.
""")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        insights = response.content
        
        return {
            "status": "success",
            "synthesis_type": "key_insights",
            "content": insights
        }
    
    async def _create_presentation_format(
        self,
        research_findings: Dict[str, Any],
        research_plan: Dict[str, Any],
        analyses: Dict[str, Any],
        critiques: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a presentation-formatted synthesis of research findings.
        
        Args:
            research_findings: The research findings
            research_plan: The original research plan
            analyses: Analysis results
            critiques: Critique results
            
        Returns:
            Presentation-formatted synthesis
        """
        objectives = research_plan.get("objectives", [])
        objectives_str = "\n".join([f"- {obj}" for obj in objectives]) if objectives else "Not specified"
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
Create a presentation-formatted synthesis of the research findings.

RESEARCH OBJECTIVES:
{objectives_str}

RESEARCH FINDINGS:
{json.dumps(research_findings, indent=2)}

ANALYSES:
{json.dumps(analyses, indent=2)}

Create a presentation-formatted synthesis that:
1. Is organized into slides with clear titles
2. Uses concise bullet points rather than paragraphs
3. Highlights key findings and insights visually
4. Focuses on the most important and impactful information
5. Includes an executive summary, key findings, and conclusions
6. Is suitable for presentation to stakeholders

Format your response as a markdown document with each slide clearly delineated.
Use the format "## Slide X: Title" for slide headers, followed by content in bullet point format.
Include approximately 10-15 slides covering the essential elements of the research.
""")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        presentation = response.content
        
        return {
            "status": "success",
            "synthesis_type": "presentation",
            "content": presentation
        }