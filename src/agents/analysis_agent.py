"""
Analysis Agent Module

This module provides the AnalysisAgent class which is responsible for processing
and analyzing information gathered by research agents.
"""

from typing import Dict, List, Any, Optional
import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from .base_agent import BaseAgent


class AnalysisAgent(BaseAgent):
    """
    The Analysis Agent processes and analyzes information gathered by research agents.
    
    This agent identifies patterns, extracts insights, and organizes information
    to support the overall research objectives.
    """
    
    ANALYSIS_SYSTEM_PROMPT = """You are an Analysis Agent in a multi-agent research system.
Your role is to process and analyze information gathered by research agents. You should:
1. Identify patterns and relationships across multiple sources
2. Extract key insights and implications from research findings
3. Organize information hierarchically and conceptually
4. Recognize contradictions and inconsistencies in the data
5. Assess the reliability and relevance of information
6. Identify gaps requiring further investigation
7. Integrate qualitative and quantitative information
8. Provide analytical frameworks for understanding complex topics

You excel at pattern recognition, critical thinking, and systematic organization of information.
Your analysis should be thorough, balanced, and closely tied to the research objectives.
"""
    
    def __init__(
        self,
        llm: BaseChatModel,
        name: str = "Analyst",
        system_prompt: Optional[str] = None,
        memory: Optional[Any] = None,
    ):
        """
        Initialize an AnalysisAgent instance.
        
        Args:
            llm: The language model to use
            name: The name/identifier for this agent (default: "Analyst")
            system_prompt: Optional custom system prompt (uses default if None)
            memory: Optional memory system
        """
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt or self.ANALYSIS_SYSTEM_PROMPT,
            memory=memory,
        )
        self.analysis_results = {}
    
    async def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute an analysis task and return the result.
        
        Args:
            task: The analysis task to execute
            context: Additional context including research findings to analyze
            
        Returns:
            A dictionary containing the analysis results
        """
        context = context or {}
        research_findings = context.get("research_findings", {})
        
        if not research_findings:
            return {
                "status": "error",
                "error": "No research findings provided for analysis"
            }
        
        analysis_type = context.get("analysis_type", "general")
        
        if analysis_type == "pattern_recognition":
            result = await self._perform_pattern_analysis(research_findings, context)
        elif analysis_type == "contradiction_analysis":
            result = await self._analyze_contradictions(research_findings, context)
        elif analysis_type == "gap_identification":
            result = await self._identify_gaps(research_findings, context)
        elif analysis_type == "source_reliability":
            result = await self._assess_source_reliability(research_findings, context)
        elif analysis_type == "concept_mapping":
            result = await self._create_concept_map(research_findings, context)
        else:  # general analysis
            result = await self._perform_general_analysis(research_findings, context)
        
        # Store analysis results
        task_id = context.get("task_id", task)
        self.analysis_results[task_id] = result
        
        # Add to history
        self.add_to_history(task, result)
        
        return result
    
    async def _perform_general_analysis(
        self, research_findings: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform a general analysis of research findings.
        
        Args:
            research_findings: The research findings to analyze
            context: Additional context including research objectives
            
        Returns:
            Analysis results
        """
        research_objectives = context.get("objectives", [])
        objectives_str = "\n".join([f"- {obj}" for obj in research_objectives]) if research_objectives else "Not specified"
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
Perform a comprehensive analysis of the following research findings.

RESEARCH OBJECTIVES:
{objectives_str}

RESEARCH FINDINGS:
{json.dumps(research_findings, indent=2)}

Analyze the findings to identify:
1. Key patterns and relationships between concepts
2. Main insights and their implications
3. The logical organization and hierarchy of information
4. Any contradictions or inconsistencies
5. The reliability and relevance of different information pieces
6. Significant gaps requiring further investigation
7. How qualitative and quantitative information integrate
8. Analytical frameworks for understanding the topic

Format your response as a JSON object with the following structure:
{{
    "key_patterns": [
        {{
            "pattern": "description of pattern",
            "supporting_evidence": ["evidence1", "evidence2", ...],
            "significance": "why this pattern matters"
        }}
    ],
    "main_insights": [
        {{
            "insight": "description of insight",
            "implications": ["implication1", "implication2", ...],
            "confidence": "high/medium/low"
        }}
    ],
    "information_structure": {{
        "primary_categories": ["category1", "category2", ...],
        "hierarchical_organization": "description of how information is organized",
        "concept_relationships": "description of how key concepts relate"
    }},
    "contradictions": [
        {{
            "description": "nature of contradiction",
            "conflicting_elements": ["element1", "element2"],
            "possible_resolutions": ["resolution1", "resolution2", ...]
        }}
    ],
    "information_quality": {{
        "most_reliable_sources": ["source1", "source2", ...],
        "less_reliable_elements": ["element1", "element2", ...],
        "potential_biases": ["bias1", "bias2", ...]
    }},
    "information_gaps": [
        {{
            "gap": "description of missing information",
            "importance": "high/medium/low",
            "research_suggestions": ["suggestion1", "suggestion2", ...]
        }}
    ],
    "analytical_framework": "proposed framework for understanding the topic",
    "overall_assessment": "comprehensive assessment of the research findings"
}}
""")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        try:
            # Extract JSON from the response content
            analysis_json = response.content
            if "```json" in analysis_json:
                analysis_json = analysis_json.split("```json")[1].split("```")[0].strip()
            elif "```" in analysis_json:
                analysis_json = analysis_json.split("```")[1].split("```")[0].strip()
                
            analysis = json.loads(analysis_json)
            
            return {
                "status": "success",
                "analysis_type": "general",
                "analysis": analysis
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to parse analysis: {str(e)}",
                "raw_analysis": response.content
            }
    
    async def _perform_pattern_analysis(
        self, research_findings: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Identify patterns and relationships across research findings.
        
        Args:
            research_findings: The research findings to analyze
            context: Additional context
            
        Returns:
            Pattern analysis results
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
Perform pattern analysis on the following research findings.

RESEARCH FINDINGS:
{json.dumps(research_findings, indent=2)}

Identify recurring patterns, trends, relationships, and correlations across the data.
Focus on both explicit patterns (directly stated) and implicit patterns (indirectly suggested).

Format your response as a JSON object with the following structure:
{{
    "explicit_patterns": [
        {{
            "pattern": "description of pattern",
            "supporting_evidence": ["evidence1", "evidence2", ...],
            "significance": "why this pattern matters"
        }}
    ],
    "implicit_patterns": [
        {{
            "pattern": "description of pattern",
            "supporting_evidence": ["evidence1", "evidence2", ...],
            "significance": "why this pattern matters",
            "confidence": "high/medium/low"
        }}
    ],
    "relationships": [
        {{
            "entities": ["entity1", "entity2"],
            "relationship_type": "type of relationship",
            "description": "detailed description of relationship",
            "evidence": ["evidence1", "evidence2", ...]
        }}
    ],
    "trends": [
        {{
            "trend": "description of trend",
            "direction": "increasing/decreasing/fluctuating/stable",
            "supporting_evidence": ["evidence1", "evidence2", ...],
            "implications": ["implication1", "implication2", ...]
        }}
    ],
    "anomalies": [
        {{
            "anomaly": "description of anomaly",
            "why_significant": "explanation of significance",
            "possible_explanations": ["explanation1", "explanation2", ...]
        }}
    ]
}}
""")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        try:
            # Extract JSON from the response content
            analysis_json = response.content
            if "```json" in analysis_json:
                analysis_json = analysis_json.split("```json")[1].split("```")[0].strip()
            elif "```" in analysis_json:
                analysis_json = analysis_json.split("```")[1].split("```")[0].strip()
                
            pattern_analysis = json.loads(analysis_json)
            
            return {
                "status": "success",
                "analysis_type": "pattern_recognition",
                "pattern_analysis": pattern_analysis
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to parse pattern analysis: {str(e)}",
                "raw_analysis": response.content
            }
    
    async def _analyze_contradictions(
        self, research_findings: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Identify and analyze contradictions within research findings.
        
        Args:
            research_findings: The research findings to analyze
            context: Additional context
            
        Returns:
            Contradiction analysis results
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
Analyze the following research findings to identify contradictions and inconsistencies.

RESEARCH FINDINGS:
{json.dumps(research_findings, indent=2)}

Look for:
1. Direct contradictions where sources provide opposite information
2. Partial contradictions where information is inconsistent but not opposite
3. Contextual contradictions where information is valid in different contexts
4. Temporal contradictions due to changes over time
5. Methodological contradictions due to different approaches

Format your response as a JSON object with the following structure:
{{
    "direct_contradictions": [
        {{
            "contradiction": "description of contradiction",
            "source_a": "source information",
            "source_b": "conflicting source information",
            "assessment": "analysis of contradiction",
            "resolution_approach": "suggested approach to resolve"
        }}
    ],
    "partial_contradictions": [
        {{
            "contradiction": "description of contradiction",
            "conflicting_elements": ["element1", "element2", ...],
            "nuanced_analysis": "explanation of nuance",
            "reconciliation_possibility": "high/medium/low",
            "reconciliation_approach": "suggested approach"
        }}
    ],
    "contextual_contradictions": [
        {{
            "contradiction": "description of contradiction",
            "context_a": "first context",
            "context_b": "second context",
            "explanation": "why information differs by context"
        }}
    ],
    "temporal_contradictions": [
        {{
            "contradiction": "description of contradiction",
            "earlier_information": "information from earlier time",
            "later_information": "information from later time",
            "timeframes": "relevant timeframes",
            "explanation": "explanation of change over time"
        }}
    ],
    "methodological_contradictions": [
        {{
            "contradiction": "description of contradiction",
            "method_a": "first methodology",
            "method_b": "second methodology",
            "explanation": "why methods produced different results",
            "methodology_assessment": "which method is more reliable and why"
        }}
    ]
}}
""")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        try:
            # Extract JSON from the response content
            analysis_json = response.content
            if "```json" in analysis_json:
                analysis_json = analysis_json.split("```json")[1].split("```")[0].strip()
            elif "```" in analysis_json:
                analysis_json = analysis_json.split("```")[1].split("```")[0].strip()
                
            contradiction_analysis = json.loads(analysis_json)
            
            return {
                "status": "success",
                "analysis_type": "contradiction_analysis",
                "contradiction_analysis": contradiction_analysis
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to parse contradiction analysis: {str(e)}",
                "raw_analysis": response.content
            }
    
    async def _identify_gaps(
        self, research_findings: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Identify information gaps in the research findings.
        
        Args:
            research_findings: The research findings to analyze
            context: Additional context including research objectives
            
        Returns:
            Gap analysis results
        """
        research_objectives = context.get("objectives", [])
        objectives_str = "\n".join([f"- {obj}" for obj in research_objectives]) if research_objectives else "Not specified"
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
Identify information gaps in the following research findings.

RESEARCH OBJECTIVES:
{objectives_str}

RESEARCH FINDINGS:
{json.dumps(research_findings, indent=2)}

Identify:
1. Missing information needed to fully address research objectives
2. Underdeveloped areas that need deeper investigation
3. Questions that remain unanswered
4. Areas with insufficient source diversity
5. Topics mentioned but not adequately explored

Format your response as a JSON object with the following structure:
{{
    "critical_gaps": [
        {{
            "gap": "description of critical information gap",
            "related_objective": "which research objective this affects",
            "impact": "how this gap affects overall understanding",
            "research_suggestions": ["suggestion1", "suggestion2", ...]
        }}
    ],
    "underdeveloped_areas": [
        {{
            "area": "topic area needing more depth",
            "current_depth": "assessment of current coverage",
            "needed_depth": "explanation of information needed",
            "research_suggestions": ["suggestion1", "suggestion2", ...]
        }}
    ],
    "unanswered_questions": [
        {{
            "question": "specific unanswered question",
            "importance": "high/medium/low",
            "research_suggestions": ["suggestion1", "suggestion2", ...]
        }}
    ],
    "source_diversity_gaps": [
        {{
            "topic": "topic with insufficient source diversity",
            "current_sources": ["source1", "source2", ...],
            "suggested_additional_sources": ["source3", "source4", ...]
        }}
    ],
    "overall_gap_assessment": "comprehensive assessment of information gaps",
    "prioritized_research_needs": ["priority1", "priority2", ...]
}}
""")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        try:
            # Extract JSON from the response content
            analysis_json = response.content
            if "```json" in analysis_json:
                analysis_json = analysis_json.split("```json")[1].split("```")[0].strip()
            elif "```" in analysis_json:
                analysis_json = analysis_json.split("```")[1].split("```")[0].strip()
                
            gap_analysis = json.loads(analysis_json)
            
            return {
                "status": "success",
                "analysis_type": "gap_identification",
                "gap_analysis": gap_analysis
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to parse gap analysis: {str(e)}",
                "raw_analysis": response.content
            }
    
    async def _assess_source_reliability(
        self, research_findings: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess the reliability of information sources.
        
        Args:
            research_findings: The research findings to analyze
            context: Additional context
            
        Returns:
            Source reliability assessment
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
Assess the reliability of information sources in the following research findings.

RESEARCH FINDINGS:
{json.dumps(research_findings, indent=2)}

Evaluate:
1. The credibility and authority of sources
2. Potential biases or conflicts of interest
3. Currency and timeliness of information
4. Methodological rigor where applicable
5. Consistency with other reliable sources
6. Primary vs. secondary source status

Format your response as a JSON object with the following structure:
{{
    "source_assessments": [
        {{
            "source": "source identifier",
            "credibility_score": 1-10,
            "authority_assessment": "assessment of source authority",
            "potential_biases": ["bias1", "bias2", ...],
            "currency": "assessment of information timeliness",
            "consistency_with_other_sources": "high/medium/low",
            "source_type": "primary/secondary/tertiary",
            "overall_reliability": "high/medium/low",
            "rationale": "explanation of reliability assessment"
        }}
    ],
    "most_reliable_sources": ["source1", "source2", ...],
    "less_reliable_sources": ["source3", "source4", ...],
    "source_diversity_assessment": "assessment of overall source diversity",
    "reliability_improvement_suggestions": ["suggestion1", "suggestion2", ...]
}}
""")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        try:
            # Extract JSON from the response content
            analysis_json = response.content
            if "```json" in analysis_json:
                analysis_json = analysis_json.split("```json")[1].split("```")[0].strip()
            elif "```" in analysis_json:
                analysis_json = analysis_json.split("```")[1].split("```")[0].strip()
                
            reliability_assessment = json.loads(analysis_json)
            
            return {
                "status": "success",
                "analysis_type": "source_reliability",
                "reliability_assessment": reliability_assessment
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to parse reliability assessment: {str(e)}",
                "raw_analysis": response.content
            }
    
    async def _create_concept_map(
        self, research_findings: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a concept map from research findings.
        
        Args:
            research_findings: The research findings to analyze
            context: Additional context
            
        Returns:
            Concept mapping results
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
Create a concept map from the following research findings.

RESEARCH FINDINGS:
{json.dumps(research_findings, indent=2)}

Identify key concepts and their relationships to create a structured representation
of the knowledge domain. Focus on:
1. Primary concepts and subconcepts
2. Relationships between concepts (causal, hierarchical, temporal, etc.)
3. The overall structure and organization of the domain
4. Central/core concepts vs. peripheral concepts

Format your response as a JSON object with the following structure:
{{
    "concepts": [
        {{
            "id": "unique_id",
            "name": "concept name",
            "description": "brief description",
            "importance": "core/supporting/peripheral",
            "related_sources": ["source1", "source2", ...]
        }}
    ],
    "relationships": [
        {{
            "source": "concept_id1",
            "target": "concept_id2",
            "type": "relationship type (e.g., causes, contains, precedes)",
            "description": "description of relationship",
            "strength": "strong/moderate/weak",
            "supporting_evidence": ["evidence1", "evidence2", ...]
        }}
    ],
    "clusters": [
        {{
            "name": "cluster name",
            "concept_ids": ["concept_id1", "concept_id2", ...],
            "theme": "theme description"
        }}
    ],
    "central_concepts": ["concept_id1", "concept_id2", ...],
    "map_structure": "description of overall concept map structure"
}}
""")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        try:
            # Extract JSON from the response content
            concept_map_json = response.content
            if "```json" in concept_map_json:
                concept_map_json = concept_map_json.split("```json")[1].split("```")[0].strip()
            elif "```" in concept_map_json:
                concept_map_json = concept_map_json.split("```")[1].split("```")[0].strip()
                
            concept_map = json.loads(concept_map_json)
            
            return {
                "status": "success",
                "analysis_type": "concept_mapping",
                "concept_map": concept_map
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to parse concept map: {str(e)}",
                "raw_concept_map": response.content
            }