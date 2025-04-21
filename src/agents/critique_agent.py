"""
Critique Agent Module

This module provides the CritiqueAgent class which is responsible for internal
evaluation and quality assurance of research findings.
"""

from typing import Dict, List, Any, Optional
import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from .base_agent import BaseAgent


class CritiqueAgent(BaseAgent):
    """
    The Critique Agent provides internal evaluation and quality assurance.
    
    This agent evaluates research completeness, information quality, logical consistency,
    and identifies biases or limitations in research findings.
    """
    
    CRITIQUE_SYSTEM_PROMPT = """You are a Critique Agent in a multi-agent research system.
Your role is to evaluate research quality and completeness. You should:
1. Assess the completeness of research relative to objectives
2. Verify the quality and reliability of information
3. Check for logical consistency across findings
4. Identify potential biases or limitations
5. Analyze whether research addresses all key questions
6. Evaluate the methodological soundness of the research
7. Apply rigorous quality standards to all research outputs
8. Provide constructive feedback for improvement

You use a critical, detail-oriented approach to ensure high research standards.
Your evaluations should be fair, balanced, and focused on improving research quality.
You should identify both strengths and weaknesses in the research findings.
"""
    
    def __init__(
        self,
        llm: BaseChatModel,
        name: str = "Critic",
        system_prompt: Optional[str] = None,
        memory: Optional[Any] = None,
    ):
        """
        Initialize a CritiqueAgent instance.
        
        Args:
            llm: The language model to use
            name: The name/identifier for this agent (default: "Critic")
            system_prompt: Optional custom system prompt (uses default if None)
            memory: Optional memory system
        """
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt or self.CRITIQUE_SYSTEM_PROMPT,
            memory=memory,
        )
        self.critique_results = {}
    
    async def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a critique task and return the result.
        
        Args:
            task: The critique task to execute
            context: Additional context including research findings to critique
            
        Returns:
            A dictionary containing the critique results
        """
        context = context or {}
        research_findings = context.get("research_findings", {})
        research_plan = context.get("research_plan", {})
        
        if not research_findings:
            return {
                "status": "error",
                "error": "No research findings provided for critique"
            }
        
        critique_type = context.get("critique_type", "completeness")
        
        if critique_type == "completeness":
            result = await self._evaluate_completeness(research_findings, research_plan)
        elif critique_type == "quality":
            result = await self._evaluate_quality(research_findings)
        elif critique_type == "consistency":
            result = await self._check_consistency(research_findings)
        elif critique_type == "bias":
            result = await self._identify_bias(research_findings)
        elif critique_type == "comprehensive":
            result = await self._comprehensive_critique(research_findings, research_plan)
        else:
            return {
                "status": "error",
                "error": f"Unknown critique type: {critique_type}"
            }
        
        # Store critique results
        task_id = context.get("task_id", task)
        self.critique_results[task_id] = result
        
        # Add to history
        self.add_to_history(task, result)
        
        return result
    
    async def _evaluate_completeness(
        self, research_findings: Dict[str, Any], research_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate the completeness of research findings relative to objectives.
        
        Args:
            research_findings: The research findings to evaluate
            research_plan: The original research plan with objectives
            
        Returns:
            Completeness evaluation
        """
        objectives = research_plan.get("objectives", [])
        objectives_str = "\n".join([f"- {obj}" for obj in objectives]) if objectives else "Not specified"
        
        key_questions = research_plan.get("key_questions", [])
        questions_str = "\n".join([f"- {q}" for q in key_questions]) if key_questions else "Not specified"
        
        completion_criteria = research_plan.get("completion_criteria", [])
        criteria_str = "\n".join([
            f"- {criterion.get('criterion', '')}: {criterion.get('measurement', '')}"
            for criterion in completion_criteria
        ]) if completion_criteria else "Not specified"
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
Evaluate the completeness of the following research findings relative to objectives.

RESEARCH OBJECTIVES:
{objectives_str}

KEY QUESTIONS:
{questions_str}

COMPLETION CRITERIA:
{criteria_str}

RESEARCH FINDINGS:
{json.dumps(research_findings, indent=2)}

Assess:
1. How completely the findings address each research objective
2. Whether all key questions have been answered
3. If the research meets all specified completion criteria
4. If the depth of research is sufficient for each topic area
5. Whether any important aspects of the topic have been omitted

Format your response as a JSON object with the following structure:
{{
    "objectives_assessment": [
        {{
            "objective": "research objective",
            "completeness": 0-100,
            "evaluation": "detailed assessment",
            "missing_elements": ["element1", "element2", ...]
        }}
    ],
    "questions_assessment": [
        {{
            "question": "key question",
            "answered": true/false,
            "completeness": 0-100,
            "evaluation": "detailed assessment",
            "missing_information": ["info1", "info2", ...]
        }}
    ],
    "criteria_assessment": [
        {{
            "criterion": "completion criterion",
            "satisfied": true/false,
            "evaluation": "detailed assessment",
            "evidence": ["evidence1", "evidence2", ...]
        }}
    ],
    "depth_assessment": {{
        "sufficient_depth_areas": ["area1", "area2", ...],
        "insufficient_depth_areas": [
            {{
                "area": "topic area",
                "current_depth": "assessment of current depth",
                "needed_depth": "description of needed depth"
            }}
        ]
    }},
    "omitted_aspects": ["aspect1", "aspect2", ...],
    "overall_completeness_score": 0-100,
    "overall_assessment": "comprehensive assessment of completeness",
    "recommended_improvements": ["improvement1", "improvement2", ...]
}}
""")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        try:
            # Extract JSON from the response content
            completeness_json = response.content
            if "```json" in completeness_json:
                completeness_json = completeness_json.split("```json")[1].split("```")[0].strip()
            elif "```" in completeness_json:
                completeness_json = completeness_json.split("```")[1].split("```")[0].strip()
                
            completeness_evaluation = json.loads(completeness_json)
            
            return {
                "status": "success",
                "critique_type": "completeness",
                "completeness_evaluation": completeness_evaluation
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to parse completeness evaluation: {str(e)}",
                "raw_evaluation": response.content
            }
    
    async def _evaluate_quality(self, research_findings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the quality and reliability of research findings.
        
        Args:
            research_findings: The research findings to evaluate
            
        Returns:
            Quality evaluation
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
Evaluate the quality and reliability of the following research findings.

RESEARCH FINDINGS:
{json.dumps(research_findings, indent=2)}

Assess:
1. The credibility and authority of information sources
2. The accuracy and precision of information
3. The recency and relevance of information
4. The diversity of sources and perspectives
5. The use of primary vs secondary sources
6. The presence of verifiable facts vs opinions
7. The methodological soundness where applicable

Format your response as a JSON object with the following structure:
{{
    "source_quality": {{
        "credible_sources": ["source1", "source2", ...],
        "questionable_sources": ["source3", "source4", ...],
        "source_diversity_assessment": "assessment of source diversity",
        "primary_vs_secondary_ratio": "assessment of primary vs secondary sources"
    }},
    "information_quality": [
        {{
            "topic": "topic area",
            "accuracy_assessment": "assessment of information accuracy",
            "precision_assessment": "assessment of information precision",
            "recency_assessment": "assessment of information timeliness",
            "relevance_assessment": "assessment of information relevance",
            "fact_vs_opinion_ratio": "assessment of fact vs opinion content",
            "quality_score": 0-100
        }}
    ],
    "methodological_quality": {{
        "sound_methodologies": ["methodology1", "methodology2", ...],
        "questionable_methodologies": ["methodology3", "methodology4", ...],
        "methodological_assessment": "overall assessment of methodologies used"
    }},
    "overall_quality_score": 0-100,
    "quality_strengths": ["strength1", "strength2", ...],
    "quality_weaknesses": ["weakness1", "weakness2", ...],
    "quality_improvement_recommendations": ["recommendation1", "recommendation2", ...]
}}
""")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        try:
            # Extract JSON from the response content
            quality_json = response.content
            if "```json" in quality_json:
                quality_json = quality_json.split("```json")[1].split("```")[0].strip()
            elif "```" in quality_json:
                quality_json = quality_json.split("```")[1].split("```")[0].strip()
                
            quality_evaluation = json.loads(quality_json)
            
            return {
                "status": "success",
                "critique_type": "quality",
                "quality_evaluation": quality_evaluation
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to parse quality evaluation: {str(e)}",
                "raw_evaluation": response.content
            }
    
    async def _check_consistency(self, research_findings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check for logical consistency across research findings.
        
        Args:
            research_findings: The research findings to check
            
        Returns:
            Consistency evaluation
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
Check for logical consistency across the following research findings.

RESEARCH FINDINGS:
{json.dumps(research_findings, indent=2)}

Assess:
1. Internal consistency within information from the same source
2. Cross-source consistency between different sources
3. Logical coherence of arguments and conclusions
4. Temporal consistency of information across different timeframes
5. Definitional consistency in how terms and concepts are used
6. Quantitative consistency in numerical information and statistics

Format your response as a JSON object with the following structure:
{{
    "internal_consistency_issues": [
        {{
            "source": "source identifier",
            "inconsistency": "description of internal inconsistency",
            "affected_elements": ["element1", "element2", ...],
            "severity": "high/medium/low",
            "resolution_approach": "suggested approach to resolve"
        }}
    ],
    "cross_source_inconsistencies": [
        {{
            "sources": ["source1", "source2"],
            "inconsistency": "description of cross-source inconsistency",
            "assessment": "analysis of inconsistency",
            "preferred_information": "which source appears more reliable and why",
            "resolution_approach": "suggested approach to resolve"
        }}
    ],
    "logical_coherence_issues": [
        {{
            "issue": "description of logical issue",
            "affected_reasoning": "specific reasoning with issue",
            "correction": "suggested correction"
        }}
    ],
    "temporal_inconsistencies": [
        {{
            "inconsistency": "description of temporal inconsistency",
            "affected_timeframes": ["timeframe1", "timeframe2"],
            "resolution": "suggested resolution"
        }}
    ],
    "definitional_inconsistencies": [
        {{
            "term": "inconsistently defined term",
            "different_definitions": ["definition1", "definition2"],
            "recommended_definition": "suggested consistent definition"
        }}
    ],
    "quantitative_inconsistencies": [
        {{
            "data_point": "inconsistent data point",
            "different_values": ["value1", "value2"],
            "assessment": "analysis of discrepancy",
            "recommended_value": "suggested correct value"
        }}
    ],
    "consistency_score": 0-100,
    "major_consistency_issues": ["issue1", "issue2", ...],
    "minor_consistency_issues": ["issue3", "issue4", ...],
    "overall_consistency_assessment": "comprehensive assessment of consistency"
}}
""")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        try:
            # Extract JSON from the response content
            consistency_json = response.content
            if "```json" in consistency_json:
                consistency_json = consistency_json.split("```json")[1].split("```")[0].strip()
            elif "```" in consistency_json:
                consistency_json = consistency_json.split("```")[1].split("```")[0].strip()
                
            consistency_evaluation = json.loads(consistency_json)
            
            return {
                "status": "success",
                "critique_type": "consistency",
                "consistency_evaluation": consistency_evaluation
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to parse consistency evaluation: {str(e)}",
                "raw_evaluation": response.content
            }
    
    async def _identify_bias(self, research_findings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify potential biases or limitations in research findings.
        
        Args:
            research_findings: The research findings to evaluate
            
        Returns:
            Bias and limitations evaluation
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
Identify potential biases and limitations in the following research findings.

RESEARCH FINDINGS:
{json.dumps(research_findings, indent=2)}

Assess:
1. Source biases (political, commercial, ideological, etc.)
2. Selection biases in information gathering
3. Confirmation biases in analysis and interpretation
4. Cultural or geographical biases in perspective
5. Recency or historical biases in temporal focus
6. Methodological limitations
7. Scope limitations of the research
8. Language or accessibility limitations

Format your response as a JSON object with the following structure:
{{
    "source_biases": [
        {{
            "source": "source identifier",
            "bias_type": "type of bias",
            "description": "description of bias",
            "impact": "how this bias affects findings",
            "mitigation_strategy": "how to mitigate this bias"
        }}
    ],
    "selection_biases": [
        {{
            "description": "description of selection bias",
            "affected_areas": ["area1", "area2", ...],
            "impact": "how this bias affects findings",
            "mitigation_strategy": "how to mitigate this bias"
        }}
    ],
    "confirmation_biases": [
        {{
            "description": "description of confirmation bias",
            "affected_conclusions": ["conclusion1", "conclusion2", ...],
            "alternative_interpretations": ["interpretation1", "interpretation2", ...],
            "mitigation_strategy": "how to mitigate this bias"
        }}
    ],
    "cultural_geographical_biases": [
        {{
            "bias": "description of cultural/geographical bias",
            "missing_perspectives": ["perspective1", "perspective2", ...],
            "mitigation_strategy": "how to mitigate this bias"
        }}
    ],
    "temporal_biases": [
        {{
            "bias": "description of temporal bias",
            "overlooked_timeframes": ["timeframe1", "timeframe2", ...],
            "impact": "how this affects findings",
            "mitigation_strategy": "how to mitigate this bias"
        }}
    ],
    "methodological_limitations": [
        {{
            "limitation": "description of methodological limitation",
            "impact": "how this affects findings",
            "alternative_approaches": ["approach1", "approach2", ...]
        }}
    ],
    "scope_limitations": [
        {{
            "limitation": "description of scope limitation",
            "excluded_areas": ["area1", "area2", ...],
            "impact": "how this affects completeness"
        }}
    ],
    "language_accessibility_limitations": [
        {{
            "limitation": "description of language/accessibility limitation",
            "impact": "how this affects findings",
            "mitigation_strategy": "how to address this limitation"
        }}
    ],
    "overall_bias_assessment": "comprehensive assessment of biases",
    "bias_impact_score": 0-100,
    "priority_mitigation_strategies": ["strategy1", "strategy2", ...]
}}
""")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        try:
            # Extract JSON from the response content
            bias_json = response.content
            if "```json" in bias_json:
                bias_json = bias_json.split("```json")[1].split("```")[0].strip()
            elif "```" in bias_json:
                bias_json = bias_json.split("```")[1].split("```")[0].strip()
                
            bias_evaluation = json.loads(bias_json)
            
            return {
                "status": "success",
                "critique_type": "bias",
                "bias_evaluation": bias_evaluation
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to parse bias evaluation: {str(e)}",
                "raw_evaluation": response.content
            }
    
    async def _comprehensive_critique(
        self, research_findings: Dict[str, Any], research_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform a comprehensive critique covering all evaluation aspects.
        
        Args:
            research_findings: The research findings to evaluate
            research_plan: The original research plan
            
        Returns:
            Comprehensive critique results
        """
        # Perform individual critiques
        completeness_result = await self._evaluate_completeness(research_findings, research_plan)
        quality_result = await self._evaluate_quality(research_findings)
        consistency_result = await self._check_consistency(research_findings)
        bias_result = await self._identify_bias(research_findings)
        
        # Combine results
        comprehensive_critique = {
            "status": "success",
            "critique_type": "comprehensive",
            "completeness": completeness_result.get("completeness_evaluation", {}),
            "quality": quality_result.get("quality_evaluation", {}),
            "consistency": consistency_result.get("consistency_evaluation", {}),
            "bias": bias_result.get("bias_evaluation", {})
        }
        
        # Generate overall assessment
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
Synthesize the following critique results into an overall assessment.

COMPLETENESS EVALUATION:
{json.dumps(comprehensive_critique["completeness"], indent=2)}

QUALITY EVALUATION:
{json.dumps(comprehensive_critique["quality"], indent=2)}

CONSISTENCY EVALUATION:
{json.dumps(comprehensive_critique["consistency"], indent=2)}

BIAS EVALUATION:
{json.dumps(comprehensive_critique["bias"], indent=2)}

Provide a comprehensive synthesis that:
1. Identifies the most critical issues across all evaluation dimensions
2. Highlights key strengths of the research
3. Suggests prioritized improvements for the research
4. Provides an overall quality assessment

Format your response as a JSON object with the following structure:
{{
    "critical_issues": [
        {{
            "issue": "description of critical issue",
            "category": "completeness/quality/consistency/bias",
            "severity": "high/medium/low",
            "impact": "impact on research value",
            "priority": 1-10
        }}
    ],
    "key_strengths": [
        {{
            "strength": "description of research strength",
            "category": "completeness/quality/consistency/bias",
            "value": "contribution to research value"
        }}
    ],
    "prioritized_improvements": [
        {{
            "improvement": "suggested improvement",
            "addresses": ["issue1", "issue2", ...],
            "implementation_approach": "how to implement",
            "priority": 1-10
        }}
    ],
    "overall_assessment": {{
        "quality_score": 0-100,
        "research_value": "high/medium/low",
        "confidence_in_findings": "high/medium/low",
        "summary": "comprehensive assessment of research quality"
    }}
}}
""")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        try:
            # Extract JSON from the response content
            synthesis_json = response.content
            if "```json" in synthesis_json:
                synthesis_json = synthesis_json.split("```json")[1].split("```")[0].strip()
            elif "```" in synthesis_json:
                synthesis_json = synthesis_json.split("```")[1].split("```")[0].strip()
                
            overall_assessment = json.loads(synthesis_json)
            
            comprehensive_critique["overall_assessment"] = overall_assessment
            return comprehensive_critique
        except Exception as e:
            comprehensive_critique["overall_assessment_error"] = str(e)
            comprehensive_critique["raw_overall_assessment"] = response.content
            return comprehensive_critique