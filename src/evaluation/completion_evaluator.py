"""
Completion Evaluator Module

This module provides the CompletionEvaluator class which evaluates research
results against completion criteria to determine when a task is adequately complete.
"""

from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import time
import json
import logging
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from .completion_criteria import CompletionCriteria, CriteriaLevel
from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class CompletionEvaluator(BaseEvaluator):
    """
    CompletionEvaluator evaluates research results against completion criteria.
    
    This class uses a combination of quantitative metrics, qualitative assessments,
    and LLM-based evaluation to determine when a research task is adequately complete.
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        criteria: Optional[CompletionCriteria] = None,
        task_type: str = "general",
        task_complexity: str = "medium",
        name: str = "CompletionEvaluator",
    ):
        """
        Initialize a CompletionEvaluator instance.
        
        Args:
            llm: Language model for qualitative assessments
            criteria: Optional completion criteria (will create default if None)
            task_type: Type of research task (for default criteria)
            task_complexity: Complexity of research task (for default criteria)
            name: Evaluator name
        """
        super().__init__(name=name)
        self.llm = llm
        
        # Create default criteria if none provided
        if criteria:
            self.criteria = criteria
        else:
            self.criteria = CompletionCriteria.create_task_specific(
                task_type=task_type,
                complexity=task_complexity
            )
        
        # Track evaluation history
        self.evaluation_history = []
        
        # Define LLM prompt templates
        self.quality_assessment_template = """You are a research quality evaluator with expertise in assessing the thoroughness and completeness of research. 
Your task is to evaluate the research results for the following query and determine if the research is complete.

RESEARCH QUERY: {query}

RESEARCH RESULTS:
{results}

EVALUATION CRITERIA:
- Source Diversity: At least {min_sources} diverse sources should be consulted.
- Depth of Coverage: Key topics should be explored in depth (score >= {min_depth}).
- Breadth of Coverage: At least {min_breadth}% of relevant subtopics should be addressed.
- Cross-Verification: At least {min_verification} of key facts should be verified across multiple sources.
- Strategic Objective Coverage: Research should address {min_objectives} of the strategic objectives.

Please evaluate the research results based on these criteria and provide a detailed assessment with scores for each criterion. 
Then conclude with an overall evaluation of whether the research is COMPLETE or INCOMPLETE, and specific recommendations for improvement if needed.
"""

    async def evaluate(self, research_results: Dict[str, Any], research_query: str) -> Dict[str, Any]:
        """
        Evaluate research results against completion criteria.
        
        Args:
            research_results: Dictionary of research results
            research_query: The original research query
            
        Returns:
            Evaluation results with completion status and recommendations
        """
        # 1. Compute quantitative metrics
        metrics = await self._compute_metrics(research_results)
        
        # 2. Evaluate against criteria
        criteria_evaluation = self.criteria.evaluate({**research_results, **metrics})
        
        # 3. Perform qualitative assessment using LLM
        qualitative_assessment = await self._perform_qualitative_assessment(
            research_results, 
            research_query
        )
        
        # 4. Determine overall completion status
        # - Criteria-based evaluation (quantitative)
        criteria_complete = criteria_evaluation["overall_complete"]
        # - LLM-based evaluation (qualitative)
        llm_complete = qualitative_assessment["overall_complete"]
        
        # 5. Reconcile evaluations and make final determination
        overall_complete, completion_confidence, reasoning = self._reconcile_evaluations(
            criteria_complete, 
            llm_complete,
            criteria_evaluation,
            qualitative_assessment
        )
        
        # 6. Create comprehensive evaluation result
        evaluation_result = {
            "timestamp": time.time(),
            "research_query": research_query,
            "metrics": metrics,
            "criteria_evaluation": criteria_evaluation,
            "qualitative_assessment": qualitative_assessment,
            "overall_complete": overall_complete,
            "completion_confidence": completion_confidence,
            "reasoning": reasoning,
            "recommendations": self._generate_recommendations(
                criteria_evaluation, 
                qualitative_assessment,
                overall_complete
            )
        }
        
        # Add to evaluation history
        self.evaluation_history.append(evaluation_result)
        
        return evaluation_result
    
    async def _compute_metrics(self, research_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute quantitative metrics from research results.
        
        Args:
            research_results: Dictionary of research results
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        # Source metrics
        sources = research_results.get("sources", [])
        metrics["sources_count"] = len(sources)
        
        # Calculate source diversity score based on domain and type distribution
        source_domains = {}
        source_types = {}
        
        for source in sources:
            # Extract domain from URL
            domain = source.get("domain", "unknown")
            if domain in source_domains:
                source_domains[domain] += 1
            else:
                source_domains[domain] = 1
                
            # Count source types
            source_type = source.get("type", "unknown")
            if source_type in source_types:
                source_types[source_type] += 1
            else:
                source_types[source_type] = 1
        
        # Calculate diversity scores (normalized entropy-based metric)
        metrics["source_diversity_score"] = self._calculate_diversity_score(source_domains, source_types)
        
        # Coverage metrics
        subtopics = research_results.get("subtopics", [])
        total_subtopics = len(subtopics)
        covered_subtopics = sum(1 for topic in subtopics if topic.get("covered", False))
        
        if total_subtopics > 0:
            metrics["breadth_coverage_percentage"] = (covered_subtopics / total_subtopics) * 100
        else:
            metrics["breadth_coverage_percentage"] = 0
        
        # Depth coverage score (avg of subtopic depth scores)
        if total_subtopics > 0:
            depth_scores = [topic.get("depth_score", 0) for topic in subtopics]
            metrics["depth_coverage_score"] = sum(depth_scores) / total_subtopics
        else:
            metrics["depth_coverage_score"] = 0
        
        # Cross-verification rate
        facts = research_results.get("facts", [])
        if facts:
            verified_facts = sum(1 for fact in facts if fact.get("verified_sources", 0) >= 2)
            metrics["cross_verification_rate"] = verified_facts / len(facts)
        else:
            metrics["cross_verification_rate"] = 0
        
        # Strategic objective coverage
        objectives = research_results.get("objectives", [])
        if objectives:
            achieved_objectives = sum(1 for obj in objectives if obj.get("achieved", False))
            metrics["strategic_objective_coverage"] = achieved_objectives / len(objectives)
        else:
            metrics["strategic_objective_coverage"] = 0
        
        # Time allocation
        steps = research_results.get("steps", [])
        if steps:
            # Calculate time spent in different phases
            exploration_time = sum(step.get("duration", 0) for step in steps if step.get("phase") == "exploration")
            deep_investigation_time = sum(step.get("duration", 0) for step in steps if step.get("phase") == "deep_investigation")
            analysis_time = sum(step.get("duration", 0) for step in steps if step.get("phase") == "analysis")
            
            total_time = sum(step.get("duration", 0) for step in steps)
            
            if total_time > 0:
                metrics["time_allocation"] = {
                    "exploration": (exploration_time / total_time) * 100,
                    "deep_investigation": (deep_investigation_time / total_time) * 100,
                    "analysis": (analysis_time / total_time) * 100
                }
                metrics["research_time_minutes"] = total_time / 60  # Convert seconds to minutes
            else:
                metrics["time_allocation"] = {
                    "exploration": 0,
                    "deep_investigation": 0,
                    "analysis": 0
                }
                metrics["research_time_minutes"] = 0
        else:
            metrics["time_allocation"] = {
                "exploration": 0,
                "deep_investigation": 0,
                "analysis": 0
            }
            metrics["research_time_minutes"] = 0
        
        # Content type distribution
        content = research_results.get("content", [])
        content_types = {"factual": 0, "conceptual": 0, "analytical": 0, "opinion": 0}
        
        if content:
            for item in content:
                item_type = item.get("type", "unknown")
                if item_type in content_types:
                    content_types[item_type] += 1
            
            # Normalize to percentages
            total_content = sum(content_types.values())
            if total_content > 0:
                for content_type in content_types:
                    content_types[content_type] = content_types[content_type] / total_content
                    
            metrics["content_type_distribution"] = content_types
        else:
            metrics["content_type_distribution"] = content_types
        
        return metrics
    
    def _calculate_diversity_score(self, domains: Dict[str, int], types: Dict[str, int]) -> float:
        """
        Calculate diversity score based on entropy of source distribution.
        
        Args:
            domains: Dictionary of domain counts
            types: Dictionary of source type counts
            
        Returns:
            Diversity score between 0 and 1
        """
        import math
        
        # Calculate domain entropy
        domain_entropy = 0
        total_domains = sum(domains.values())
        
        if total_domains > 0:
            for count in domains.values():
                p = count / total_domains
                domain_entropy -= p * math.log2(p)
                
            # Normalize by maximum possible entropy (log2 of unique domains)
            max_domain_entropy = math.log2(len(domains)) if len(domains) > 0 else 1
            domain_entropy = domain_entropy / max_domain_entropy if max_domain_entropy > 0 else 0
        
        # Calculate type entropy
        type_entropy = 0
        total_types = sum(types.values())
        
        if total_types > 0:
            for count in types.values():
                p = count / total_types
                type_entropy -= p * math.log2(p)
                
            # Normalize by maximum possible entropy (log2 of unique types)
            max_type_entropy = math.log2(len(types)) if len(types) > 0 else 1
            type_entropy = type_entropy / max_type_entropy if max_type_entropy > 0 else 0
        
        # Combine domain and type entropy (weighted average)
        diversity_score = (0.7 * domain_entropy) + (0.3 * type_entropy)
        
        return diversity_score
    
    async def _perform_qualitative_assessment(
        self, 
        research_results: Dict[str, Any], 
        research_query: str
    ) -> Dict[str, Any]:
        """
        Perform qualitative assessment of research results using LLM.
        
        Args:
            research_results: Dictionary of research results
            research_query: The original research query
            
        Returns:
            Qualitative assessment results
        """
        # Format research results for LLM
        formatted_results = self._format_results_for_llm(research_results)
        
        # Create prompt with criteria from this evaluator
        prompt = self.quality_assessment_template.format(
            query=research_query,
            results=formatted_results,
            min_sources=self.criteria.min_sources,
            min_depth=self.criteria.min_depth_coverage_score,
            min_breadth=self.criteria.min_breadth_coverage_percentage,
            min_verification=self.criteria.min_cross_verification_rate,
            min_objectives=self.criteria.min_strategic_objective_coverage
        )
        
        # Query LLM for assessment
        messages = [
            SystemMessage(content="You are a research quality evaluator that assesses the completeness and quality of research results."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            assessment_text = response.content
            
            # Parse assessment to extract scores and completion determination
            assessment_results = self._parse_llm_assessment(assessment_text)
            
            return assessment_results
        except Exception as e:
            logger.error(f"Error in LLM qualitative assessment: {str(e)}")
            # Return default assessment on error
            return {
                "overall_complete": False,
                "reasoning": f"Failed to perform qualitative assessment: {str(e)}",
                "criterion_scores": {},
                "assessment_text": ""
            }
    
    def _format_results_for_llm(self, research_results: Dict[str, Any]) -> str:
        """
        Format research results into a readable text format for LLM assessment.
        
        Args:
            research_results: Dictionary of research results
            
        Returns:
            Formatted results string
        """
        # Extract key information
        sources = research_results.get("sources", [])
        content = research_results.get("content", [])
        extracted_info = research_results.get("extracted_information", "")
        findings = research_results.get("findings", [])
        
        # Format sources
        sources_text = "SOURCES CONSULTED:\n"
        for i, source in enumerate(sources[:10], 1):  # Limit to 10 sources to avoid token overload
            title = source.get("title", "Untitled")
            url = source.get("url", "No URL")
            source_type = source.get("type", "Unknown")
            
            sources_text += f"{i}. {title} ({source_type}) - {url}\n"
            
        if len(sources) > 10:
            sources_text += f"... and {len(sources) - 10} more sources.\n"
        
        # Format findings/content
        content_text = "KEY FINDINGS AND INFORMATION:\n"
        
        # Use findings if available
        if findings:
            for i, finding in enumerate(findings, 1):
                content_text += f"{i}. {finding}\n"
        # Otherwise use content items
        elif content:
            for i, item in enumerate(content, 1):
                item_type = item.get("type", "information")
                item_text = item.get("text", "")
                content_text += f"{i}. [{item_type}] {item_text}\n"
        # Fall back to extracted information
        elif extracted_info:
            content_text += extracted_info[:2000] + "...\n"  # Truncate if very long
        else:
            content_text += "No specific findings or content available.\n"
        
        # Combine all sections
        formatted_text = f"{sources_text}\n{content_text}"
        
        return formatted_text
    
    def _parse_llm_assessment(self, assessment_text: str) -> Dict[str, Any]:
        """
        Parse the LLM's qualitative assessment into structured data.
        
        Args:
            assessment_text: Assessment text from LLM
            
        Returns:
            Structured assessment results
        """
        # Default values
        assessment_results = {
            "overall_complete": False,
            "reasoning": "",
            "criterion_scores": {},
            "assessment_text": assessment_text
        }
        
        # Check for explicit completion determination
        assessment_text_lower = assessment_text.lower()
        if "complete" in assessment_text_lower and "incomplete" not in assessment_text_lower:
            assessment_results["overall_complete"] = True
        elif "incomplete" in assessment_text_lower:
            assessment_results["overall_complete"] = False
        
        # Try to extract scores for individual criteria
        try:
            # Look for scores in the format "Criterion: X/10" or "Criterion Score: X"
            import re
            
            # Source diversity
            source_match = re.search(r"source diversity[^0-9]*([0-9.]+)", assessment_text_lower)
            if source_match:
                assessment_results["criterion_scores"]["source_diversity"] = float(source_match.group(1))
            
            # Depth coverage
            depth_match = re.search(r"depth[^0-9]*([0-9.]+)", assessment_text_lower)
            if depth_match:
                assessment_results["criterion_scores"]["depth_coverage"] = float(depth_match.group(1))
            
            # Breadth coverage
            breadth_match = re.search(r"breadth[^0-9]*([0-9.]+)", assessment_text_lower)
            if breadth_match:
                assessment_results["criterion_scores"]["breadth_coverage"] = float(breadth_match.group(1))
            
            # Cross-verification
            verify_match = re.search(r"cross.?verification[^0-9]*([0-9.]+)", assessment_text_lower)
            if verify_match:
                assessment_results["criterion_scores"]["cross_verification"] = float(verify_match.group(1))
            
            # Strategic objectives
            objective_match = re.search(r"objective[^0-9]*([0-9.]+)", assessment_text_lower)
            if objective_match:
                assessment_results["criterion_scores"]["objective_coverage"] = float(objective_match.group(1))
        
        except Exception as e:
            logger.warning(f"Error parsing LLM assessment scores: {str(e)}")
        
        # Extract reasoning
        try:
            # Look for conclusion or recommendation sections
            conclusion_match = re.search(r"(?:conclusion|overall|summary)(?::|.{0,50})\s*(.+?)(?:\n\n|\Z)", assessment_text, re.IGNORECASE | re.DOTALL)
            if conclusion_match:
                assessment_results["reasoning"] = conclusion_match.group(1).strip()
            else:
                # Take the last paragraph as reasoning
                paragraphs = [p for p in assessment_text.split("\n\n") if p.strip()]
                if paragraphs:
                    assessment_results["reasoning"] = paragraphs[-1].strip()
        except Exception as e:
            logger.warning(f"Error extracting reasoning from LLM assessment: {str(e)}")
            assessment_results["reasoning"] = "Unable to extract specific reasoning from assessment."
        
        return assessment_results
    
    def _reconcile_evaluations(
        self,
        criteria_complete: bool,
        llm_complete: bool,
        criteria_evaluation: Dict[str, Any],
        qualitative_assessment: Dict[str, Any]
    ) -> Tuple[bool, float, str]:
        """
        Reconcile quantitative and qualitative evaluations to make a final determination.
        
        Args:
            criteria_complete: Whether criteria-based evaluation indicates completeness
            llm_complete: Whether LLM-based evaluation indicates completeness
            criteria_evaluation: Full criteria evaluation results
            qualitative_assessment: Full qualitative assessment results
            
        Returns:
            Tuple of (overall complete status, confidence score, reasoning)
        """
        # Start with the criteria-based evaluation as primary
        overall_complete = criteria_complete
        
        # Calculate confidence score based on agreement between evaluations
        if criteria_complete == llm_complete:
            # High confidence when both evaluations agree
            confidence = 0.9
            reasoning_prefix = "Both quantitative and qualitative evaluations agree: "
        else:
            # Lower confidence when evaluations disagree
            confidence = 0.6
            reasoning_prefix = "Evaluations disagree. "
            
            # Determine if we should override the criteria-based decision
            if criteria_complete and not llm_complete:
                # LLM says incomplete despite criteria being met
                if "required_criteria_met" in criteria_evaluation and criteria_evaluation["required_criteria_met"]:
                    # All required criteria met, but LLM still says incomplete
                    reasoning_prefix += "All required criteria are met, but qualitative assessment suggests incompleteness. "
                    # Lower confidence but maintain criteria-based decision
                    confidence = 0.7
                else:
                    # Not all required criteria met, defer to LLM's judgment
                    overall_complete = False
                    reasoning_prefix += "Not all required criteria are fully met, deferring to qualitative assessment. "
            elif not criteria_complete and llm_complete:
                # LLM says complete despite criteria not being met
                if "required_criteria_met" in criteria_evaluation and criteria_evaluation["required_criteria_met"]:
                    # All required criteria met but some important ones aren't
                    overall_complete = True
                    reasoning_prefix += "All required criteria are met, qualitative assessment indicates sufficient completeness. "
                    confidence = 0.7
                else:
                    # Not all required criteria met, maintain criteria-based decision
                    overall_complete = False
                    reasoning_prefix += "Not all required criteria are met, despite positive qualitative assessment. "
        
        # Combine reasoning from both evaluations
        criteria_reasoning = criteria_evaluation.get("recommendation", "")
        llm_reasoning = qualitative_assessment.get("reasoning", "")
        
        combined_reasoning = reasoning_prefix
        if criteria_reasoning:
            combined_reasoning += f"Criteria evaluation: {criteria_reasoning} "
        if llm_reasoning:
            combined_reasoning += f"Qualitative assessment: {llm_reasoning}"
        
        return overall_complete, confidence, combined_reasoning
    
    def _generate_recommendations(
        self,
        criteria_evaluation: Dict[str, Any],
        qualitative_assessment: Dict[str, Any],
        overall_complete: bool
    ) -> List[str]:
        """
        Generate recommendations for improving the research.
        
        Args:
            criteria_evaluation: Criteria-based evaluation results
            qualitative_assessment: Qualitative assessment results
            overall_complete: Overall completion status
            
        Returns:
            List of improvement recommendations
        """
        if overall_complete:
            return []  # No recommendations needed if research is complete
        
        # Start with improvements from criteria evaluation
        recommendations = criteria_evaluation.get("improvements_needed", [])
        
        # Add qualitative recommendations if available
        qualitative_reasoning = qualitative_assessment.get("reasoning", "")
        
        # Extract recommendation-like statements from qualitative reasoning
        import re
        recommendation_patterns = [
            r"should ([^.;!]+)[.;!]",
            r"need[s]? to ([^.;!]+)[.;!]",
            r"recommend[s]? ([^.;!]+)[.;!]",
            r"consider ([^.;!]+)[.;!]",
            r"improve ([^.;!]+)[.;!]"
        ]
        
        for pattern in recommendation_patterns:
            matches = re.findall(pattern, qualitative_reasoning, re.IGNORECASE)
            for match in matches:
                recommendation = match.strip().capitalize()
                if recommendation and recommendation not in recommendations:
                    recommendations.append(recommendation)
        
        return recommendations
    
    def get_criteria_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the completion criteria used by this evaluator.
        
        Returns:
            Dictionary summarizing the criteria
        """
        return self.criteria.get_criteria_summary()
    
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of evaluations performed by this evaluator.
        
        Returns:
            List of evaluation results
        """
        return self.evaluation_history