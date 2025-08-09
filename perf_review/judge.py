"""
LLM judge for pairwise comparisons of performance reviews.
"""

import json
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional
from .openai_utils import OpenAIClient
from .models import Submission, Match


@dataclass
class JudgeResult:
    """Result from a single judge."""

    winner: str  # "A" or "B"
    reasoning: str
    confidence: float
    judge_id: str
    judge_model: str


@dataclass
class MultiJudgeResult:
    """Result from multiple judges with consensus."""

    winner: str
    consensus_reasoning: str
    individual_results: List[JudgeResult]
    confidence: float
    is_unanimous: bool
    vote_distribution: Dict[str, int]


class LLMJudge:
    """LLM-powered judge for comparing performance reviews."""

    def __init__(
        self,
        openai_client: OpenAIClient,
        model: str = "gpt-4.1",
        judge_id: Optional[str] = None,
        reasoning: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the judge.

        Args:
            openai_client: OpenAI client instance
            model: Model to use for judging
            judge_id: Optional identifier for this judge
            reasoning: Reasoning configuration for GPT-5 models
        """
        self.client = openai_client
        self.model = model
        self.judge_id = judge_id or f"judge_{model.replace('.', '_')}"
        self.reasoning = reasoning

    def judge_single(
        self, submission_a: Submission, submission_b: Submission, rubric: str
    ) -> JudgeResult:
        """
        Get a single judge's decision on two submissions.

        Args:
            submission_a: First submission
            submission_b: Second submission
            rubric: Evaluation criteria

        Returns:
            JudgeResult with this judge's decision
        """
        # Create structured prompt for comparison
        system_prompt = self._create_judge_system_prompt(rubric)
        user_prompt = self._create_comparison_prompt(submission_a, submission_b)

        # Define function for structured output (reasoning first)
        tools = [
            {
                "type": "function",
                "name": "judge_comparison",
                "description": "Judge which performance review is better based on the rubric",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "Detailed reasoning for why this submission is better (provide this first)",
                        },
                        "winner": {
                            "type": "string",
                            "enum": ["A", "B"],
                            "description": "Which submission is better (A or B)",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence in the judgment (0-1)",
                        },
                    },
                    "required": ["reasoning", "winner", "confidence"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ]

        # Get response
        messages = self.client.create_structured_prompt(
            system_message=system_prompt, user_message=user_prompt
        )

        response = self.client.create_response(
            input_messages=messages,
            tools=tools,
            tool_choice={"type": "function", "name": "judge_comparison"},
            reasoning=self.reasoning,
        )

        # Extract function call result
        function_calls = self.client.extract_function_calls(response)

        if not function_calls:
            raise ValueError("Judge did not provide structured comparison result")

        result = function_calls[0]["arguments"]

        return JudgeResult(
            winner=result["winner"],
            reasoning=result["reasoning"],
            confidence=result["confidence"],
            judge_id=self.judge_id,
            judge_model=self.model,
        )

    def compare_submissions(
        self, submission_a: Submission, submission_b: Submission, rubric: str
    ) -> Match:
        """
        Compare two submissions using single judge (legacy compatibility).

        Args:
            submission_a: First submission
            submission_b: Second submission
            rubric: Evaluation criteria/career ladder

        Returns:
            Match object with winner and reasoning
        """
        judge_result = self.judge_single(submission_a, submission_b, rubric)

        # Determine winner ID
        winner_id = submission_a.id if judge_result.winner == "A" else submission_b.id

        # Create match object
        match = Match(
            id=str(uuid.uuid4()),
            submission_a_id=submission_a.id,
            submission_b_id=submission_b.id,
            winner_id=winner_id,
            judge_reasoning=judge_result.reasoning,
            metadata={
                "confidence": judge_result.confidence,
                "judge_model": self.model,
                "single_judge": True,
            },
        )

        return match

    def _create_judge_system_prompt(self, rubric: str) -> str:
        """Create system prompt for the judge."""
        return f"""You are an expert performance review evaluator. Your job is to compare two performance reviews and determine which one is better based on the provided rubric.

EVALUATION RUBRIC:
{rubric}

EVALUATION CRITERIA:
1. Clarity and specificity of accomplishments
2. Quantifiable impact and results
3. Alignment with career ladder/rubric requirements
4. Professional presentation and structure
5. Evidence of growth and learning

You must be objective and base your judgment solely on how well each review meets the rubric criteria. 

IMPORTANT: Provide your detailed reasoning FIRST, then make your decision. This helps ensure thorough analysis.

Use the judge_comparison function to provide your structured judgment."""

    def _create_comparison_prompt(
        self, submission_a: Submission, submission_b: Submission
    ) -> str:
        """Create user prompt comparing two submissions."""
        return f"""Please compare these two performance reviews and determine which is better:

SUBMISSION A:
{submission_a.content}

SUBMISSION B:
{submission_b.content}

Evaluate both submissions against the rubric and provide:
1. Which submission is better (A or B)
2. Detailed reasoning explaining your decision
3. Confidence level in your judgment (0-1)

Focus on objective criteria from the rubric rather than subjective preferences."""

    def batch_compare(
        self, submissions: list[Submission], rubric: str, max_comparisons: int = None
    ) -> list[Match]:
        """
        Run multiple pairwise comparisons.

        Args:
            submissions: List of submissions to compare
            rubric: Evaluation rubric
            max_comparisons: Maximum number of comparisons to run

        Returns:
            List of match results
        """
        matches = []
        comparison_count = 0

        for i in range(len(submissions)):
            for j in range(i + 1, len(submissions)):
                if max_comparisons and comparison_count >= max_comparisons:
                    break

                match = self.compare_submissions(submissions[i], submissions[j], rubric)
                matches.append(match)
                comparison_count += 1

        return matches


class MultiJudge:
    """Multi-judge system with majority voting."""

    def __init__(self, judges: List[LLMJudge]):
        """
        Initialize multi-judge system.

        Args:
            judges: List of individual judges
        """
        self.judges = judges
        if len(self.judges) % 2 == 0:
            raise ValueError("Number of judges must be odd to avoid ties")

    def compare_submissions(
        self, submission_a: Submission, submission_b: Submission, rubric: str
    ) -> Match:
        """
        Compare submissions using multiple judges with majority vote.

        Args:
            submission_a: First submission
            submission_b: Second submission
            rubric: Evaluation criteria

        Returns:
            Match object with consensus result
        """
        # Get results from all judges
        individual_results = []
        for judge in self.judges:
            result = judge.judge_single(submission_a, submission_b, rubric)
            individual_results.append(result)

        # Count votes
        vote_counts = Counter([r.winner for r in individual_results])
        winner = vote_counts.most_common(1)[0][0]
        is_unanimous = len(vote_counts) == 1

        # Calculate consensus confidence (average of winning votes)
        winning_results = [r for r in individual_results if r.winner == winner]
        avg_confidence = sum(r.confidence for r in winning_results) / len(
            winning_results
        )

        # Create consensus reasoning
        consensus_reasoning = self._create_consensus_reasoning(
            individual_results, winner, is_unanimous
        )

        # Store multi-judge result for potential future use
        # multi_result = MultiJudgeResult(
        #     winner=winner,
        #     consensus_reasoning=consensus_reasoning,
        #     individual_results=individual_results,
        #     confidence=avg_confidence,
        #     is_unanimous=is_unanimous,
        #     vote_distribution=dict(vote_counts)
        # )

        # Determine winner ID
        winner_id = submission_a.id if winner == "A" else submission_b.id

        # Create match with enhanced metadata
        match = Match(
            id=str(uuid.uuid4()),
            submission_a_id=submission_a.id,
            submission_b_id=submission_b.id,
            winner_id=winner_id,
            judge_reasoning=consensus_reasoning,
            metadata={
                "multi_judge": True,
                "consensus_confidence": avg_confidence,
                "is_unanimous": is_unanimous,
                "vote_distribution": dict(vote_counts),
                "individual_judgments": [
                    {
                        "judge_id": r.judge_id,
                        "judge_model": r.judge_model,
                        "winner": r.winner,
                        "confidence": r.confidence,
                        "reasoning": r.reasoning,
                    }
                    for r in individual_results
                ],
            },
        )

        return match

    def _create_consensus_reasoning(
        self, results: List[JudgeResult], winner: str, is_unanimous: bool
    ) -> str:
        """Create consensus reasoning from individual judge results."""

        winning_reasons = [r.reasoning for r in results if r.winner == winner]
        losing_reasons = [r.reasoning for r in results if r.winner != winner]

        consensus = f"**Multi-Judge Consensus (Winner: {winner})**\\n\\n"

        if is_unanimous:
            consensus += "**UNANIMOUS DECISION**\\n\\n"
        else:
            vote_breakdown = Counter([r.winner for r in results])
            consensus += f"**MAJORITY DECISION** ({dict(vote_breakdown)})\\n\\n"

        consensus += "**Winning Arguments:**\\n"
        for i, reason in enumerate(winning_reasons, 1):
            consensus += f"{i}. {reason}\\n\\n"

        if losing_reasons:
            consensus += "**Dissenting Arguments:**\\n"
            for i, reason in enumerate(losing_reasons, 1):
                consensus += f"{i}. {reason}\\n\\n"

        return consensus
