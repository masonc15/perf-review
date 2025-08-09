"""
LLM agents for optimizing performance reviews.
"""

import json
import uuid
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from .openai_utils import OpenAIClient
from .models import Submission


@dataclass
class OptimizationStrategy:
    """Configuration for an optimization strategy."""

    name: str
    description: str
    focus_areas: List[str]
    model: Optional[str] = None  # If None, uses default
    max_attempts: int = 2
    reasoning: Optional[Dict[str, Any]] = None  # GPT-5 reasoning config


@dataclass
class OptimizationFeedback:
    """Feedback from a previous optimization attempt."""

    submission_id: str
    content: str
    performance_score: float  # From ELO or judge feedback
    judge_feedback: Optional[str] = None
    suggestions: List[str] = None


class OptimizerAgent:
    """LLM agent that optimizes performance reviews with configurable strategies."""

    def __init__(
        self,
        strategy: OptimizationStrategy,
        openai_client: OpenAIClient,
        default_model: str = "gpt-4.1",
    ):
        """
        Initialize optimizer agent.

        Args:
            strategy: Optimization strategy configuration
            openai_client: OpenAI client instance
            default_model: Default model if not specified in strategy
        """
        self.strategy = strategy
        self.client = openai_client
        self.model = strategy.model or default_model
        self.attempts_made = 0
        self.feedback_history: List[OptimizationFeedback] = []

    @property
    def name(self) -> str:
        """Get agent name from strategy."""
        return self.strategy.name

    def add_feedback(self, feedback: OptimizationFeedback) -> None:
        """Add feedback from previous attempts."""
        self.feedback_history.append(feedback)

    def optimize_submission(
        self,
        original_submission: Submission,
        rubric: str,
        previous_attempts: Optional[List[Submission]] = None,
        use_feedback: bool = True,
    ) -> Submission:
        """
        Create an optimized version of a submission.

        Args:
            original_submission: Submission to optimize
            rubric: Career ladder/evaluation criteria
            previous_attempts: Previous optimization attempts to learn from
            use_feedback: Whether to use feedback from previous attempts

        Returns:
            New optimized submission
        """
        self.attempts_made += 1

        # Check attempt limits
        if self.attempts_made > self.strategy.max_attempts:
            raise ValueError(
                f"Agent {self.name} has exceeded max attempts ({self.strategy.max_attempts})"
            )

        # Create optimization prompt
        system_prompt = self._create_optimizer_system_prompt(rubric)
        user_prompt = self._create_optimization_prompt(
            original_submission, previous_attempts, use_feedback
        )

        # Define function for structured optimization with feedback request
        tools = [
            {
                "type": "function",
                "name": "optimize_review",
                "description": "Optimize a performance review based on the rubric and strategy",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "optimized_content": {
                            "type": "string",
                            "description": "The improved performance review content",
                        },
                        "improvements_made": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of specific improvements made",
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Explanation of optimization strategy and reasoning",
                        },
                        "confidence_level": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Agent's confidence in this optimization (0-1)",
                        },
                        "areas_needing_feedback": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific areas where the agent would benefit from judge feedback",
                        },
                    },
                    "required": [
                        "optimized_content",
                        "improvements_made",
                        "rationale",
                        "confidence_level",
                        "areas_needing_feedback",
                    ],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ]

        # Get optimization response
        messages = self.client.create_structured_prompt(
            system_message=system_prompt, user_message=user_prompt
        )

        # Get reasoning config from strategy if available
        reasoning_config = getattr(self.strategy, "reasoning", None)

        response = self.client.create_response(
            input_messages=messages,
            tools=tools,
            tool_choice={"type": "function", "name": "optimize_review"},
            reasoning=reasoning_config,
        )

        # Extract optimization result
        function_calls = self.client.extract_function_calls(response)

        if not function_calls:
            raise ValueError("Optimizer did not provide structured optimization result")

        result = function_calls[0]["arguments"]

        # Create optimized submission
        optimized_submission = Submission(
            id=str(uuid.uuid4()),
            content=result["optimized_content"],
            parent_id=original_submission.id,
            agent_name=self.name,
            metadata={
                "improvements_made": result["improvements_made"],
                "rationale": result["rationale"],
                "confidence_level": result["confidence_level"],
                "areas_needing_feedback": result["areas_needing_feedback"],
                "strategy": self.strategy.description,
                "strategy_config": {
                    "name": self.strategy.name,
                    "focus_areas": self.strategy.focus_areas,
                    "max_attempts": self.strategy.max_attempts,
                },
                "attempt_number": self.attempts_made,
                "optimizer_model": self.model,
                "used_feedback": use_feedback and len(self.feedback_history) > 0,
                "feedback_count": len(self.feedback_history),
            },
        )

        return optimized_submission

    def optimize_with_feedback_loop(
        self,
        original_submission: Submission,
        rubric: str,
        judge: "LLMJudge",
        max_iterations: int = 3,
        min_confidence_threshold: float = 0.8,
    ) -> Submission:
        """
        Optimize submission with multi-turn feedback loop.

        Args:
            original_submission: Submission to optimize
            rubric: Career ladder/evaluation criteria
            judge: Judge to provide feedback
            max_iterations: Maximum optimization iterations
            min_confidence_threshold: Stop if agent confidence exceeds this

        Returns:
            Final optimized submission
        """
        current_submission = original_submission
        best_submission = original_submission
        best_score = 0.0

        for iteration in range(max_iterations):
            # Create optimized submission
            optimized = self.optimize_submission(
                original_submission=original_submission,
                rubric=rubric,
                use_feedback=True,
            )

            # Get confidence from the optimization
            confidence = optimized.metadata.get("confidence_level", 0.0)

            # If confident enough, return this submission
            if confidence >= min_confidence_threshold:
                return optimized

            # Get judge feedback for the optimization
            if iteration < max_iterations - 1:  # Don't judge on last iteration
                # Create a dummy comparison for feedback (compare with original)
                from .judge import JudgeResult

                judge_result = judge.judge_single(
                    original_submission, optimized, rubric
                )

                # Create feedback from judge result
                feedback = OptimizationFeedback(
                    submission_id=optimized.id,
                    content=optimized.content,
                    performance_score=confidence,  # Use agent confidence as proxy score
                    judge_feedback=judge_result.reasoning,
                    suggestions=optimized.metadata.get("areas_needing_feedback", []),
                )

                # Add feedback for next iteration
                self.add_feedback(feedback)

                # Update current submission
                current_submission = optimized
                if confidence > best_score:
                    best_submission = optimized
                    best_score = confidence

        return best_submission

    def _create_optimizer_system_prompt(self, rubric: str) -> str:
        """Create system prompt for the optimizer agent."""
        focus_areas_text = "\n".join(
            [f"- {area}" for area in self.strategy.focus_areas]
        )

        return f"""You are an expert performance review optimizer with the following specialization:

OPTIMIZATION STRATEGY: {self.strategy.description}

FOCUS AREAS:
{focus_areas_text}

YOUR MISSION: Improve performance reviews to better align with career advancement criteria while maintaining authenticity and accuracy.

EVALUATION RUBRIC:
{rubric}

OPTIMIZATION PRINCIPLES:
1. Enhance clarity and specificity without exaggerating
2. Add quantifiable metrics where possible
3. Improve structure and professional presentation
4. Highlight growth, learning, and impact
5. Align language with career ladder expectations
6. Maintain truthfulness - never fabricate accomplishments
7. Apply your specialized focus areas strategically

Your optimizations should be strategic improvements that help the review better demonstrate the person's value and growth potential.

Use the optimize_review function to provide your structured optimization."""

    def _create_optimization_prompt(
        self,
        original_submission: Submission,
        previous_attempts: Optional[List[Submission]] = None,
        use_feedback: bool = True,
    ) -> str:
        """Create user prompt for optimization with feedback integration."""
        prompt = f"""Please optimize this performance review:

ORIGINAL SUBMISSION:
{original_submission.content}

OPTIMIZATION REQUIREMENTS:
1. Apply your specialized strategy: {self.strategy.description}
2. Focus on these key areas: {', '.join(self.strategy.focus_areas)}
3. Improve alignment with the rubric
4. Enhance professional presentation
5. Maintain authenticity - do not fabricate achievements
6. Make the content more compelling for career advancement"""

        # Add feedback from previous performance if available
        if use_feedback and self.feedback_history:
            prompt += "\n\nFEEDBACK FROM PREVIOUS ATTEMPTS:"
            for feedback in self.feedback_history[-2:]:  # Last 2 pieces of feedback
                prompt += f"\n- Score: {feedback.performance_score:.2f}"
                if feedback.judge_feedback:
                    prompt += f"\n- Judge feedback: {feedback.judge_feedback}"
                if feedback.suggestions:
                    prompt += f"\n- Suggestions: {'; '.join(feedback.suggestions)}"
            prompt += "\n\nUse this feedback to improve your optimization approach."

        if previous_attempts:
            prompt += "\n\nPREVIOUS OPTIMIZATION ATTEMPTS:\n"
            for i, attempt in enumerate(
                previous_attempts[-3:], 1
            ):  # Show last 3 attempts
                prompt += f"\nAttempt {i} (by {attempt.agent_name}):\n"
                prompt += f"{attempt.content[:200]}{'...' if len(attempt.content) > 200 else ''}\n"

            prompt += "\nAvoid repeating previous approaches. Try a different optimization strategy."

        return prompt

    def reset_attempts(self) -> None:
        """Reset the attempt counter."""
        self.attempts_made = 0


class OptimizerFactory:
    """Factory for creating different types of optimizer agents."""

    @staticmethod
    def create_agents(
        openai_client: OpenAIClient, num_agents: int = 4, model: str = "gpt-4.1"
    ) -> List[OptimizerAgent]:
        """
        Create basic optimizer agents (deprecated - use create_agents_from_config).

        Args:
            openai_client: OpenAI client instance
            num_agents: Number of agents to create
            model: Model to use for all agents

        Returns:
            List of basic optimizer agents
        """
        # Create basic strategies for backward compatibility
        basic_strategies = []
        for i in range(num_agents):
            strategy = {
                "name": f"Agent{i+1}",
                "description": f"Basic optimizer agent #{i+1}",
                "focus_areas": ["General optimization"],
                "model": model,
            }
            basic_strategies.append(strategy)

        return OptimizerFactory.create_agents_from_config(
            openai_client=openai_client,
            strategies=basic_strategies,
            default_model=model,
        )

    @staticmethod
    def create_agents_from_config(
        openai_client: OpenAIClient,
        strategies: List[Dict[str, Any]],
        default_model: str = "gpt-4.1",
    ) -> List[OptimizerAgent]:
        """
        Create optimizer agents from configuration.

        Args:
            openai_client: OpenAI client instance
            strategies: List of strategy configurations
            default_model: Default model for agents

        Returns:
            List of configured optimizer agents
        """
        agents = []
        for config in strategies:
            strategy = OptimizationStrategy(
                name=config["name"],
                description=config["description"],
                focus_areas=config.get("focus_areas", []),
                model=config.get("model"),
                max_attempts=config.get("max_attempts", 2),
                reasoning=config.get("reasoning"),
            )

            agent = OptimizerAgent(
                strategy=strategy,
                openai_client=openai_client,
                default_model=default_model,
            )
            agents.append(agent)

        return agents
