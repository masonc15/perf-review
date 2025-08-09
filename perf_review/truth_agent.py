"""
Truth agent for factual verification of performance review submissions.
"""

import uuid
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from .openai_utils import OpenAIClient
from .models import Submission


@dataclass
class TruthVerificationResult:
    """Result from truth verification."""
    is_truthful: bool
    confidence: float
    flagged_claims: List[str]
    reasoning: str
    verification_id: str
    agent_model: str


class TruthAgent:
    """Agent for verifying factual accuracy of performance review submissions."""
    
    def __init__(self, openai_client: OpenAIClient, model: str = "gpt-4.1", reasoning: Optional[Dict[str, Any]] = None):
        """
        Initialize the truth agent.
        
        Args:
            openai_client: OpenAI client instance
            model: Model to use for verification
            reasoning: Reasoning configuration for GPT-5 models
        """
        self.client = openai_client
        self.model = model
        self.agent_id = f"truth_agent_{model.replace('.', '_')}"
        self.reasoning = reasoning
    
    def verify_submission(
        self,
        submission: Submission,
        ground_truth: str,
        rubric: str
    ) -> TruthVerificationResult:
        """
        Verify if a submission contains truthful claims based on ground truth data.
        
        Args:
            submission: The performance review submission to verify
            ground_truth: Text containing factual data (PRs, tickets, etc.)
            rubric: Career ladder/evaluation criteria
            
        Returns:
            TruthVerificationResult with verification outcome
        """
        system_prompt = self._create_verification_system_prompt(rubric)
        user_prompt = self._create_verification_prompt(submission, ground_truth)
        
        # Define function for structured verification output
        tools = [{
            "type": "function",
            "name": "verify_truthfulness",
            "description": "Verify the factual accuracy of performance review claims",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Detailed analysis of factual accuracy and any concerning claims"
                    },
                    "is_truthful": {
                        "type": "boolean",
                        "description": "Whether the submission appears to be factually accurate"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Confidence in the verification (0-1)"
                    },
                    "flagged_claims": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of specific claims that appear exaggerated, unsubstantiated, or false"
                    }
                },
                "required": ["reasoning", "is_truthful", "confidence", "flagged_claims"],
                "additionalProperties": False
            },
            "strict": True
        }]
        
        # Get response
        messages = self.client.create_structured_prompt(
            system_message=system_prompt,
            user_message=user_prompt
        )
        
        response = self.client.create_response(
            input_messages=messages,
            tools=tools,
            tool_choice={"type": "function", "name": "verify_truthfulness"},
            reasoning=self.reasoning
        )
        
        # Extract function call result
        function_calls = self.client.extract_function_calls(response)
        
        if not function_calls:
            raise ValueError("Truth agent did not provide structured verification result")
        
        result = function_calls[0]["arguments"]
        
        return TruthVerificationResult(
            is_truthful=result["is_truthful"],
            confidence=result["confidence"],
            flagged_claims=result["flagged_claims"],
            reasoning=result["reasoning"],
            verification_id=str(uuid.uuid4()),
            agent_model=self.model
        )
    
    def batch_verify(
        self,
        submissions: List[Submission],
        ground_truth: str,
        rubric: str,
        batch_size: int = 10
    ) -> List[TruthVerificationResult]:
        """
        Verify multiple submissions in parallel.
        
        Args:
            submissions: List of submissions to verify
            ground_truth: Text containing factual data
            rubric: Career ladder/evaluation criteria
            batch_size: Number of parallel verification tasks
            
        Returns:
            List of verification results
        """
        results = []
        
        # Run verifications in parallel
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Submit all verification tasks
            future_to_submission = {}
            for submission in submissions:
                future = executor.submit(self._verify_single, submission, ground_truth, rubric)
                future_to_submission[future] = submission
            
            # Collect results as they complete
            for future in as_completed(future_to_submission):
                submission = future_to_submission[future]
                try:
                    result = future.result()
                    results.append(result)
                    if result.is_truthful:
                        print(f"✅ Truth verification passed: {submission.agent_name or 'Original'}")
                    else:
                        print(f"❌ Truth verification failed: {submission.agent_name or 'Original'}")
                except Exception as e:
                    print(f"⚠️ Truth verification error for {submission.agent_name or 'Original'}: {e}")
                    # Create a failed result
                    results.append(TruthVerificationResult(
                        is_truthful=False,
                        confidence=0.0,
                        flagged_claims=[f"Verification failed: {str(e)}"],
                        reasoning=f"Error during verification: {str(e)}",
                        verification_id=str(uuid.uuid4()),
                        agent_model=self.model
                    ))
        
        return results
    
    def _verify_single(self, submission: Submission, ground_truth: str, rubric: str) -> TruthVerificationResult:
        """Verify a single submission."""
        return self.verify_submission(submission, ground_truth, rubric)
    
    def _create_verification_system_prompt(self, rubric: str) -> str:
        """Create system prompt for truth verification."""
        return f"""You are an expert fact-checker specializing in performance review accuracy. Your job is to verify whether claims in performance reviews are truthful, realistic, and supported by the provided ground truth data.

EVALUATION RUBRIC:
{rubric}

VERIFICATION GUIDELINES:
1. Cross-reference specific claims against ground truth data
2. Flag numerical claims that seem exaggerated (e.g., impossible percentages, unrealistic impact numbers)
3. Identify vague claims that cannot be verified
4. Look for claims that contradict the ground truth evidence
5. Consider whether achievements align with typical senior engineer scope

RED FLAGS TO WATCH FOR:
- Claiming credit for work not in ground truth data
- Exaggerated impact metrics (e.g., "300% improvement")
- Timeline inconsistencies
- Claims about projects, bugs, or features not reflected in data
- Taking sole credit for team achievements

IMPORTANT: Be thorough but fair. Some optimization and enhancement is expected in performance reviews. Focus on flagging clear fabrications, significant exaggerations, or claims that contradict evidence.

Use the verify_truthfulness function to provide your structured analysis."""
    
    def _create_verification_prompt(self, submission: Submission, ground_truth: str) -> str:
        """Create prompt for verifying a specific submission."""
        
        # Format ground truth data for verification
        ground_truth_summary = self._format_ground_truth(ground_truth)
        
        return f"""Please verify the factual accuracy of this performance review submission against the provided ground truth data:

SUBMISSION TO VERIFY:
{submission.content}

GROUND TRUTH DATA:
{ground_truth_summary}

VERIFICATION TASK:
1. Compare specific claims in the submission against the ground truth data
2. Identify any claims that appear to be fabricated, significantly exaggerated, or unsupported
3. Assess whether numerical metrics and impact claims are realistic
4. Check for timeline consistency and proper attribution
5. Determine overall truthfulness and provide confidence level

Focus on catching clear fabrications while allowing for reasonable enhancement and professional presentation of achievements."""
    
    def _format_ground_truth(self, ground_truth: str) -> str:
        """Format ground truth data for verification prompt."""
        if not ground_truth or not ground_truth.strip():
            return "No ground truth data provided - verification will be limited to detecting obvious exaggerations."
        
        return ground_truth.strip()


class TruthGuardedArena:
    """Arena extension that uses truth agent to filter submissions."""
    
    def __init__(self, truth_agent: TruthAgent, min_truthfulness_threshold: float = 0.7):
        """
        Initialize truth-guarded arena.
        
        Args:
            truth_agent: Truth verification agent
            min_truthfulness_threshold: Minimum truthfulness score to allow submission
        """
        self.truth_agent = truth_agent
        self.min_threshold = min_truthfulness_threshold
    
    def filter_submissions(
        self,
        submissions: List[Submission],
        ground_truth: str,
        rubric: str,
        batch_size: int = 10
    ) -> tuple[List[Submission], List[TruthVerificationResult]]:
        """
        Filter submissions based on truthfulness verification.
        
        Args:
            submissions: List of submissions to filter
            ground_truth: Text containing factual data for verification
            rubric: Career ladder/evaluation criteria
            batch_size: Number of parallel verification tasks
            
        Returns:
            Tuple of (approved_submissions, all_verification_results)
        """
        verification_results = self.truth_agent.batch_verify(submissions, ground_truth, rubric, batch_size)
        
        approved_submissions = []
        for submission, verification in zip(submissions, verification_results):
            if verification.is_truthful and verification.confidence >= self.min_threshold:
                approved_submissions.append(submission)
            else:
                print(f"⚠️  Submission blocked by truth agent: {verification.reasoning}")
                if verification.flagged_claims:
                    print(f"   Flagged claims: {verification.flagged_claims}")
        
        return approved_submissions, verification_results