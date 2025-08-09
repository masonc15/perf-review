"""
Performance Review Optimization System

An AI-powered system that uses ELO rankings and LLM agents to optimize
performance reviews through competitive tournaments.
"""

from .models import Submission, ELORating, Match, Tournament
from .elo_system import ELOSystem
from .openai_utils import OpenAIClient
from .judge import LLMJudge, MultiJudge, JudgeResult, MultiJudgeResult
from .truth_agent import TruthAgent, TruthVerificationResult, TruthGuardedArena
from .optimizer import OptimizerAgent, OptimizerFactory, OptimizationStrategy, OptimizationFeedback
from .arena import Arena

__version__ = "0.1.0"