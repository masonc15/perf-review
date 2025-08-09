"""
Analytics module for tournament performance analysis.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from .models import Submission


@dataclass
class BaselineComparison:
    """Comparison between original submission and AI-optimized versions."""

    baseline_id: str
    total_matches: int
    baseline_wins: int
    optimized_wins: int
    avg_confidence_when_optimized_wins: float
    individual_matchups: List[Dict]

    @property
    def optimized_win_rate(self) -> float:
        return self.optimized_wins / self.total_matches if self.total_matches > 0 else 0


@dataclass
class TournamentWinner:
    """Details about the tournament's top performer."""

    submission: Submission
    final_elo: float
    win_rate: float
    total_matches: int
    wins: int
    losses: int
    avg_confidence_when_winning: float
    beat_baseline: bool
    baseline_matchup_confidence: Optional[float] = None


def analyze_baseline_performance(tournament) -> BaselineComparison:
    """Extract baseline (original) vs optimized performance from tournament."""
    baseline_matches = [
        m
        for m in tournament.matches
        if "original" in [m.submission_a_id, m.submission_b_id]
    ]

    optimized_wins = sum(1 for m in baseline_matches if m.winner_id != "original")
    baseline_wins = len(baseline_matches) - optimized_wins

    # Get confidence scores when optimized version won
    optimized_win_confidences = [
        m.metadata.get("consensus_confidence", m.metadata.get("confidence", 0))
        for m in baseline_matches
        if m.winner_id != "original"
    ]

    return BaselineComparison(
        baseline_id="original",
        total_matches=len(baseline_matches),
        baseline_wins=baseline_wins,
        optimized_wins=optimized_wins,
        avg_confidence_when_optimized_wins=(
            np.mean(optimized_win_confidences) if optimized_win_confidences else 0
        ),
        individual_matchups=[
            {
                "opponent": (
                    m.submission_a_id
                    if m.submission_b_id == "original"
                    else m.submission_b_id
                ),
                "winner": "baseline" if m.winner_id == "original" else "optimized",
                "confidence": m.metadata.get(
                    "consensus_confidence", m.metadata.get("confidence", 0)
                ),
            }
            for m in baseline_matches
        ],
    )


def get_tournament_winner_analysis(tournament) -> TournamentWinner:
    """Get comprehensive stats for the highest-rated submission."""
    # Get top ELO performer from tournament ratings
    ratings = {sub_id: rating.rating for sub_id, rating in tournament.ratings.items()}
    winner_id = max(ratings.keys(), key=lambda k: ratings[k])
    winner_submission = next(s for s in tournament.submissions if s.id == winner_id)

    # Calculate winner's match statistics
    winner_matches = [
        m
        for m in tournament.matches
        if winner_id in [m.submission_a_id, m.submission_b_id]
    ]

    wins = sum(1 for m in winner_matches if m.winner_id == winner_id)
    losses = len(winner_matches) - wins

    # Get confidence when this submission won
    win_confidences = [
        m.metadata.get("consensus_confidence", m.metadata.get("confidence", 0))
        for m in winner_matches
        if m.winner_id == winner_id
    ]

    # Check if winner beat baseline
    baseline_match = next(
        (
            m
            for m in winner_matches
            if "original" in [m.submission_a_id, m.submission_b_id]
        ),
        None,
    )
    beat_baseline = baseline_match and baseline_match.winner_id == winner_id
    baseline_confidence = None
    if baseline_match and beat_baseline:
        baseline_confidence = baseline_match.metadata.get(
            "consensus_confidence", baseline_match.metadata.get("confidence", 0)
        )

    return TournamentWinner(
        submission=winner_submission,
        final_elo=ratings[winner_id],
        win_rate=wins / len(winner_matches) if winner_matches else 0,
        total_matches=len(winner_matches),
        wins=wins,
        losses=losses,
        avg_confidence_when_winning=np.mean(win_confidences) if win_confidences else 0,
        beat_baseline=beat_baseline,
        baseline_matchup_confidence=baseline_confidence,
    )
