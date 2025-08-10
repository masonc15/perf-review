"""
ELO rating system for performance review tournaments.
"""

import math
import random
from typing import Tuple
from .models import ELORating


class ELOSystem:
    """Handles ELO rating calculations and updates."""

    def __init__(
        self,
        default_k_factor: float = 32.0,
        min_rating: float = 800.0,
        max_rating: float = 2800.0,
        starting_rating: float = 1500.0,
        use_stochastic_start: bool = True,
    ):
        """
        Initialize ELO system.

        Args:
            default_k_factor: Default K-factor for rating changes
            min_rating: Minimum allowed rating (prevents extreme negatives)
            max_rating: Maximum allowed rating (prevents extreme positives)
            starting_rating: Base starting rating for new players
            use_stochastic_start: Add small random variation to starting ratings
        """
        self.default_k_factor = default_k_factor
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.starting_rating = starting_rating
        self.use_stochastic_start = use_stochastic_start

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score for player A against player B.

        Args:
            rating_a: ELO rating of player A
            rating_b: ELO rating of player B

        Returns:
            Expected score (0-1) for player A
        """
        return 1 / (1 + math.pow(10, (rating_b - rating_a) / 400))

    def update_ratings(
        self, rating_a: ELORating, rating_b: ELORating, winner_is_a: bool
    ) -> Tuple[ELORating, ELORating]:
        """
        Update ELO ratings after a match.

        Args:
            rating_a: ELO rating object for player A
            rating_b: ELO rating object for player B
            winner_is_a: True if player A won, False if player B won

        Returns:
            Updated rating objects (rating_a, rating_b)
        """
        # Calculate expected scores
        expected_a = self.expected_score(rating_a.rating, rating_b.rating)
        expected_b = 1 - expected_a

        # Actual scores
        actual_a = 1.0 if winner_is_a else 0.0
        actual_b = 1.0 - actual_a

        # Use adaptive K-factor based on matches played
        k_a = self._get_k_factor(rating_a)
        k_b = self._get_k_factor(rating_b)

        # Calculate new ratings
        new_rating_a = rating_a.rating + k_a * (actual_a - expected_a)
        new_rating_b = rating_b.rating + k_b * (actual_b - expected_b)

        # Apply rating bounds
        new_rating_a = max(self.min_rating, min(self.max_rating, new_rating_a))
        new_rating_b = max(self.min_rating, min(self.max_rating, new_rating_b))

        # Update rating objects
        rating_a.rating = new_rating_a
        rating_a.matches_played += 1
        if winner_is_a:
            rating_a.wins += 1
        else:
            rating_a.losses += 1

        rating_b.rating = new_rating_b
        rating_b.matches_played += 1
        if winner_is_a:
            rating_b.losses += 1
        else:
            rating_b.wins += 1

        return rating_a, rating_b

    def _get_k_factor(self, rating: ELORating) -> float:
        """
        Get adaptive K-factor based on matches played, current rating, and performance.
        Uses more aggressive K-factors for faster convergence with fewer matches.

        Args:
            rating: ELO rating object

        Returns:
            K-factor to use for this player
        """
        # Base K-factor from matches played (more aggressive than standard)
        if rating.matches_played < 5:
            base_k = 64.0  # Very high for initial rapid adjustment
        elif rating.matches_played < 12:
            base_k = 48.0  # High for new players
        elif rating.matches_played < 25:
            base_k = 32.0  # Medium
        else:
            base_k = 24.0  # Higher floor than standard (16.0)

        # Rating-based modifier: extreme ratings get lower K to prevent wild swings
        if rating.rating < 1200 or rating.rating > 1800:
            base_k *= 0.8  # Reduce volatility at extremes

        # Performance-based modifier: consistent performers get slightly lower K
        if rating.matches_played >= 8:
            win_rate = rating.win_rate
            if 0.4 <= win_rate <= 0.6:  # Very balanced performance
                base_k *= 0.9  # Slight reduction for stable players

        return base_k

    def create_initial_rating(self, submission_id: str) -> ELORating:
        """
        Create initial ELO rating with optional stochastic variation.

        Args:
            submission_id: Unique ID for the submission

        Returns:
            New ELORating with initial rating
        """
        base_rating = self.starting_rating

        if self.use_stochastic_start:
            # Add small random variation (±25 points) to prevent identical starts
            variation = random.uniform(-25, 25)
            initial_rating = base_rating + variation
            # Ensure within bounds
            initial_rating = max(self.min_rating, min(self.max_rating, initial_rating))
        else:
            initial_rating = base_rating

        return ELORating(submission_id=submission_id, rating=initial_rating)

    def rating_difference_threshold(self, rating_a: float, rating_b: float) -> str:
        """
        Get descriptive threshold for rating difference.

        Args:
            rating_a: First player's rating
            rating_b: Second player's rating

        Returns:
            Description of skill gap
        """
        diff = abs(rating_a - rating_b)

        if diff < 50:
            return "Very close"
        elif diff < 100:
            return "Close"
        elif diff < 200:
            return "Moderate gap"
        elif diff < 300:
            return "Large gap"
        else:
            return "Huge gap"

    def win_probability(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate probability that player A beats player B.

        Args:
            rating_a: ELO rating of player A
            rating_b: ELO rating of player B

        Returns:
            Probability (0-1) that A beats B
        """
        return self.expected_score(rating_a, rating_b)
