"""
ELO rating system for performance review tournaments.
"""

import math
from typing import Tuple
from .models import ELORating


class ELOSystem:
    """Handles ELO rating calculations and updates."""
    
    def __init__(self, default_k_factor: float = 32.0):
        """
        Initialize ELO system.
        
        Args:
            default_k_factor: Default K-factor for rating changes
        """
        self.default_k_factor = default_k_factor
    
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
        self,
        rating_a: ELORating,
        rating_b: ELORating,
        winner_is_a: bool
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
        Get adaptive K-factor based on number of matches played.
        New players have higher K-factor for faster adjustment.
        
        Args:
            rating: ELO rating object
            
        Returns:
            K-factor to use for this player
        """
        if rating.matches_played < 10:
            return 48.0  # High K-factor for new players
        elif rating.matches_played < 25:
            return 32.0  # Medium K-factor
        else:
            return 16.0  # Low K-factor for experienced players
    
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