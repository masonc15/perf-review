"""
Data models for the performance review optimization system.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import json


@dataclass
class Submission:
    """Represents a performance review submission."""

    id: str
    content: str
    parent_id: Optional[str] = None  # ID of submission this was optimized from
    agent_name: Optional[str] = None  # Name of agent that created this
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "parent_id": self.parent_id,
            "agent_name": self.agent_name,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Submission":
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class ELORating:
    """ELO rating for a submission with confidence tracking."""

    submission_id: str
    rating: float = 1500.0
    matches_played: int = 0
    wins: int = 0
    losses: int = 0
    k_factor: float = 32.0

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.matches_played == 0:
            return 0.0
        return self.wins / self.matches_played

    @property
    def confidence_interval(self) -> float:
        """Rough confidence interval based on matches played."""
        if self.matches_played < 3:
            return 200.0
        elif self.matches_played < 10:
            return 100.0
        elif self.matches_played < 20:
            return 50.0
        else:
            return 25.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "submission_id": self.submission_id,
            "rating": self.rating,
            "matches_played": self.matches_played,
            "wins": self.wins,
            "losses": self.losses,
            "k_factor": self.k_factor,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ELORating":
        return cls(**data)


@dataclass
class Match:
    """Represents a pairwise comparison match."""

    id: str
    submission_a_id: str
    submission_b_id: str
    winner_id: str
    judge_reasoning: str
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "submission_a_id": self.submission_a_id,
            "submission_b_id": self.submission_b_id,
            "winner_id": self.winner_id,
            "judge_reasoning": self.judge_reasoning,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Match":
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class Tournament:
    """Represents a tournament with all submissions, ratings, and matches."""

    id: str
    name: str
    rubric: str
    original_submission: Submission
    submissions: List[Submission] = field(default_factory=list)
    ratings: Dict[str, ELORating] = field(default_factory=dict)
    matches: List[Match] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_submission(self, submission: Submission) -> None:
        """Add a new submission and initialize its ELO rating."""
        self.submissions.append(submission)
        if submission.id not in self.ratings:
            self.ratings[submission.id] = ELORating(submission_id=submission.id)

    def add_match(self, match: Match) -> None:
        """Add a match result."""
        self.matches.append(match)

    def get_leaderboard(self) -> List[tuple]:
        """Get submissions sorted by ELO rating."""
        leaderboard = []
        for submission in self.submissions:
            rating = self.ratings.get(submission.id)
            if rating:
                leaderboard.append((submission, rating))

        return sorted(leaderboard, key=lambda x: x[1].rating, reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "rubric": self.rubric,
            "original_submission": self.original_submission.to_dict(),
            "submissions": [s.to_dict() for s in self.submissions],
            "ratings": {k: v.to_dict() for k, v in self.ratings.items()},
            "matches": [m.to_dict() for m in self.matches],
            "created_at": self.created_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tournament":
        tournament = cls(
            id=data["id"],
            name=data["name"],
            rubric=data["rubric"],
            original_submission=Submission.from_dict(data["original_submission"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            completed_at=(
                datetime.fromisoformat(data["completed_at"])
                if data["completed_at"]
                else None
            ),
            metadata=data["metadata"],
        )

        tournament.submissions = [Submission.from_dict(s) for s in data["submissions"]]
        tournament.ratings = {
            k: ELORating.from_dict(v) for k, v in data["ratings"].items()
        }
        tournament.matches = [Match.from_dict(m) for m in data["matches"]]

        return tournament

    def save_to_file(self, filepath: str) -> None:
        """Save tournament to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> "Tournament":
        """Load tournament from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
