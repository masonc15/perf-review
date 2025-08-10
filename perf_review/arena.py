"""
Tournament arena for running performance review optimization competitions.
"""

import uuid
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict, Any
from .models import Tournament, Submission, Match, ELORating
from .elo_system import ELOSystem
from .judge import LLMJudge
from .optimizer import OptimizerAgent


class Arena:
    """Manages tournaments and competitions between performance reviews."""

    def __init__(self, judge: LLMJudge, elo_system: Optional[ELOSystem] = None):
        """
        Initialize the arena.

        Args:
            judge: LLM judge for comparisons
            elo_system: ELO rating system (creates default if None)
        """
        self.judge = judge
        self.elo_system = elo_system or ELOSystem()

    def create_tournament(
        self, name: str, original_submission: Submission, rubric: str
    ) -> Tournament:
        """
        Create a new tournament.

        Args:
            name: Tournament name
            original_submission: Initial performance review
            rubric: Evaluation criteria

        Returns:
            New tournament instance
        """
        tournament = Tournament(
            id=str(uuid.uuid4()),
            name=name,
            rubric=rubric,
            original_submission=original_submission,
        )

        # Add original submission to tournament with initial ELO rating
        original_rating = self.elo_system.create_initial_rating(original_submission.id)
        tournament.add_submission(original_submission, original_rating)

        return tournament

    def run_optimization_round(
        self,
        tournament: Tournament,
        optimizers: List[OptimizerAgent],
        max_attempts_per_agent: int = 1,
        batch_size: int = 10,
    ) -> List[Submission]:
        """
        Run one round of optimization with all agents in parallel.

        Args:
            tournament: Tournament to add optimized submissions to
            optimizers: List of optimizer agents
            max_attempts_per_agent: Maximum attempts each agent can make
            batch_size: Number of parallel optimization tasks

        Returns:
            List of newly created submissions
        """
        new_submissions = []

        # Get existing submissions as potential parents
        existing_submissions = tournament.submissions.copy()

        # Create optimization tasks
        optimization_tasks = []
        for optimizer in optimizers:
            for attempt in range(max_attempts_per_agent):
                if optimizer.attempts_made >= max_attempts_per_agent:
                    break

                # Select submission to optimize
                parent_submission = self._select_parent_submission(existing_submissions)
                optimization_tasks.append(
                    (optimizer, parent_submission, tournament.rubric)
                )

        # Run optimizations in parallel batches
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Submit all tasks
            future_to_task = {}
            for optimizer, parent_submission, rubric in optimization_tasks:
                future = executor.submit(
                    self._run_single_optimization, optimizer, parent_submission, rubric
                )
                future_to_task[future] = (optimizer, parent_submission)

            # Collect results as they complete
            for future in as_completed(future_to_task):
                optimizer, parent_submission = future_to_task[future]
                try:
                    submission = future.result()
                    if submission:
                        new_submissions.append(submission)
                        # Create ELO rating for new submission
                        new_rating = self.elo_system.create_initial_rating(
                            submission.id
                        )
                        tournament.add_submission(submission, new_rating)
                        optimizer.attempts_made += 1
                        print(
                            f"✅ {optimizer.strategy.name}: Created optimized submission"
                        )
                    else:
                        print(
                            f"⚠️  {optimizer.strategy.name}: Failed to create submission"
                        )
                except Exception as e:
                    print(f"❌ {optimizer.strategy.name}: Optimization failed: {e}")

        return new_submissions

    def _run_single_optimization(
        self, optimizer: OptimizerAgent, parent_submission: Submission, rubric: str
    ) -> Optional[Submission]:
        """Run a single optimization task."""
        try:
            return optimizer.optimize_submission(parent_submission, rubric)
        except Exception as e:
            print(f"Optimization error: {e}")
            return None

    def _select_parent_submission(self, submissions: List[Submission]) -> Submission:
        """Select a parent submission for optimization."""
        if not submissions:
            raise ValueError("No submissions available for optimization")

        # Select highest rated submission as parent
        if len(submissions) == 1:
            return submissions[0]

        # Simple random selection for now - could be improved with rating-based selection
        return random.choice(submissions)

    def run_tournament_matches(
        self,
        tournament: Tournament,
        num_matches: Optional[int] = None,
        match_strategy: str = "round_robin",
        batch_size: int = 10,
    ) -> List[Match]:
        """
        Run matches between submissions in the tournament in parallel.

        Args:
            tournament: Tournament to run matches for
            num_matches: Maximum number of matches (None for all possible)
            match_strategy: Strategy for selecting matches
            batch_size: Number of parallel match tasks

        Returns:
            List of match results
        """
        new_matches = []

        if match_strategy == "round_robin":
            pairs = self._get_round_robin_pairs(tournament.submissions)
        elif match_strategy == "swiss":
            pairs = self._get_swiss_pairs(tournament)
        elif match_strategy == "random":
            pairs = self._get_random_pairs(tournament.submissions, num_matches or 20)
        else:
            raise ValueError(f"Unknown match strategy: {match_strategy}")

        # Limit matches if specified
        if num_matches:
            pairs = pairs[:num_matches]

        # Ensure all submissions involved in matches have ratings
        for submission_a, submission_b in pairs:
            if submission_a.id not in tournament.ratings:
                tournament.ratings[submission_a.id] = (
                    self.elo_system.create_initial_rating(submission_a.id)
                )
            if submission_b.id not in tournament.ratings:
                tournament.ratings[submission_b.id] = (
                    self.elo_system.create_initial_rating(submission_b.id)
                )

        # Run matches in parallel batches
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Submit all match tasks
            future_to_pair = {}
            for submission_a, submission_b in pairs:
                future = executor.submit(
                    self._run_single_match,
                    submission_a,
                    submission_b,
                    tournament.rubric,
                )
                future_to_pair[future] = (submission_a, submission_b)

            # Collect results as they complete
            for future in as_completed(future_to_pair):
                submission_a, submission_b = future_to_pair[future]
                try:
                    match = future.result()
                    if match:
                        # Add match to tournament and update ratings
                        tournament.add_match(match)
                        new_matches.append(match)

                        # Update ELO ratings
                        rating_a = tournament.ratings[submission_a.id]
                        rating_b = tournament.ratings[submission_b.id]

                        winner_is_a = match.winner_id == submission_a.id
                        new_rating_a, new_rating_b = self.elo_system.update_ratings(
                            rating_a, rating_b, winner_is_a
                        )

                        tournament.ratings[submission_a.id] = new_rating_a
                        tournament.ratings[submission_b.id] = new_rating_b

                        print(
                            f"⚖️ Match completed: {submission_a.agent_name or 'Original'} vs {submission_b.agent_name or 'Original'}"
                        )
                    else:
                        print(
                            f"⚠️ Match failed: {submission_a.agent_name or 'Original'} vs {submission_b.agent_name or 'Original'}"
                        )
                except Exception as e:
                    print(f"❌ Match error: {e}")

        return new_matches

    def _run_single_match(
        self, submission_a: Submission, submission_b: Submission, rubric: str
    ) -> Optional[Match]:
        """Run a single match between two submissions."""
        try:
            return self.judge.compare_submissions(submission_a, submission_b, rubric)
        except Exception as e:
            print(f"Match error: {e}")
            return None

    def run_full_tournament(
        self,
        tournament: Tournament,
        optimizers: List[OptimizerAgent],
        num_rounds: int = 3,
        matches_per_round: int = 10,
        max_attempts_per_agent: int = 2,
        optimization_batch_size: int = 10,
        match_batch_size: int = 10,
    ) -> Tournament:
        """
        Run a complete tournament with multiple optimization and match rounds.

        Args:
            tournament: Tournament to run
            optimizers: List of optimizer agents
            num_rounds: Number of optimization rounds
            matches_per_round: Number of matches to run each round
            max_attempts_per_agent: Max attempts per agent per round
            optimization_batch_size: Number of parallel optimization tasks
            match_batch_size: Number of parallel match tasks

        Returns:
            Completed tournament
        """
        print(f"Starting tournament: {tournament.name}")
        print(f"Optimizers: {[opt.name for opt in optimizers]}")

        for round_num in range(1, num_rounds + 1):
            print(f"\n--- Round {round_num} ---")

            # Reset agent attempts for this round
            for optimizer in optimizers:
                optimizer.reset_attempts()

            # Run optimization round with parallel processing
            print("Running optimization...")
            new_submissions = self.run_optimization_round(
                tournament, optimizers, max_attempts_per_agent, optimization_batch_size
            )
            print(f"Created {len(new_submissions)} new submissions")

            # Run matches with parallel processing
            print("Running matches...")
            new_matches = self.run_tournament_matches(
                tournament,
                num_matches=matches_per_round,
                match_strategy="swiss",
                batch_size=match_batch_size,
            )
            print(f"Completed {len(new_matches)} matches")

            # Show current leaderboard
            self._print_leaderboard(tournament, top_n=5)

        # Final comprehensive match round with parallel processing
        print(f"\n--- Final Round ---")
        final_matches = self.run_tournament_matches(
            tournament,
            num_matches=50,
            match_strategy="swiss",
            batch_size=match_batch_size,
        )
        print(f"Completed {len(final_matches)} final matches")

        tournament.completed_at = tournament.created_at  # Set completion time

        print("\nTournament completed!")
        self._print_leaderboard(tournament)

        return tournament

    def _get_round_robin_pairs(
        self, submissions: List[Submission]
    ) -> List[Tuple[Submission, Submission]]:
        """Get all possible pairs for round-robin tournament."""
        pairs = []
        for i in range(len(submissions)):
            for j in range(i + 1, len(submissions)):
                pairs.append((submissions[i], submissions[j]))
        return pairs

    def _get_swiss_pairs(
        self, tournament: Tournament
    ) -> List[Tuple[Submission, Submission]]:
        """Get pairs using Swiss tournament system (similar ratings face each other)."""
        # Sort submissions by ELO rating
        # Ensure all submissions have ratings before sorting
        for submission in tournament.submissions:
            if submission.id not in tournament.ratings:
                tournament.ratings[submission.id] = (
                    self.elo_system.create_initial_rating(submission.id)
                )

        sorted_submissions = sorted(
            tournament.submissions,
            key=lambda s: tournament.ratings[s.id].rating,
            reverse=True,
        )

        pairs = []
        used_indices = set()

        for i in range(0, len(sorted_submissions) - 1, 2):
            if i not in used_indices and (i + 1) not in used_indices:
                pairs.append((sorted_submissions[i], sorted_submissions[i + 1]))
                used_indices.add(i)
                used_indices.add(i + 1)

        return pairs

    def _get_random_pairs(
        self, submissions: List[Submission], num_pairs: int
    ) -> List[Tuple[Submission, Submission]]:
        """Get random pairs of submissions."""
        pairs = []
        all_pairs = self._get_round_robin_pairs(submissions)

        if len(all_pairs) <= num_pairs:
            return all_pairs

        return random.sample(all_pairs, num_pairs)

    def _print_leaderboard(self, tournament: Tournament, top_n: int = 10) -> None:
        """Print current tournament leaderboard."""
        leaderboard = tournament.get_leaderboard()

        print(f"\n🏆 Leaderboard (Top {min(top_n, len(leaderboard))}):")
        print("-" * 80)
        print(
            f"{'Rank':<5} {'Rating':<8} {'W-L':<8} {'Agent':<15} {'Confidence':<12} {'Preview'}"
        )
        print("-" * 80)

        for rank, (submission, rating) in enumerate(leaderboard[:top_n], 1):
            win_loss = f"{rating.wins}-{rating.losses}"
            confidence = f"±{rating.confidence_interval:.0f}"
            agent_name = submission.agent_name or "Original"
            preview = submission.content[:50].replace("\n", " ") + "..."

            print(
                f"{rank:<5} {rating.rating:<8.0f} {win_loss:<8} {agent_name:<15} {confidence:<12} {preview}"
            )
