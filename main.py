#!/usr/bin/env python3
"""
Main script for running performance review optimization tournaments.
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, List

from perf_review import (
    OpenAIClient,
    LLMJudge,
    MultiJudge,
    Arena,
    ELOSystem,
    OptimizerFactory,
    Tournament,
    Submission,
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return {}

    with open(config_path, "r") as f:
        return json.load(f)


def create_sample_config() -> Dict[str, Any]:
    """Create a sample configuration."""
    return {
        "tournament": {
            "name": "Performance Review Optimization",
            "num_rounds": 3,
            "matches_per_round": 15,
            "max_attempts_per_agent": 2,
        },
        "agents": [
            {
                "name": "MetricsOptimizer",
                "description": "Focus on quantifying impact with specific numbers, percentages, and measurable outcomes. Transform vague accomplishments into concrete, data-driven achievements.",
                "model": "gpt-5",
                "reasoning": {"effort": "high"},
                "focus_areas": [
                    "Quantifiable metrics",
                    "Data-driven results",
                    "Performance indicators",
                    "Impact measurement",
                ],
            },
            {
                "name": "LeadershipOptimizer",
                "description": "Emphasize leadership qualities, team impact, mentoring, and influence beyond individual contributions. Highlight collaboration and people development.",
                "model": "gpt-5",
                "reasoning": {"effort": "high"},
                "focus_areas": [
                    "Team leadership",
                    "Mentoring",
                    "Cross-functional collaboration",
                    "Influence and consensus building",
                ],
            },
            {
                "name": "StrategicOptimizer",
                "description": "Frame accomplishments in terms of business impact, strategic thinking, and long-term value creation. Connect individual work to broader organizational goals.",
                "model": "gpt-5",
                "reasoning": {"effort": "high"},
                "focus_areas": [
                    "Business alignment",
                    "Strategic thinking",
                    "Long-term planning",
                    "Organizational impact",
                ],
            },
            {
                "name": "GrowthOptimizer",
                "description": "Highlight learning, skill development, innovation, and adaptability. Demonstrate continuous improvement and future potential.",
                "model": "gpt-5-mini",
                "focus_areas": [
                    "Continuous learning",
                    "Skill development",
                    "Innovation",
                    "Adaptability",
                ],
            },
            {
                "name": "ClarityOptimizer",
                "description": "Improve structure, flow, and professional presentation. Make the review more readable, organized, and compelling through better writing.",
                "model": "gpt-5-mini",
                "focus_areas": [
                    "Clear communication",
                    "Professional structure",
                    "Narrative flow",
                    "Readability",
                ],
            },
            {
                "name": "ImpactOptimizer",
                "description": "Focus on customer impact, revenue generation, cost savings, and external value creation. Emphasize market-facing results.",
                "model": "gpt-5",
                "reasoning": {"effort": "minimal"},
                "focus_areas": [
                    "Customer impact",
                    "Revenue generation",
                    "Cost optimization",
                    "Market value creation",
                ],
            },
            {
                "name": "TechnicalOptimizer",
                "description": "Emphasize technical depth, architectural decisions, and engineering excellence. Highlight complex problem-solving and technical leadership.",
                "model": "gpt-5",
                "reasoning": {"effort": "high"},
                "focus_areas": [
                    "Technical depth",
                    "System design",
                    "Code quality",
                    "Engineering excellence",
                ],
            },
            {
                "name": "ConciseOptimizer",
                "description": "Create focused, high-impact summaries that maximize readability and scanning efficiency. Perfect for executive audiences.",
                "model": "gpt-5-nano",
                "focus_areas": [
                    "Brevity",
                    "Impact focus",
                    "Executive communication",
                    "Key highlights",
                ],
            },
            {
                "name": "GeneralOptimizer_GPT5",
                "description": "Comprehensive optimization focusing on overall improvement across all dimensions. Uses advanced reasoning for holistic enhancement.",
                "model": "gpt-5",
                "reasoning": {"effort": "high"},
                "focus_areas": [
                    "Overall optimization",
                    "Comprehensive improvement",
                    "Professional enhancement",
                    "Quality maximization",
                ],
            },
            {
                "name": "GeneralOptimizer_O3",
                "description": "General purpose optimization with balanced approach to improving all aspects of the review or resume.",
                "model": "o3",
                "reasoning": {"effort": "medium"},
                "focus_areas": [
                    "General improvement",
                    "Balanced optimization",
                    "Content enhancement",
                    "Professional polish",
                ],
            },
            {
                "name": "GeneralOptimizer_O4Mini",
                "description": "Efficient general optimization focused on quick, effective improvements across multiple areas.",
                "model": "o4-mini",
                "reasoning": {"effort": "medium"},
                "focus_areas": [
                    "Efficient optimization",
                    "Multi-area improvement",
                    "Quick enhancement",
                    "Streamlined polish",
                ],
            },
        ],
        "judge": {
            "model": "gpt-5",
            "reasoning": {"effort": "medium"},
            "multi_judge": {
                "enabled": True,
                "models": ["gpt-5", "o3", "o4-mini"],
                "require_majority": True,
            },
        },
        "truth_agent": {
            "enabled": True,
            "model": "gpt-5",
            "reasoning": {"effort": "medium"},
            "min_truthfulness_threshold": 0.7,
        },
        "original_review": "I worked on several projects this year including implementing new features for the web application, fixing bugs, and helping with code reviews. I collaborated with the team and learned new technologies. I also participated in planning meetings and contributed to architecture decisions.",
        "ground_truth": "Pull Requests: Implemented user authentication system (450 lines), added responsive design to dashboard (320 lines), fixed memory leak in data processing (85 lines), optimized database queries for reports (200 lines), added unit tests for payment module (380 lines). \n\nTickets: Fixed login page not loading on mobile (high priority bug), dashboard showing incorrect user data (medium priority bug), added two-factor authentication (medium priority feature), performance optimization for large datasets (high priority enhancement).\n\nProjects: Led Authentication Overhaul project for 4 weeks with 3-person team, redesigned secure authentication system. Led Dashboard Modernization project for 6 weeks with 2-person team, updated dashboard with responsive design and improved UX.\n\nMetrics: Completed 45 code reviews, fixed 12 bugs, delivered 3 features, created 8 documentation pages, mentored 2 team members.\n\nCollaborations: Led architecture discussion for payment system redesign, mentored junior developer on testing best practices, collaborated with design team on user experience improvements, participated in cross-team API standardization effort.",
        "rubric": """Career Ladder - Senior Engineer Level:
        
        Technical Impact:
        - Delivers complex features end-to-end with minimal guidance
        - Writes clean, maintainable, well-tested code
        - Makes sound technical decisions and considers trade-offs
        
        Leadership & Collaboration:
        - Mentors junior engineers and provides constructive feedback
        - Leads technical discussions and drives consensus
        - Collaborates effectively across teams
        
        Business Impact:
        - Understands business context and priorities
        - Delivers measurable value to customers
        - Identifies and addresses technical debt
        
        Growth & Learning:
        - Stays current with industry trends and best practices
        - Shares knowledge through documentation and presentations
        - Takes on challenging projects outside comfort zone""",
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AI Performance Review Optimizer")
    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    parser.add_argument(
        "--create-sample-config",
        action="store_true",
        help="Create a sample configuration file",
    )
    parser.add_argument(
        "--review", type=str, help="Performance review text (overrides config)"
    )
    parser.add_argument(
        "--rubric", type=str, help="Career ladder/rubric text (overrides config)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tournament_results.json",
        help="Output file for tournament results",
    )
    parser.add_argument(
        "--agents", type=int, default=4, help="Number of optimizer agents"
    )
    parser.add_argument(
        "--rounds", type=int, default=3, help="Number of tournament rounds"
    )

    args = parser.parse_args()

    # Handle sample config creation
    if args.create_sample_config:
        sample_config = create_sample_config()
        config_path = "sample_config.json"
        with open(config_path, "w") as f:
            json.dump(sample_config, f, indent=2)
        print(f"Sample configuration created: {config_path}")
        return

    # Load configuration
    config = {}
    if args.config:
        config = load_config(args.config)

    if not config:
        config = create_sample_config()
        print("Using default configuration. Use --create-sample-config to save it.")

    # Override with command line arguments
    if args.review:
        config["original_review"] = args.review
    if args.rubric:
        config["rubric"] = args.rubric
    if args.agents:
        # Limit agents array to specified count for backward compatibility
        config["agents"] = config["agents"][: args.agents]
    if args.rounds:
        config["tournament"]["num_rounds"] = args.rounds

    # Validate required fields
    if not config.get("original_review"):
        print("Error: No performance review provided. Use --review or config file.")
        sys.exit(1)

    if not config.get("rubric"):
        print("Error: No rubric provided. Use --rubric or config file.")
        sys.exit(1)

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key or source env.sh")
        sys.exit(1)

    try:
        # Initialize components
        print("Initializing AI components...")
        openai_client = OpenAIClient(model=config["judge"]["model"])

        # Initialize judge system (single or multi-judge)
        if config["judge"].get("multi_judge", {}).get("enabled", False):
            # Create multiple judges
            judge_models = config["judge"]["multi_judge"]["models"]
            judges = []
            print(f"Creating multi-judge system with models: {judge_models}")
            for model in judge_models:
                judge_instance = LLMJudge(
                    openai_client,
                    model=model,
                    reasoning=config["judge"].get("reasoning"),
                )
                judges.append(judge_instance)
            judge = MultiJudge(judges)
        else:
            # Single judge
            judge = LLMJudge(
                openai_client,
                model=config["judge"]["model"],
                reasoning=config["judge"].get("reasoning"),
            )

        elo_system = ELOSystem()
        arena = Arena(judge, elo_system)

        # Create optimizer agents from config list
        optimizers = OptimizerFactory.create_agents_from_config(
            openai_client=openai_client,
            strategies=config["agents"],
            default_model="gpt-4.1",  # fallback if no model specified
        )

        print(f"Created {len(optimizers)} optimizer agents")

        # Create original submission
        original_submission = Submission(
            id="original", content=config["original_review"], agent_name=None
        )

        # Create tournament
        tournament = arena.create_tournament(
            name=config["tournament"]["name"],
            original_submission=original_submission,
            rubric=config["rubric"],
        )

        print(f"Created tournament: {tournament.name}")
        print(f"Original review length: {len(original_submission.content)} characters")

        # Run tournament with parallel processing
        completed_tournament = arena.run_full_tournament(
            tournament=tournament,
            optimizers=optimizers,
            num_rounds=config["tournament"]["num_rounds"],
            matches_per_round=config["tournament"]["matches_per_round"],
            max_attempts_per_agent=config["tournament"]["max_attempts_per_agent"],
            optimization_batch_size=10,  # Parallel optimization tasks
            match_batch_size=10,  # Parallel match tasks
        )

        # Save results
        completed_tournament.save_to_file(args.output)
        print(f"\nResults saved to: {args.output}")

        # Display final results
        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)

        leaderboard = completed_tournament.get_leaderboard()

        print(f"\nTotal submissions: {len(completed_tournament.submissions)}")
        print(f"Total matches: {len(completed_tournament.matches)}")

        print(f"\n🥇 WINNER (ELO: {leaderboard[0][1].rating:.0f}):")
        print("-" * 50)
        winner_submission = leaderboard[0][0]
        print(f"Agent: {winner_submission.agent_name or 'Original'}")
        print(f"Content:\n{winner_submission.content}")

        if winner_submission.metadata.get("improvements_made"):
            print(f"\nImprovements made:")
            for improvement in winner_submission.metadata["improvements_made"]:
                print(f"• {improvement}")

        if winner_submission.metadata.get("rationale"):
            print(f"\nOptimization rationale:")
            print(winner_submission.metadata["rationale"])

        print("\nTournament completed successfully!")

    except KeyboardInterrupt:
        print("\nTournament interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
