#!/usr/bin/env python3
"""
Streamlit app for AI-powered performance review optimization.
"""

import streamlit as st
import json
import os
import tempfile
import random
import time
from typing import Dict, Any, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path for imports
import sys

sys.path.append(".")

from perf_review import (
    OpenAIClient,
    LLMJudge,
    MultiJudge,
    Arena,
    ELOSystem,
    OptimizerFactory,
    Tournament,
    Submission,
    TruthAgent,
    TruthGuardedArena,
)
from perf_review.analytics import (
    analyze_baseline_performance,
    get_tournament_winner_analysis,
)


def generate_optimizer_agents() -> List[Dict[str, Any]]:
    """Generate standardized optimizer agents across all model types."""

    # Base strategy types
    strategies = [
        {
            "name": "Metrics",
            "description": "Focus on quantifying impact with specific numbers, percentages, and measurable outcomes. Transform vague accomplishments into concrete, data-driven achievements.",
            "focus_areas": [
                "Quantifiable metrics",
                "Data-driven results",
                "Performance indicators",
                "Impact measurement",
            ],
        },
        {
            "name": "Leadership",
            "description": "Emphasize leadership qualities, team impact, mentoring, and influence beyond individual contributions. Highlight collaboration and people development.",
            "focus_areas": [
                "Team leadership",
                "Mentoring",
                "Cross-functional collaboration",
                "Influence and consensus building",
            ],
        },
        {
            "name": "Strategic",
            "description": "Frame accomplishments in terms of business impact, strategic thinking, and long-term value creation. Connect individual work to broader organizational goals.",
            "focus_areas": [
                "Business alignment",
                "Strategic thinking",
                "Long-term planning",
                "Organizational impact",
            ],
        },
        {
            "name": "Technical",
            "description": "Emphasize technical depth, architectural decisions, and engineering excellence. Highlight complex problem-solving and technical leadership.",
            "focus_areas": [
                "Technical depth",
                "System design",
                "Code quality",
                "Engineering excellence",
            ],
        },
        {
            "name": "Impact",
            "description": "Focus on customer impact, revenue generation, cost savings, and external value creation. Emphasize market-facing results.",
            "focus_areas": [
                "Customer impact",
                "Revenue generation",
                "Cost optimization",
                "Market value creation",
            ],
        },
        {
            "name": "Growth",
            "description": "Highlight learning, skill development, innovation, and adaptability. Demonstrate continuous improvement and future potential.",
            "focus_areas": [
                "Continuous learning",
                "Skill development",
                "Innovation",
                "Adaptability",
            ],
        },
        {
            "name": "Clarity",
            "description": "Improve structure, flow, and professional presentation. Make content more readable, organized, and compelling through better writing.",
            "focus_areas": [
                "Clear communication",
                "Professional structure",
                "Narrative flow",
                "Readability",
            ],
        },
        {
            "name": "Executive",
            "description": "Transform content for C-level and executive audience consumption. Focus on high-level strategic impact and ROI.",
            "focus_areas": [
                "Executive summary style",
                "ROI language",
                "Strategic vision",
                "Scalability",
            ],
        },
        {
            "name": "Recruiter",
            "description": "Optimize for what recruiters and hiring managers actually scan for. Emphasize keywords and role alignment.",
            "focus_areas": [
                "Keywords optimization",
                "ATS compatibility",
                "Role alignment",
                "Quick scanning",
            ],
        },
        {
            "name": "Concise",
            "description": "Maximize impact per word with executive-friendly brevity. Focus on essential information and key highlights.",
            "focus_areas": [
                "Bullet points",
                "Key highlights",
                "Essential information",
                "Brevity",
            ],
        },
        {
            "name": "Storytelling",
            "description": "Structure accomplishments as compelling narratives with clear problem-solution arcs that engage readers.",
            "focus_areas": [
                "Problem-solution structure",
                "Narrative arc",
                "Reader engagement",
                "Story flow",
            ],
        },
        {
            "name": "Corporate",
            "description": "Use formal business language and established corporate frameworks. Adopt professional business terminology.",
            "focus_areas": [
                "Business terminology",
                "Formal structure",
                "Corporate frameworks",
                "Professional tone",
            ],
        },
        {
            "name": "Buzzword",
            "description": "Deliberately incorporate trendy business and tech buzzwords to align with current industry language patterns.",
            "focus_areas": [
                "AI/ML terminology",
                "Industry buzzwords",
                "Trendy language",
                "Modern business speak",
            ],
        },
        {
            "name": "Humble",
            "description": "Downplay individual achievements while emphasizing team credit, learning moments, and collaborative success.",
            "focus_areas": [
                "Team attribution",
                "Learning moments",
                "Collaborative success",
                "Modest framing",
            ],
        },
        {
            "name": "Aggressive",
            "description": "Use strong, confident language about achievements and impact. Emphasize ownership and competitive advantage.",
            "focus_areas": [
                "Confident language",
                "Ownership claims",
                "Competitive framing",
                "Strong assertions",
            ],
        },
    ]

    # Model configurations
    models = [
        {"name": "gpt-5", "reasoning": {"effort": "high"}},
        {"name": "gpt-5-mini", "reasoning": {"effort": "medium"}},
        {"name": "gpt-5-nano", "reasoning": {"effort": "minimal"}},
        {"name": "o3", "reasoning": {"effort": "medium"}},
        {"name": "o4-mini", "reasoning": {"effort": "medium"}},
    ]

    # Generate cartesian product
    agents = []
    for strategy in strategies:
        for model in models:
            agent = {
                "name": f"{strategy['name']}_{model['name'].replace('-', '_')}",
                "description": strategy["description"],
                "model": model["name"],
                "reasoning": model["reasoning"],
                "focus_areas": strategy["focus_areas"],
            }
            agents.append(agent)

    return agents


def get_default_selected_agents() -> List[str]:
    """Get the list of agent names that should be selected by default."""
    return [
        "Metrics_gpt_5",
        "Leadership_gpt_5",
        "Strategic_gpt_5",
        "Technical_gpt_5",
        "Growth_gpt_5_mini",
        "Clarity_gpt_5_mini",
        "Impact_gpt_5",
        "Executive_gpt_5",
        "Recruiter_gpt_5_mini",
        "Storytelling_gpt_5_mini",
    ]


def load_default_config() -> Dict[str, Any]:
    """Load the default configuration."""
    return {
        "tournament": {
            "name": "AI Content Optimization Tournament",
            "num_rounds": 2,
            "matches_per_round": 10,
            "max_attempts_per_agent": 2,
        },
        "agents": generate_optimizer_agents(),
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
            "reasoning": {"effort": "minimal"},
            "min_truthfulness_threshold": 0.7,
        },
    }


def create_leaderboard_df(tournament: Tournament) -> pd.DataFrame:
    """Create a pandas DataFrame for the leaderboard."""
    leaderboard = tournament.get_leaderboard()

    data = []
    for submission, rating in leaderboard:
        agent_name = submission.agent_name or "Original"
        confidence = submission.metadata.get("confidence_level", "N/A")
        strategy_config = submission.metadata.get("strategy_config", {})
        strategy = strategy_config.get("name", agent_name)

        # Get model name from metadata
        model_name = submission.metadata.get("optimizer_model", "")
        if model_name and agent_name != "Original":
            display_name = f"{agent_name} ({model_name})"
        else:
            display_name = agent_name

        data.append(
            {
                "Rank": len(data) + 1,
                "Agent": display_name,
                "Strategy": strategy,
                "ELO Rating": f"{rating.rating:.0f}",
                "Matches": rating.matches_played,
                "Wins": rating.wins,
                "Losses": rating.losses,
                "Win Rate": (
                    f"{(rating.wins/rating.matches_played*100):.1f}%"
                    if rating.matches_played > 0
                    else "N/A"
                ),
                "Confidence": (
                    f"{confidence:.2f}"
                    if isinstance(confidence, (int, float))
                    else confidence
                ),
            }
        )

    return pd.DataFrame(data)


def create_confidence_intervals_chart(tournament: Tournament):
    """Create a confidence intervals chart showing ELO ratings with error bars."""
    leaderboard = tournament.get_leaderboard()

    # Prepare data for the chart
    models = []
    ratings = []
    lower_bounds = []
    upper_bounds = []
    matches_counts = []

    for submission, rating in leaderboard:
        agent_name = submission.agent_name or "Original"
        model_name = submission.metadata.get("optimizer_model", "")

        if model_name and agent_name != "Original":
            display_name = f"{agent_name}_{model_name.replace('-', '_')}"
        else:
            display_name = agent_name

        models.append(display_name)
        ratings.append(rating.rating)

        # Calculate confidence bounds
        ci = rating.confidence_interval
        lower_bounds.append(rating.rating - ci)
        upper_bounds.append(rating.rating + ci)
        matches_counts.append(rating.matches_played)

    # Sort by rating (descending)
    sorted_data = sorted(
        zip(models, ratings, lower_bounds, upper_bounds, matches_counts),
        key=lambda x: x[1],
        reverse=True,
    )
    models, ratings, lower_bounds, upper_bounds, matches_counts = zip(*sorted_data)

    # Create the plotly figure
    fig = go.Figure()

    # Add error bars and points
    fig.add_trace(
        go.Scatter(
            x=list(range(len(models))),
            y=ratings,
            mode="markers",
            marker=dict(size=8, color="#1f77b4"),
            error_y=dict(
                type="data",
                array=[r - l for r, l in zip(ratings, lower_bounds)],
                arrayminus=[u - r for u, r in zip(upper_bounds, ratings)],
                visible=True,
                color="#1f77b4",
                thickness=2,
                width=4,
            ),
            name="ELO Rating",
            hovertemplate="<b>%{text}</b><br>ELO Rating: %{y:.0f}<br>Matches: %{customdata}<extra></extra>",
            text=models,
            customdata=matches_counts,
        )
    )

    # Update layout to match the example style
    fig.update_layout(
        title={
            "text": "Confidence Intervals on Model Strength (via Bootstrapping)",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18, "color": "#333"},
        },
        xaxis=dict(
            title="Model",
            tickangle=-45,
            tickmode="array",
            tickvals=list(range(len(models))),
            ticktext=models,
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=1,
        ),
        yaxis=dict(
            title="ELO Rating", showgrid=True, gridcolor="lightgray", gridwidth=1
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        width=800,
        height=500,
        margin=dict(l=60, r=60, t=80, b=120),
    )

    return fig


def display_submission_details(submission: Submission):
    """Display detailed information about a submission."""
    agent_name = submission.agent_name or "Original"
    model_name = submission.metadata.get("optimizer_model", "")

    if model_name and agent_name != "Original":
        display_name = f"{agent_name} ({model_name})"
    else:
        display_name = agent_name

    st.subheader(f"🏆 {display_name}")

    # Show the optimized content as markdown
    st.markdown("**Optimized Content:**")
    st.markdown(submission.content)


def show_tournament_winner(tournament):
    """Display the tournament winner with key stats only."""
    winner_analysis = get_tournament_winner_analysis(tournament)

    st.subheader("Tournament Winner")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            f"**{(winner_analysis.submission.agent_name or 'Original').replace('_', ' ').title()}**"
        )

        with st.expander("View Content"):
            with st.container():
                st.markdown(winner_analysis.submission.content)

    with col2:
        # Get original ELO for comparison
        original_elo = tournament.ratings.get("original")
        if original_elo and winner_analysis.submission.id != "original":
            elo_display = (
                f"{winner_analysis.final_elo:.0f} vs {original_elo.rating:.0f}"
            )
            elo_delta = f"+{winner_analysis.final_elo - original_elo.rating:.0f}"
        else:
            elo_display = f"{winner_analysis.final_elo:.0f}"
            elo_delta = None

        st.metric("ELO Rating", elo_display, delta=elo_delta)
        st.metric("Record", f"{winner_analysis.wins}–{winner_analysis.losses}")
        st.metric("Win Rate", f"{winner_analysis.win_rate:.1%}")


def show_baseline_analysis(completed_tournament):
    """Show baseline vs optimized comparison."""

    if not any(s.id == "original" for s in completed_tournament.submissions):
        return

    analysis = analyze_baseline_performance(completed_tournament)

    st.subheader("Original vs Optimized Performance")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("AI Win Rate", f"{analysis.optimized_win_rate:.1%}")
    with col2:
        st.metric("Head-to-Head", f"{analysis.optimized_wins}–{analysis.baseline_wins}")
    with col3:
        st.metric(
            "Judge Confidence", f"{analysis.avg_confidence_when_optimized_wins:.1%}"
        )


def show_content_comparison(original_submission, winner_submission):
    """Show original vs winner content side-by-side."""

    st.subheader("Content Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original**")
        with st.container():
            st.markdown(original_submission.content)

    with col2:
        st.markdown(
            f"**{(winner_submission.agent_name or 'Winner').replace('_', ' ').title()}**"
        )
        with st.container():
            st.markdown(winner_submission.content)


def display_tournament_results(completed_tournament):
    """Clean results display focused on key insights."""

    # Winner
    show_tournament_winner(completed_tournament)

    st.divider()

    # Content comparison
    original = next(
        (s for s in completed_tournament.submissions if s.id == "original"), None
    )
    winner = get_tournament_winner_analysis(completed_tournament).submission

    if original and winner.id != "original":
        show_content_comparison(original, winner)

    # Full rankings tucked away
    with st.expander("All Results"):
        st.header("🏆 Leaderboard")
        leaderboard_df = create_leaderboard_df(completed_tournament)
        st.dataframe(leaderboard_df, use_container_width=True)

        # Add confidence intervals chart
        if len(completed_tournament.submissions) > 1:
            st.subheader("📊 Model Strength Confidence Intervals")
            confidence_chart = create_confidence_intervals_chart(completed_tournament)
            st.plotly_chart(confidence_chart, use_container_width=True)


def get_example_data():
    """Get example data for seeding inputs."""
    return {
        "self_review": "This year I've taken on increasingly complex technical challenges while growing my leadership responsibilities within the team. I led the development of our new user authentication system, which involved designing the database schema, implementing OAuth2 integration, and creating comprehensive unit tests. The project took 4 weeks and resulted in a 40% reduction in security-related support tickets.\n\nI've been actively mentoring two junior developers, conducting weekly 1:1s and helping them navigate code reviews. I established a best practices guide for our React components that has been adopted across our entire frontend team. I also drove the decision to migrate from REST to GraphQL for our user dashboard API, which improved page load times by 25%.\n\nI completed 67 code reviews this year and resolved 23 production bugs, including a critical memory leak that was affecting our data processing pipeline. I've been staying current with industry trends by attending React Conference and completing a cloud architecture certification. I also presented our team's testing strategy to the broader engineering organization.",
        "resume": """**John Smith**  
Senior Software Engineer  
john.smith@email.com | (555) 123-4567 | San Francisco, CA  
LinkedIn: linkedin.com/in/johnsmith | GitHub: github.com/johnsmith

**PROFESSIONAL SUMMARY**
Senior Software Engineer with 6+ years of experience building scalable web applications and leading high-impact projects. Expert in full-stack development with React, Node.js, and AWS. Proven track record of delivering user-facing features to millions of users while mentoring junior developers and driving technical initiatives.

**TECHNICAL SKILLS**
• **Languages:** JavaScript, TypeScript, Python, Java, SQL
• **Frontend:** React, Redux, Vue.js, HTML5, CSS3, Webpack, Jest
• **Backend:** Node.js, Express, Django, PostgreSQL, MongoDB, Redis
• **Cloud & DevOps:** AWS (EC2, S3, Lambda), Docker, Kubernetes, CI/CD
• **Tools:** Git, Jira, Figma, DataDog, New Relic

**PROFESSIONAL EXPERIENCE**

**Senior Software Engineer** | TechFlow Inc. | Jan 2022 - Present
• Led development of real-time messaging platform serving 2M+ daily active users
• Architected and implemented GraphQL API reducing page load times by 35%
• Mentored 4 junior engineers and established code review best practices across 15-person team
• Designed automated testing strategy increasing test coverage from 60% to 90%
• Built responsive dashboard using React and D3.js for enterprise analytics product

**Software Engineer** | DataVision Corp | Jun 2019 - Dec 2021
• Developed customer-facing web application processing 100K+ daily transactions
• Optimized database queries reducing API response times by 50%
• Implemented OAuth2 authentication system improving security posture
• Collaborated with product team to ship 12 major features including payment processing
• Created deployment pipeline using Docker and AWS reducing deployment time by 40%

**Junior Software Engineer** | StartupXYZ | Aug 2018 - May 2019
• Built RESTful APIs using Node.js and Express serving mobile application
• Implemented responsive UI components using React and CSS3
• Fixed 50+ bugs and contributed to 20+ feature releases
• Participated in agile development process and daily standups

**PROJECTS**

**Personal Finance Tracker** | github.com/sarahchen/finance-tracker
• Full-stack web app built with React, Node.js, and PostgreSQL
• Integrated with Plaid API for bank account connectivity
• Implemented data visualization using Chart.js and custom algorithms

**EDUCATION**
**Bachelor of Science, Computer Science** | UC Berkeley | 2018
• Relevant Coursework: Data Structures, Algorithms, Database Systems, Software Engineering
• Activities: Women in Tech Club Secretary, Programming Contest Participant

**CERTIFICATIONS & ACHIEVEMENTS**
• AWS Certified Solutions Architect Associate (2023)
• React Conference Speaker: "Building Scalable Component Libraries" (2023)
• Company Hackathon Winner: Employee Recognition Platform (2022)""",
        "career_ladder": """Career Ladder - Senior Engineer Level:

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
        "job_requirements": """Senior Software Engineer - Requirements:

Technical Skills:
- 5+ years experience in software development
- Proficiency in modern web technologies (React, Node.js, Python)
- Experience with cloud platforms (AWS, Azure, GCP)
- Strong understanding of databases and API design

Leadership & Communication:
- Demonstrated ability to mentor junior developers
- Excellent written and verbal communication skills
- Experience leading technical projects and initiatives
- Collaborative team player with cross-functional experience

Problem Solving:
- Strong analytical and problem-solving abilities
- Experience with system design and architecture
- Ability to optimize performance and scalability
- Track record of delivering high-quality software solutions""",
        "self_review_background": "GitHub Activity: 247 commits this year, 12 pull requests merged, contributed to 3 major repositories (auth-service, dashboard-ui, data-pipeline). \n\nJira Tickets: Resolved TECH-1234 (OAuth2 integration), TECH-1456 (memory leak fix), TECH-1789 (GraphQL migration), TECH-2001 (React component refactor). Total: 23 bugs closed, 8 features delivered.\n\nMeetings & Presentations: Weekly 1:1s with Jake and Maria (junior developers), presented at All-Hands on 3/15/24 about testing strategy, attended React Conference 2024, completed AWS Solutions Architect certification in September.\n\nCode Review Stats: Reviewed 67 PRs, average review turnaround 4.2 hours, provided mentoring feedback in 45+ reviews. Established team coding standards document adopted by frontend team.",
        "resume_background": 'Employment Verification: TechFlow Inc (2022-present, Manager: David Kim, +1-555-0123), DataVision Corp (2019-2021, Manager: Lisa Zhang), StartupXYZ (2018-2019, Manager: Mike Johnson).\n\nEducation Verification: UC Berkeley BS Computer Science 2018, GPA 3.7/4.0, Dean\'s List 3 semesters.\n\nCertifications: AWS Certified Solutions Architect Associate #AWS-ASA-12345 (issued Sep 2023, expires Sep 2026).\n\nPublic Speaking: React Conference 2023 speaker "Building Scalable Component Libraries" (800+ attendees), TechFlow engineering blog contributor (5 published articles).\n\nOpen Source: Personal Finance Tracker (github.com/johnsmith/finance-tracker, 234 stars, 45 forks), contributor to React ecosystem packages.',
    }


def main():
    st.set_page_config(page_title="AI Content Optimizer", page_icon="🚀", layout="wide")

    st.title("AI Judge Preference Demo")

    st.markdown(
        """
    **What if AI models were judging your performance review or resume?** This system reveals the hidden biases and preferences of AI judges by running competitive tournaments between different writing styles and optimization strategies.

    ⚠️ **Research Tool**: This explores how AI models evaluate professional content, not career advice. It shows which specific wording, metrics, and presentation styles make AI judges rank one version higher than another.
    
    * Data you provide is processed by OpenAI models but not stored by this demo
    * The system uses competitive ELO tournaments to identify optimization strategies that AI judges favor
    * This explores AI bias patterns rather than providing definitive career advice
    * The demo takes 5-15 minutes to run depending on settings and uses a variety of strategies and models including GPT-5
    * [View the code on GitHub](https://github.com/sshh12/perf-review)
    """
    )

    st.write(
        "Discover what kinds of professional content AI judges prefer through competitive tournaments"
    )

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error(
            "⚠️ OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
        )
        st.stop()

    # Sidebar for configuration (shared across both modes)
    st.sidebar.header("⚙️ Configuration")

    # Load default config
    config = load_default_config()

    # Tournament settings
    st.sidebar.subheader("Tournament Settings")
    num_rounds = st.sidebar.slider(
        "Number of Rounds",
        1,
        5,
        config["tournament"]["num_rounds"],
        help="More rounds = better ELO accuracy but longer runtime. 1-2 rounds for quick tests, 3-5 for comprehensive analysis.",
    )
    matches_per_round = st.sidebar.slider(
        "Matches per Round",
        5,
        50,
        config["tournament"]["matches_per_round"],
        help="More matches = more data points and stable rankings. Too few (<10) = unreliable results. Too many (>30) = diminishing returns vs time cost.",
    )
    num_agents = st.sidebar.slider(
        "Number of Optimizer Agents",
        1,
        len(config["agents"]),
        10,
        help=f"Randomly select agents from {len(config['agents'])} available strategies. More agents = more diverse optimization approaches but longer runtime.",
    )

    # Truth agent is always enabled with default settings
    use_truth_agent = True
    truth_threshold = 0.7

    # Mode selection tabs
    tab1, tab2 = st.tabs(["📝 Performance Review Mode", "💼 Resume/Recruiting Mode"])

    with tab1:
        run_optimization_interface(
            "performance_review",
            config,
            num_rounds,
            matches_per_round,
            num_agents,
            use_truth_agent,
            truth_threshold,
        )

    with tab2:
        run_optimization_interface(
            "resume",
            config,
            num_rounds,
            matches_per_round,
            num_agents,
            use_truth_agent,
            truth_threshold,
        )


def run_optimization_interface(
    mode: str,
    config: Dict[str, Any],
    num_rounds: int,
    matches_per_round: int,
    num_agents: int,
    use_truth_agent: bool,
    truth_threshold: float,
):
    """Run the optimization interface for the specified mode."""
    if mode == "performance_review":
        mode_config = {
            "title": "Performance Review Optimization",
            "content_label": "Enter your performance review to optimize:",
            "content_example_key": "self_review",
            "rubric_label": "Enter career ladder or evaluation criteria:",
            "rubric_example_key": "career_ladder",
            "background_label": "Enter supporting facts and achievements:",
            "background_example_key": "self_review_background",
        }
    else:  # resume mode
        mode_config = {
            "title": "Resume/Recruiting Optimization",
            "content_label": "Enter your resume to optimize:",
            "content_example_key": "resume",
            "rubric_label": "Enter job requirements or evaluation criteria:",
            "rubric_example_key": "job_requirements",
            "background_label": "Enter supporting facts and employment details:",
            "background_example_key": "resume_background",
        }

    st.header(f"🎯 {mode_config['title']}")

    st.info(
        f"""
    **How the tournament works:** {num_agents} AI agents optimize your content using different strategies (Metrics, Leadership, Technical, Executive, etc.) over {num_rounds} rounds. 
    Each round, agents create improved versions which then compete in {matches_per_round} head-to-head comparisons judged by a 3-model panel (GPT-5, O3, O4-Mini) using ELO ratings. 
    The system reveals which optimization approaches AI judges prefer and how they compare to your original content.
    """
    )

    # Update config with mode-specific title
    config["tournament"]["name"] = mode_config["title"]

    # Randomly sample agents based on slider value
    if f"selected_agents_{mode}_{num_agents}" not in st.session_state:
        # Set random seed for reproducibility during the session
        random.seed(42)
        st.session_state[f"selected_agents_{mode}_{num_agents}"] = random.sample(
            [agent["name"] for agent in config["agents"]], num_agents
        )

    # Get selected agents and create lookup
    selected_agents = st.session_state[f"selected_agents_{mode}_{num_agents}"]
    available_agents = {agent["name"]: agent for agent in config["agents"]}

    # Get example data
    examples = get_example_data()

    # Initialize session state for text areas (mode-specific)
    if f"original_content_{mode}" not in st.session_state:
        st.session_state[f"original_content_{mode}"] = examples[
            mode_config["content_example_key"]
        ]
    if f"rubric_content_{mode}" not in st.session_state:
        st.session_state[f"rubric_content_{mode}"] = examples[
            mode_config["rubric_example_key"]
        ]
    if f"background_content_{mode}" not in st.session_state:
        st.session_state[f"background_content_{mode}"] = examples[
            mode_config["background_example_key"]
        ]

    # Single example loading button
    example_label = (
        "📋 Load Example Data"
        if mode == "performance_review"
        else "📄 Load Resume Example"
    )
    if st.button(
        example_label,
        help=f"Load all example data for {mode.replace('_', ' ')}",
        key=f"load_example_{mode}",
    ):
        st.session_state[f"original_content_{mode}"] = examples[
            mode_config["content_example_key"]
        ]
        st.session_state[f"rubric_content_{mode}"] = examples[
            mode_config["rubric_example_key"]
        ]
        st.session_state[f"background_content_{mode}"] = examples[
            mode_config["background_example_key"]
        ]
        st.rerun()

    # Main inputs
    col1, col2 = st.columns(2)

    with col1:
        st.header("📝 Content to Optimize")
        st.caption(
            "Your original content serves as the baseline competitor in every tournament match. AI agents will create optimized versions to compete against it head-to-head."
        )

        original_content = st.text_area(
            mode_config["content_label"],
            value=st.session_state[f"original_content_{mode}"],
            height=200,
            key=f"original_text_{mode}",
        )
        # Update session state when text changes
        if original_content != st.session_state[f"original_content_{mode}"]:
            st.session_state[f"original_content_{mode}"] = original_content

    with col2:
        st.header("📋 Evaluation Criteria")
        st.caption(
            "AI judges use these criteria to determine winners in each head-to-head comparison. This defines what 'better' means in tournament matches."
        )

        rubric = st.text_area(
            mode_config["rubric_label"],
            value=st.session_state[f"rubric_content_{mode}"],
            height=200,
            key=f"rubric_text_{mode}",
        )
        # Update session state when text changes
        if rubric != st.session_state[f"rubric_content_{mode}"]:
            st.session_state[f"rubric_content_{mode}"] = rubric

    # Background/Ground Truth
    st.header("🏗️ Background Information")
    st.caption(
        "Factual data that truth agents use to verify optimized content for accuracy. Helps prevent AI from fabricating claims while ensuring competitive versions remain truthful."
    )

    background = st.text_area(
        mode_config["background_label"],
        value=st.session_state[f"background_content_{mode}"],
        height=100,
        key=f"background_text_{mode}",
    )
    # Update session state when text changes
    if background != st.session_state[f"background_content_{mode}"]:
        st.session_state[f"background_content_{mode}"] = background

    # Run Tournament Button
    if st.button("🚀 Run AI Optimization", type="primary", key=f"run_btn_{mode}"):
        if not original_content.strip():
            st.error(f"Please enter your {mode.replace('_', ' ')}")
            return

        if not rubric.strip():
            st.error("Please enter evaluation criteria")
            return

        if not selected_agents:
            st.error("Please select at least one optimizer agent")
            return

        # Update config with user selections
        config["tournament"]["num_rounds"] = num_rounds
        config["tournament"]["matches_per_round"] = matches_per_round
        config["agents"] = [available_agents[name] for name in selected_agents]
        config["truth_agent"]["enabled"] = use_truth_agent
        config["truth_agent"]["min_truthfulness_threshold"] = truth_threshold

        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Initialize components
            status_text.text("🔧 Initializing AI components...")
            progress_bar.progress(10)

            openai_client = OpenAIClient(model=config["judge"]["model"])

            # Initialize judge system (single or multi-judge)
            if config["judge"].get("multi_judge", {}).get("enabled", False):
                # Create multiple judges
                judge_models = config["judge"]["multi_judge"]["models"]
                judges = []
                for model in judge_models:
                    judge = LLMJudge(
                        openai_client,
                        model=model,
                        reasoning=config["judge"].get("reasoning"),
                    )
                    judges.append(judge)
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

            # Create optimizer agents
            status_text.text("🤖 Creating optimizer agents...")
            progress_bar.progress(20)

            optimizers = OptimizerFactory.create_agents_from_config(
                openai_client=openai_client,
                strategies=config["agents"],
                default_model="gpt-4.1",
            )

            # Create original submission
            original_submission = Submission(
                id="original", content=original_content, agent_name=None
            )

            # Create tournament
            status_text.text("🏟️ Setting up tournament...")
            progress_bar.progress(30)

            tournament = arena.create_tournament(
                name=config["tournament"]["name"],
                original_submission=original_submission,
                rubric=rubric,
            )

            # Run tournament with detailed progress updates
            total_steps = (
                num_rounds * 2 + 2
            )  # 2 steps per round (optimize + matches) + setup + final
            current_step = 0

            # Calculate estimated total time
            estimated_total_minutes = (
                num_rounds * 6 + 3
            )  # ~6 min per round + 3 min final
            status_text.text(
                f"🚀 Starting optimization tournament... (Estimated {estimated_total_minutes} minutes remaining - grab a coffee! ☕)"
            )
            progress_bar.progress(40)

            # We need to implement our own tournament loop for granular updates
            for round_num in range(1, num_rounds + 1):
                current_step += 1
                progress = (
                    40 + (current_step / total_steps) * 40
                )  # Use 40-80% of progress bar

                # Optimization round
                remaining_rounds = num_rounds - round_num + 1
                estimated_remaining_minutes = (
                    remaining_rounds * 6 + 3
                )  # 6 min per remaining round + 3 min final
                status_text.text(
                    f"⚙️ Round {round_num}/{num_rounds}: Optimizing with {len(selected_agents)} agents... (Est. {estimated_remaining_minutes} min remaining)"
                )
                progress_bar.progress(int(progress))

                # Reset agent attempts for this round
                for optimizer in optimizers:
                    optimizer.reset_attempts()

                # Run optimization round
                new_submissions = arena.run_optimization_round(
                    tournament,
                    optimizers,
                    config["tournament"]["max_attempts_per_agent"],
                    30,  # optimization_batch_size
                )

                current_step += 1
                progress = 40 + (current_step / total_steps) * 40

                # Matches round
                remaining_rounds_after_matches = num_rounds - round_num
                estimated_remaining_minutes = (
                    remaining_rounds_after_matches * 6 + 3
                )  # remaining rounds + final
                status_text.text(
                    f"⚖️ Round {round_num}/{num_rounds}: Running {matches_per_round} matches... (Est. {estimated_remaining_minutes} min remaining)"
                )
                progress_bar.progress(int(progress))

                # Run tournament matches
                new_matches = arena.run_tournament_matches(
                    tournament,
                    num_matches=matches_per_round,
                    match_strategy="swiss",
                    batch_size=30,  # match_batch_size
                )

                status_text.text(
                    f"✅ Round {round_num} complete: {len(new_submissions)} new submissions, {len(new_matches)} matches"
                )
                time.sleep(0.5)  # Brief pause so users can see the completion message

            # Final comprehensive match round
            current_step += 1
            progress = 40 + (current_step / total_steps) * 40
            status_text.text(
                "🏁 Running final comprehensive matches... (Est. 3 min remaining)"
            )
            progress_bar.progress(int(progress))
            time.sleep(0.3)

            final_matches = arena.run_tournament_matches(
                tournament, num_matches=50, match_strategy="swiss", batch_size=30
            )

            tournament.completed_at = tournament.created_at
            completed_tournament = tournament

            # Truth agent filtering (if enabled)
            if use_truth_agent and background.strip():
                submissions_to_verify = [
                    s for s in completed_tournament.submissions if s.id != "original"
                ]
                status_text.text(
                    f"🔍 Fact-checking {len(submissions_to_verify)} submissions..."
                )
                progress_bar.progress(80)
                time.sleep(0.3)

                truth_agent = TruthAgent(
                    openai_client,
                    model=config["truth_agent"]["model"],
                    reasoning=config["truth_agent"].get("reasoning"),
                )
                truth_guarded = TruthGuardedArena(truth_agent, truth_threshold)

                # Filter submissions (excluding original) with parallel processing
                approved_submissions, verification_results = (
                    truth_guarded.filter_submissions(
                        submissions_to_verify, background, rubric, batch_size=30
                    )
                )

                # Show verification results
                blocked_count = len(submissions_to_verify) - len(approved_submissions)
                if blocked_count > 0:
                    status_text.text(
                        f"🛡️ Truth verification complete: {blocked_count} submissions blocked"
                    )
                    st.warning(
                        f"⚠️ Truth agent blocked {blocked_count} submissions for potential fabrications"
                    )
                else:
                    status_text.text(
                        f"✅ Truth verification complete: All {len(approved_submissions)} submissions approved"
                    )
                time.sleep(0.5)

            progress_bar.progress(100)
            status_text.text("✅ Tournament completed!")

            # Display results
            st.success(
                f"🎉 Tournament completed! {len(completed_tournament.submissions)} submissions, {len(completed_tournament.matches)} matches"
            )

            # New streamlined results display
            display_tournament_results(completed_tournament)

            # Raw results (expandable)
            st.header("📋 Raw Results")
            with st.expander("View Raw Tournament Data (JSON)", expanded=False):
                # Create a temporary file to save results
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as f:
                    completed_tournament.save_to_file(f.name)
                    with open(f.name, "r") as rf:
                        raw_data = json.load(rf)
                    os.unlink(f.name)

                st.json(raw_data)

                # Download button for raw results
                st.download_button(
                    label="📥 Download Raw Results",
                    data=json.dumps(raw_data, indent=2),
                    file_name="tournament_results.json",
                    mime="application/json",
                )

        except Exception as e:
            error_msg = str(e)
            progress_bar.empty()
            status_text.empty()

            # Make error messages more user-friendly
            if "No module named" in error_msg:
                st.error(
                    "❌ Missing required dependency. Please check your installation."
                )
            elif "API key" in error_msg.lower():
                st.error(
                    "❌ OpenAI API key issue. Please check your OPENAI_API_KEY environment variable."
                )
            elif "rate limit" in error_msg.lower():
                st.error(
                    "❌ OpenAI API rate limit reached. Please wait a moment and try again."
                )
            elif "timeout" in error_msg.lower():
                st.error(
                    "❌ Request timed out. The AI models may be experiencing high demand. Please try again."
                )
            elif len(error_msg) == 36 and "-" in error_msg:  # Likely a UUID error
                st.error(
                    "❌ Internal processing error. This may be due to a configuration issue with the selected agents. Please try running again."
                )
            else:
                st.error(f"❌ An error occurred during optimization: {error_msg}")

            # Show detailed error for debugging if helpful
            with st.expander("🔧 Technical Details (for debugging)", expanded=False):
                st.code(f"Error: {type(e).__name__}: {error_msg}")
                if hasattr(e, "__traceback__"):
                    import traceback

                    st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
