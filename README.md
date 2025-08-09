# 🚀 AI Performance Review Optimizer

Transform your performance reviews with AI-powered optimization strategies using competitive ELO tournaments.

## Features

- **Multi-Strategy Optimization**: 6 different AI agents with specialized optimization strategies
- **ELO Tournament System**: Competitive ranking to find the best optimized version
- **Truth Agent**: Prevents fabrications by verifying against ground truth data  
- **Streamlit Web Interface**: Easy-to-use web app for running optimizations
- **Multi-Judge System**: Optional consensus-based judging with multiple models
- **Configurable Strategies**: Fully customizable agent strategies and focus areas

## Quick Start

### 1. Setup Environment

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Streamlit App

```bash
streamlit run streamlit_app.py
```

Then open your browser to the provided URL (typically http://localhost:8501)

### 3. Run Command Line Version

```bash
# Create sample config
python main.py --create-sample-config

# Run tournament
python main.py --rounds 2 --output results.json
```

## How It Works

1. **Input**: Provide your original performance review, career ladder/rubric, and optional background data
2. **Optimization**: AI agents create optimized versions using different strategies:
   - **MetricsOptimizer**: Quantifies impact with specific numbers and data
   - **LeadershipOptimizer**: Emphasizes mentoring and team influence  
   - **StrategicOptimizer**: Frames work in business context and strategic value
   - **GrowthOptimizer**: Highlights learning and adaptability
   - **ClarityOptimizer**: Improves structure and professional presentation
   - **ImpactOptimizer**: Focuses on customer and market impact
3. **Tournament**: Optimized reviews compete in ELO-ranked matches judged by LLM
4. **Results**: Get ranked leaderboard with detailed explanations and raw data

## Configuration

The system is highly configurable via JSON. Key settings:

- **Tournament**: Rounds, matches per round, max attempts per agent
- **Agents**: List of optimization strategies with models and focus areas  
- **Judge**: Model selection and multi-judge consensus options
- **Truth Agent**: Factual verification against ground truth data

## Architecture

- `perf_review/` - Core system modules
  - `models.py` - Data structures and tournament state
  - `optimizer.py` - AI optimization agents and strategies
  - `judge.py` - LLM judges for pairwise comparisons  
  - `truth_agent.py` - Factual verification system
  - `arena.py` - Tournament orchestration and ELO system
- `main.py` - Command line interface
- `streamlit_app.py` - Web interface
- `sample_config.json` - Example configuration

## Example Results

The system transforms vague reviews like:
> "I worked on several projects including new features and bug fixes..."

Into optimized versions like:
> "Led delivery of authentication system (450 lines, 4-week project with 3-person team), resulting in 40% reduction in security incidents. Mentored 2 junior engineers on testing practices, completed 45 code reviews..."

## Requirements

- Python 3.8+
- OpenAI API key
- Dependencies in `requirements.txt`