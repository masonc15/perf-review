# 🚀 AI Judge Preference Demo

**What if AI models were judging your performance review or resume?** This system reveals the hidden biases and preferences of AI judges by running competitive tournaments between different writing styles and optimization strategies.

⚠️ **Research Tool**: This explores how AI models evaluate professional content, not career advice. It shows which specific wording, metrics, and presentation styles make AI judges rank one version higher than another.

* Data you provide is processed by OpenAI models but not stored by this demo
* The system uses competitive ELO tournaments to identify optimization strategies that AI judges favor
* This explores AI bias patterns rather than providing definitive career advice
* The demo takes 5-15 minutes to run depending on settings and uses a variety of strategies and models including GPT-5
* [View the code on GitHub](https://github.com/sshh12/perf-review)

<img width="1624" height="614" alt="Screenshot 2025-08-09 at 5 15 28 PM" src="https://github.com/user-attachments/assets/67d94640-4dc4-49a9-966f-a733f3a558c5" />


## Features

- **Multi-Strategy Optimization**: 6 different AI agents with specialized optimization strategies
- **ELO Tournament System**: Competitive ranking to find the best optimized version
- **Truth Agent**: Prevents fabrications by verifying against ground truth data  
- **Streamlit Web Interface**: Easy-to-use web app for running optimizations
- **Multi-Judge System**: Optional consensus-based judging with multiple models
- **Configurable Strategies**: Fully customizable agent strategies and focus areas

## Usage

### Prerequisites

- Python 3.8+ 
- OpenAI API key (set as `OPENAI_API_KEY` environment variable)

### Installation & Setup

```bash
# Clone the repository
git clone https://github.com/sshh12/perf-review
cd perf-review

# Install dependencies
pip install -r requirements.txt

# Set up environment (includes OpenAI API key)
source env.sh
```

### Running the Web Interface

```bash
# Start the Streamlit app
source env.sh && streamlit run streamlit_app.py
```

Then open your browser to the provided URL (typically http://localhost:8501)

The web interface provides two modes:
- **📝 Performance Review Mode**: Optimize performance reviews against career ladders
- **💼 Resume/Recruiting Mode**: Optimize resumes against job requirements

### Running Command Line Version

```bash
# Set up environment and run
source env.sh && PYTHONPATH=. python main.py --rounds 2 --output results.json

# Create sample configuration
source env.sh && PYTHONPATH=. python main.py --create-sample-config

# Format code (development)
black .
```

## How It Works

1. **Input**: Provide your original content, evaluation criteria, and optional background data
2. **Optimization**: AI agents create optimized versions using different strategies and approaches
3. **Tournament**: Optimized versions compete in ELO-ranked matches judged by multiple AI models
4. **Results**: Get ranked leaderboard showing which optimization approaches AI judges prefer