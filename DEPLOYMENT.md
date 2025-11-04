# Deployment Guide

## Quick Deploy to Railway

### Prerequisites
- Railway account at [railway.app](https://railway.app)
- OpenAI API key

### Option 1: Deploy via Railway Dashboard (Recommended - 2 minutes)

1. **Go to [railway.app](https://railway.app)** and sign in

2. **Create New Project**:
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose: `masonc15/perf-review`
   - Branch: `main`

3. **Add Environment Variable**:
   - Navigate to your project → "Variables" tab
   - Click "+ New Variable"
   - Add:
     - Name: `OPENAI_API_KEY`
     - Value: `your-openai-api-key-here`

4. **Deploy**:
   - Railway automatically detects the Dockerfile
   - Builds and deploys the application
   - Provides a public URL (e.g., `https://perf-review-production.up.railway.app`)

5. **Access Your App**:
   - Click "Open App" or visit the provided URL
   - Wait 1-2 minutes for initial deployment

---

### Option 2: Deploy via Railway CLI

```bash
# 1. Clone the repository
git clone https://github.com/masonc15/perf-review.git
cd perf-review

# 2. Install Railway CLI
curl -fsSL https://railway.app/install.sh | sh

# 3. Login to Railway
railway login

# 4. Initialize project
railway init

# 5. Set environment variable
railway variables set OPENAI_API_KEY=your-openai-api-key-here

# 6. Deploy
railway up

# 7. Generate public domain
railway domain

# 8. Open in browser
railway open
```

---

## Local Development

### Setup

```bash
# 1. Clone and navigate
git clone https://github.com/masonc15/perf-review.git
cd perf-review

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
echo 'export OPENAI_API_KEY="your-key-here"' > env.sh
source env.sh
```

### Run Web Interface

```bash
source env.sh && streamlit run streamlit_app.py
```

Access at: http://localhost:8501

### Run CLI Version

```bash
# Basic run
source env.sh && PYTHONPATH=. python main.py

# Custom configuration
source env.sh && PYTHONPATH=. python main.py --rounds 2 --agents 4 --output results.json

# Generate sample config
source env.sh && PYTHONPATH=. python main.py --create-sample-config
```

---

## Docker Deployment

### Build and Run Locally

```bash
# Build image
docker build -t perf-review .

# Run container
docker run -p 8501:8501 \
  -e OPENAI_API_KEY="your-key-here" \
  perf-review
```

Access at: http://localhost:8501

---

## Environment Variables

Required:
- **OPENAI_API_KEY** - Your OpenAI API key

Optional:
- **PORT** - Server port (default: 8501, auto-set by Railway)

---

## Repository Structure

```
perf-review/
├── main.py                 # CLI entry point
├── streamlit_app.py        # Web UI entry point
├── Dockerfile             # Docker configuration
├── railway.toml           # Railway deployment config
├── requirements.txt       # Python dependencies
├── entrypoint.sh          # Docker startup script
├── perf_review/           # Main application package
└── docs/                  # Documentation
```

---

## Troubleshooting

### Railway Deployment Issues

**Build fails:**
- Check that `OPENAI_API_KEY` is set in Variables tab
- Verify Dockerfile is present in repository
- Check Railway logs for specific error messages

**App doesn't start:**
- Ensure port 8501 is exposed in Dockerfile
- Verify entrypoint.sh has execute permissions
- Check that Streamlit is running via logs

**API errors:**
- Verify OpenAI API key is valid
- Check API key has sufficient credits
- Ensure key has access to required models (GPT-5, o3, o4-mini)

### Local Development Issues

**Import errors:**
- Ensure you're using `PYTHONPATH=.` when running scripts
- Activate virtual environment: `source venv/bin/activate`

**API rate limits:**
- Reduce number of agents or rounds
- Use smaller batch sizes in configuration

---

## Support

- **GitHub Issues**: [github.com/sshh12/perf-review/issues](https://github.com/sshh12/perf-review/issues)
- **Documentation**: See README.md and docs/ folder
