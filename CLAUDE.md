## Development Workflow

- Run python files with `source env.sh && PYTHONPATH=. python ...`
- Use existing venv and env.sh for API keys
- DO NOT CREATE FAKE TEST DATA, THINGS MUST RUN E2E.
- Run `black .` to format code
- Use docs/openai-api.md for how to use the openai responses api
- Use docs/DESIGN.md to understand what we are building