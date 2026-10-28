install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src/ tests/

run:
	uvicorn arbiter.api:app --app-dir src --env-file .env --reload --port 8000

dashboard:
	streamlit run ui/dashboard.py --server.port 8501
