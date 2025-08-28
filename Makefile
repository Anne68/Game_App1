.PHONY: run test lint format cov
run:
	uvicorn api_games_plus:app --reload --port 8000

test:
	pytest

lint:
	ruff check . && black --check .

format:
	black .

cov:
	pytest --cov=.: --cov-report=term-missing
