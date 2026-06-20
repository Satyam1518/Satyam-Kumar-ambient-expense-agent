.PHONY: install playground test lint format run generate-traces grade

install:
	uv tool install google-agents-cli
	uvx google-agents-cli install

playground:
	uvx google-agents-cli playground

run:
	uv run python -m expense_agent.agent_runtime_app

test:
	uv run pytest tests/unit tests/integration

lint:
	uv run ruff check .

format:
	uv run ruff format .

generate-traces:
	uv run python tests/eval/generate_traces.py

grade:
	uv run python tests/eval/grade_traces.py
