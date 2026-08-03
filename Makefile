.PHONY: install hooks up down test lint fmt eval db-init db-seed

install:
	uv sync --extra dev

hooks:
	uv run pre-commit install

up:
	docker compose -f docker/compose.yml up -d

down:
	docker compose -f docker/compose.yml down

test:
	uv run pytest -m "not integration"

test-all:
	uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check .

fmt:
	uv run ruff format . && uv run ruff check --fix .

eval:
	uv run python -m parking_bot.eval.harness

db-init:
	uv run python -m parking_bot.db.init_db

db-seed:
	uv run python -m parking_bot.db.seed
