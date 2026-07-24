.PHONY: install hooks up down test lint fmt eval

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
