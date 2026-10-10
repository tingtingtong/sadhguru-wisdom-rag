.PHONY: install serve scrape ingest test lint docker-up docker-down seed

install:
	pip install -e ".[dev]"

serve:
	uvicorn api.main:create_app --factory --reload --host 0.0.0.0 --port 8000

scrape:
	python scripts/scrape.py --type article --max-pages 50

ingest:
	python scripts/ingest.py

seed:
	python scripts/seed_sample.py

test:
	pytest tests/ -v

lint:
	ruff check . --fix
	ruff format .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down
