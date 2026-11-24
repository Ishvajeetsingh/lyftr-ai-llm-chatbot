.PHONY: up down build test

up:
	docker compose up -d --build

down:
	docker compose down

build:
	docker compose build

test:
	docker compose exec backend pytest -v
