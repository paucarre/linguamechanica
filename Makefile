.PHONY: test format

help:
	@echo "make test   -- runs tests and generates coverage report"
	@echo "make format -- runs formats and basic code cleanup"
	@echo

test:
	poetry run pytest --cov-report term --cov=linguamechanica tests

format:
	poetry run black .
	poetry run autoflake -r . --in-place --expand-star-imports --remove-unused-variables --remove-all-unused-imports