.PHONY: install install-dev validate train evaluate report test lint clean

install:
	@echo "Install a matching torch/torchvision build first; see README.md."
	python -m pip install -r requirements.txt

install-dev:
	python -m pip install -r requirements-dev.txt

validate:
	python -m src.validate_data --config config.yaml

train:
	python -m src.train --config config.yaml

evaluate:
	python -m src.evaluate --config config.yaml

report:
	python -m src.reporting

test:
	python -m pytest

lint:
	python -m ruff check src tests

clean:
	python -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in map(Path, ['.pytest_cache', '.ruff_cache', '.runtime'])]; [p.unlink(missing_ok=True) for p in map(Path, ['reports/test_metrics.json', 'reports/experiment_dashboard.png'])]"
