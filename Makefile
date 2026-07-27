PYTHON ?= python
PIP ?= $(PYTHON) -m pip
ATTACK_CORE_PATH ?= ../attack-v19-core
SRC := src/adv_lab attack_mapping tests

.PHONY: install install-core data lint format test build security dashboard verify

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	$(PIP) install build ruff bandit pip-audit

install-core:
	$(PIP) install -e "$(ATTACK_CORE_PATH)"

data:
	$(PYTHON) "$(ATTACK_CORE_PATH)/scripts/download_attack_data.py"

lint:
	$(PYTHON) -m ruff check $(SRC)

format:
	$(PYTHON) -m ruff format $(SRC)

test: install-core data
	$(PYTHON) -m pytest tests -q

build:
	$(PYTHON) -m build

security:
	$(PYTHON) -m bandit -r src attack_mapping -ll
	$(PYTHON) -m pip_audit -r requirements.txt

dashboard:
	$(PYTHON) -m http.server 8080 --directory dashboard

verify: lint test build security