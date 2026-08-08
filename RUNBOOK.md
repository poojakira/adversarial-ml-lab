# Runbook — Adversarial ML Lab

Step-by-step guide to run adversarial attacks and defenses locally.

---

## Step 1: Prerequisites

- Python 3.10+ (`py --version` on Windows, `python3 --version` on Linux)
- pip (bundled with Python)
- Git
- Sibling repo `attack-v19-core` cloned alongside this repo (required for ATT&CK mapping tests)

Directory layout:
```
repos/
├── adversarial-ml-lab/     ← you are here
└── attack-v19-core/        ← must exist for tests
```

---

## Step 2: Clone

**Windows (PowerShell):**
```powershell
cd C:\Users\pooja\repos
git clone https://github.com/poojakira/adversarial-ml-lab.git
cd adversarial-ml-lab
```

**Linux/macOS:**
```bash
cd ~/repos
git clone https://github.com/poojakira/adversarial-ml-lab.git
cd adversarial-ml-lab
```

---

## Step 3: Install

**Windows (PowerShell):**
```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

# Install attack-v19-core (sibling dependency)
.\.venv\Scripts\python.exe -m pip install -e ..\attack-v19-core
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

# Install attack-v19-core (sibling dependency)
pip install -e ../attack-v19-core
```

**Or use Makefile (if `make` available):**
```powershell
make install
make install-core
```

---

## Step 4: Run

**Windows (PowerShell):**
```powershell
# Run FGSM attack example
.\.venv\Scripts\python.exe -m adversarial_ml_lab.attacks.fgsm --epsilon 0.03

# Run PGD attack example
.\.venv\Scripts\python.exe -m adversarial_ml_lab.attacks.pgd --steps 40

# Run adversarial training defense
.\.venv\Scripts\python.exe -m adversarial_ml_lab.defenses.adversarial_training
```

**Linux/macOS:**
```bash
python -m adversarial_ml_lab.attacks.fgsm --epsilon 0.03
python -m adversarial_ml_lab.attacks.pgd --steps 40
python -m adversarial_ml_lab.defenses.adversarial_training
```

**Or use Makefile:**
```powershell
make run       # Run default attack pipeline
make dashboard # Serve dashboard at localhost:8080
```

---

## Step 5: Expected Output

FGSM attack output:
```
[FGSM] Epsilon: 0.03
[FGSM] Clean accuracy: 0.92
[FGSM] Adversarial accuracy: 0.34
[FGSM] Attack success rate: 0.63
```

PGD attack output:
```
[PGD] Steps: 40, Epsilon: 0.03
[PGD] Clean accuracy: 0.92
[PGD] Adversarial accuracy: 0.21
[PGD] Attack success rate: 0.77
```

> **Note:** Exact numbers depend on model and dataset. These are approximate.

---

## Step 6: Run Tests

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

**Linux/macOS:**
```bash
pytest tests/ -v
```

**With coverage:**
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ --cov=src --cov-report=term-missing
```

**Full verification (lint + test + build + security):**
```powershell
make verify
```

---

## Available Makefile Targets

| Command | What it does |
|---------|-------------|
| `make install` | Install dependencies into venv |
| `make install-core` | Install attack-v19-core from sibling dir |
| `make test` | Run pytest |
| `make lint` | Run ruff linter |
| `make format` | Auto-format with ruff |
| `make build` | Build wheel package |
| `make security` | Run bandit + pip-audit |
| `make verify` | All of the above in sequence |
| `make dashboard` | Serve dashboard at localhost:8080 |

---

## View Dashboard

```powershell
py -m http.server 8080 --directory dashboard
# Open http://localhost:8080
```

Or view hosted: https://poojakira.github.io/mlsec-dashboards/adversarial-ml-lab/

> **Note:** Dashboard shows security posture indicators — treat as visual aids, not certifications.

---

## Troubleshooting

### `make test` Fails with "No module named attack_v19_core"

**Fix:** Install the sibling dependency:
```powershell
.\.venv\Scripts\python.exe -m pip install -e ..\attack-v19-core
```

---

### ImportError: No module named 'torch'

PyTorch is a dependency. Install it:
```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision
```

For CPU-only (smaller download):
```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

### Tests Pass Locally but Fail in CI

- CI runs on Linux — check for Windows-specific path issues (`\` vs `/`)
- Run `make lint` before pushing to catch formatting issues
- Ensure all dependencies are in `pyproject.toml`, not just installed locally

---

### CUDA Out of Memory

Attacks default to GPU if available. Force CPU:
```powershell
$env:CUDA_VISIBLE_DEVICES = ""
.\.venv\Scripts\python.exe -m adversarial_ml_lab.attacks.fgsm --device cpu
```

---

## Known Limitations

- No published benchmark artifacts (CIFAR-10 accuracy numbers, etc.) in this repo
- Educational/research tool — use IBM ART for production adversarial robustness
- Tested locally on Windows; re-verify on Linux/CI after pushing
- `make test` depends on `../attack-v19-core` being cloned alongside
