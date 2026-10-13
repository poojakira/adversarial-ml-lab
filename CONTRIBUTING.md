> **READ [LICENSE](LICENSE) BEFORE COPYING OR CHANGING ANYTHING.** This is an owner-maintained, proprietary repository. **Unsolicited code/documentation contributions are not accepted.** The development instructions below apply **only to people who have received prior written authorization** from Pooja Kiran. Public viewing/forking under GitHub's Terms, applicable law, third-party rights, and prior valid licenses remain unaffected. Any accepted external work requires an agreed written copyright license or assignment; a PR alone does not transfer ownership. For permission contact poojakiranbhardwaj@gmail.com.

# Contributing to adversarial-ml-lab

This repository is owner-maintained proprietary software. Only contributors who have received prior written authorization may use the technical guidance below.

---

## Who Should Contribute

This library is designed to be used by:

- **ML security engineers** validating model robustness before deployment
- **Adversarial ML researchers** building on FGSM / PGD / C&W baselines
- **Red team practitioners** quantifying evasion risk for ML-based detection systems
- **AI governance teams** who need structured benchmark output for design review documentation

Please obtain written authorization before implementing changes or opening a code PR.

---

## Quick Start

```bash
git clone https://github.com/poojakira/adversarial-ml-lab
cd adversarial-ml-lab
pip install -e ".[dev]"
pre-commit install
```

Verify everything works before making changes:

```bash
pytest tests/ -v --cov=adv_lab --cov-fail-under=80
ruff check . && ruff format --check .
bandit -r src/ -ll
```

---

## Development Workflow

1. **Fork** the repo and create a feature branch: `git checkout -b feature/your-feature`
2. **Write tests first** for any new attack or harness feature (TDD preferred).
3. **Run the full check suite** before opening a PR (see above).
4. **Open a PR** against `main`. The CI pipeline runs lint → typecheck → test → bandit automatically.

---

## Contribution Types

### Adding a new attack

1. Create `src/adv_lab/attacks/your_attack.py`.
2. Follow the existing function signature convention:
   ```python
   def your_attack(
       model: nn.Module,
       x: torch.Tensor,
       y: torch.Tensor,
       **kwargs,
   ) -> torch.Tensor:
       """Docstring: paper citation, MITRE ATLAS technique, L∞ or L2 norm."""
       ...
   ```
3. Add a MITRE ATLAS or ATT&CK technique comment explaining what threat this models.
4. Add unit tests in `tests/attacks/test_your_attack.py`.
5. Wire it into `benchmark_runner.py` and add it to the README status table.

### Improving the benchmark runner

- Keep `benchmark_runner()` self-contained (no imports from sibling modules that aren't in the package)  --  the inline attack functions exist so the runner can be used standalone.
- The JSON report schema is documented in `README.md`. Any schema changes require a version bump and schema migration notes in the PR description.

### Adding a benchmark dataset

- Do not commit large binary files (model weights, dataset archives) to the repo.
- Reference external data via download scripts in `scripts/`.
- Document expected benchmark values and their source (paper, dataset split, model architecture) clearly in the PR.

### Documentation

- READMEs, docstrings, and this file are always welcome improvements.
- Keep the **Honest Status table** in README.md accurate  --  if a feature is not implemented or a benchmark is not measured in this repo, say so.

---

## Code Style

- **Formatter:** `ruff format` (configured in `pyproject.toml`)
- **Linter:** `ruff check`  --  all warnings are errors in CI
- **Type checker:** CI currently runs `pyright` on `src/adv_lab/eval/ci_signing.py`
- **Security scanner:** `bandit -r src/ -ll`  --  medium and high findings are CI failures
- **Python version:** ≥ 3.10 (uses `str | None` union syntax, `match`, etc.)

---

## Testing Requirements

- **Coverage gate:** CI currently enforces 15% line coverage (`--cov-fail-under=15`); local contributors may run the stricter 80% command above before PRs
- **Attack liveness gate:** PGD robust accuracy on the dummy model must be **< 30%**  --  this confirms the attacks are actually doing something. A PR that breaks attack effectiveness will fail CI.
- All new public functions must have at least one unit test covering the happy path and one covering error handling.

---

## Security

If you discover a security vulnerability in this tool (not in the attacks it implements  --  those are the point), please follow the process in [SECURITY.md](SECURITY.md). Do not open a public issue.

---

## Reporting Issues

- **Bug in an attack implementation** → open a GitHub Issue with a minimal reproducer.
- **Wrong benchmark number** → open a PR with corrected values and evidence (paper citation, code, output).
- **Feature request** → open a GitHub Issue describing the use case first; we prefer to discuss before implementation.

---

## Pull Request Checklist

Before requesting a review, confirm:

- [ ] Tests pass locally: `pytest tests/ -v --cov=adv_lab --cov-fail-under=80`
- [ ] No lint errors: `ruff check . && ruff format --check .`
- [ ] No bandit findings: `bandit -r src/ -ll`
- [ ] Docstrings added for new public functions
- [ ] README status table updated if implementation status changed
- [ ] THREAT_MODEL.md updated if you changed any input handling or file loading code

---

## License

Do not submit code without prior written approval and an explicit agreement addressing its ownership and licensing. Earlier MIT-licensed versions retain their previously granted rights.
