# Reproduce the Work - Poster 06

**Repository:** `github.com/poojakira/adversarial-ml-lab`  
**Verified code snapshot:** `490b71ec480c40622b63b1adaf2bf2f75959b8d6`  
**CI run:** `37169514061`

```bash
git clone https://github.com/poojakira/adversarial-ml-lab.git
cd adversarial-ml-lab
git checkout 490b71ec480c40622b63b1adaf2bf2f75959b8d6
python -m pip install -e ".[dev]"
pytest tests/ -q --cov=adv_lab --cov-report=term
python scripts/run_real_smallcnn_benchmark.py --epochs 6 --attack-samples 1000
```

Current CI expectation: **112 tests passed**, **32.22% statement coverage**.

Committed real benchmark reference:

- clean: **71.82%**
- FGSM robust: **3.32%**
- PGD-20 robust: **0.00%**
- C&W L2 robust: **4.20%**
- attack subset: **1,024**


## Claim audit and rendered-source boundary, October 13, 2026 UTC

The reviewed printed claims, source and test commit anchors, CI dates and benchmark limitations are in `verified_claims.json`. The PDF is generated from that manifest and the PNG is rasterized from the same page by `render_poster.py`. Source inventory, dated CI and separate reported experiments must not be combined into one result. Earlier artifact versions remain in Git history; these corrections do not assert production readiness.

Offline poster rendering: `python -m pip install reportlab==4.4.9 PyMuPDF==1.26.6`, then `python poster/render_poster.py`. The font is bundled with the pinned ReportLab package; no browser or network fetch is used during rendering.
