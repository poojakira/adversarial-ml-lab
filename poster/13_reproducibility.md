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
