# Reproduce the Work — Adversarial ML Lab Poster 06

**Pinned CI code snapshot:** `05ee102c4c20fe3ebdcd3a8ac3dd1ea530de6698`  
**Successful Python 3.12 CI:** [run 38498116492](https://github.com/poojakira/adversarial-ml-lab/actions/runs/38498116492) (October 14, 2026 UTC)

```bash
git clone https://github.com/poojakira/adversarial-ml-lab.git
cd adversarial-ml-lab
git checkout 05ee102c4c20fe3ebdcd3a8ac3dd1ea530de6698
python -m pip install -e ".[dev]"
pytest tests/ -q --cov=adv_lab --cov-report=term
```

Expected from pinned CI: **112 passed, 32.22% statement coverage**. The CIFAR-10 benchmark was *skipped* in that CI. A separately committed real SmallCNN artifact reports **71.82% clean accuracy**, **3.32% FGSM**, **0% PGD-20**, and **4.20% C&W L2** on a 1,024-sample attack subset (clean accuracy uses 10,000 samples). Run the benchmark using the documented dependencies and pinned configuration to reproduce these separate measurements.
