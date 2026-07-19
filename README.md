# Adversarial ML Lab

Adversarial ML Lab provides tools for adversarial machine learning attacks and robustness evaluation.
This repository was simplified to focus on functional local implementations of common adversarial attacks.

## Features
- **Attacks**: Implementations of FGSM, PGD, and a simplified CW L2 attack using PyTorch.
- **Evaluation**: A robustness evaluation harness.
- **Defenses**: Basic adversarial training loop.

## Installation

`ash
git clone https://github.com/poojakira/adversarial-ml-lab
cd adversarial-ml-lab
pip install -e .
`

## Running Tests

Tests verify that attacks successfully generate bounded adversarial examples:
`ash
pip install -e ".[dev]"
pytest -v tests/
`

## Limitations

- **Scalability**: This repository is for educational and local research purposes. It is not currently suitable for large-scale distributed adversarial generation.
- **Integration**: The directory structure previously contained mock integrations with AWS KMS, GCP KMS, Vault, OCI Registries, and RobustBench. These were non-functional stubs and have been removed. This tool does not integrate with cloud KMS or enterprise model registries.
- **Attack Variations**: The implemented CW attack is a simplified version and may not reach state-of-the-art performance against advanced defenses compared to full libraries like AutoAttack.
- **Online Evaluation**: This repository no longer pretends to be an online evaluation service running models via Triton/TorchServe/vLLM.

