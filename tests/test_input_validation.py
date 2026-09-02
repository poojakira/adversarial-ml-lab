"""Tests for attack input validation and benchmark_runner error paths.

These lock in the hardening added to the FGSM/PGD/C&W attacks and the
benchmark runner: bad inputs must fail loud with actionable messages rather
than producing a silently wrong robustness number.
"""

import pytest
import torch
import torch.nn as nn

from adv_lab.attacks.cw import cw_l2_attack
from adv_lab.attacks.fgsm import _validate_attack_inputs, fgsm_attack
from adv_lab.attacks.pgd import pgd_attack, pgd_l2
from adv_lab.eval.benchmark_runner import benchmark_runner


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)


def _model() -> nn.Module:
    m = _Net()
    m.eval()
    return m


def _good_batch() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.rand(4, 3, 8, 8), torch.tensor([0, 1, 0, 1])


# ── _validate_attack_inputs contract ──────────────────────────────────────────


def test_non_tensor_images_raises() -> None:
    _, y = _good_batch()
    with pytest.raises(TypeError, match="images must be a torch.Tensor"):
        _validate_attack_inputs([1, 2, 3], y, 0.1)  # type: ignore[arg-type]


def test_non_tensor_labels_raises() -> None:
    x, _ = _good_batch()
    with pytest.raises(TypeError, match="labels must be a torch.Tensor"):
        _validate_attack_inputs(x, [0, 1], 0.1)  # type: ignore[arg-type]


def test_wrong_ndim_images_raises() -> None:
    with pytest.raises(ValueError, match="4D"):
        _validate_attack_inputs(torch.rand(3, 8, 8), torch.tensor([0, 1, 0]), 0.1)


def test_integer_images_raises() -> None:
    x = torch.randint(0, 2, (2, 3, 8, 8))
    with pytest.raises(TypeError, match="floating-point"):
        _validate_attack_inputs(x, torch.tensor([0, 1]), 0.1)


def test_label_batch_mismatch_raises() -> None:
    x, _ = _good_batch()
    with pytest.raises(ValueError, match="batch size mismatch"):
        _validate_attack_inputs(x, torch.tensor([0, 1]), 0.1)


def test_empty_batch_raises() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        _validate_attack_inputs(torch.rand(0, 3, 8, 8), torch.zeros(0, dtype=torch.long), 0.1)


def test_negative_epsilon_raises() -> None:
    x, y = _good_batch()
    with pytest.raises(ValueError, match="non-negative"):
        _validate_attack_inputs(x, y, -0.1)


def test_nan_epsilon_raises() -> None:
    x, y = _good_batch()
    with pytest.raises(ValueError, match="finite"):
        _validate_attack_inputs(x, y, float("nan"))


def test_out_of_range_images_raises() -> None:
    x = torch.rand(4, 3, 8, 8) * 5.0
    y = torch.tensor([0, 1, 0, 1])
    with pytest.raises(ValueError, match=r"\[0, 1\] range"):
        _validate_attack_inputs(x, y, 0.1)


# ── attacks reject bad inputs through the public entry points ─────────────────


def test_fgsm_rejects_out_of_range() -> None:
    x = torch.rand(2, 3, 8, 8) + 2.0
    with pytest.raises(ValueError):
        fgsm_attack(_model(), x, torch.tensor([0, 1]), epsilon=0.1)


def test_pgd_rejects_bad_steps() -> None:
    x, y = _good_batch()
    with pytest.raises(ValueError, match="steps must be"):
        pgd_attack(_model(), x, y, epsilon=0.1, steps=0)


def test_pgd_rejects_bad_alpha() -> None:
    x, y = _good_batch()
    with pytest.raises(ValueError, match="alpha"):
        pgd_attack(_model(), x, y, epsilon=0.1, alpha=0.0)


def test_pgd_l2_rejects_bad_steps() -> None:
    x, y = _good_batch()
    with pytest.raises(ValueError, match="steps must be"):
        pgd_l2(_model(), x, y, epsilon=0.5, steps=0)


def test_cw_rejects_bad_steps() -> None:
    x, y = _good_batch()
    with pytest.raises(ValueError, match="steps must be"):
        cw_l2_attack(_model(), x, y, steps=0)


def test_cw_rejects_bad_lr() -> None:
    x, y = _good_batch()
    with pytest.raises(ValueError, match="lr"):
        cw_l2_attack(_model(), x, y, lr=0.0)


# ── benchmark_runner error paths ──────────────────────────────────────────────


def test_benchmark_runner_rejects_negative_epsilon(tmp_path) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        benchmark_runner(epsilon=-0.1, output_path=str(tmp_path / "r.json"))


def test_benchmark_runner_rejects_zero_steps(tmp_path) -> None:
    with pytest.raises(ValueError, match="pgd_steps"):
        benchmark_runner(pgd_steps=0, output_path=str(tmp_path / "r.json"))


def test_benchmark_runner_rejects_zero_batch(tmp_path) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        benchmark_runner(batch_size=0, output_path=str(tmp_path / "r.json"))


def test_benchmark_runner_missing_model_path(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        benchmark_runner(model_path=str(tmp_path / "nope.pt"), output_path=str(tmp_path / "r.json"))


def test_benchmark_runner_bad_checkpoint(tmp_path) -> None:
    bad = tmp_path / "bad.pt"
    torch.save({"not_a_real_layer.weight": torch.zeros(3)}, bad)
    with pytest.raises(ValueError, match="does not match the expected"):
        benchmark_runner(model_path=str(bad), output_path=str(tmp_path / "r.json"))


def test_benchmark_runner_default_runs(tmp_path) -> None:
    out = tmp_path / "report.json"
    report = benchmark_runner(batch_size=8, pgd_steps=5, output_path=str(out))
    assert out.exists()
    assert report["tool"] == "adversarial-ml-lab"
    assert set(report["attacks"]) == {"fgsm", "pgd", "cw_l2_proxy"}
    assert report["pass_fail"] in {"PASS", "FAIL"}
