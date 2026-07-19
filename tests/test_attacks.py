import torch
import torch.nn as nn

from adv_lab.attacks.cw import generate as cw_generate
from adv_lab.attacks.fgsm import generate as fgsm_generate
from adv_lab.attacks.pgd import generate as pgd_generate
from adv_lab.eval.harness import evaluate_robustness


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.fc = nn.Linear(16 * 30 * 30, 2)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

def test_fgsm_attack():
    model = DummyModel()
    x = torch.rand(4, 3, 32, 32)
    y = torch.tensor([0, 1, 0, 1])

    adv_x = fgsm_generate(model, x, y, eps=0.1)
    assert adv_x.shape == x.shape
    assert not torch.allclose(adv_x, x)
    assert adv_x.max() <= 1.0 and adv_x.min() >= 0.0

def test_pgd_attack():
    model = DummyModel()
    x = torch.rand(4, 3, 32, 32)
    y = torch.tensor([0, 1, 0, 1])

    adv_x = pgd_generate(model, x, y, eps=0.1, alpha=0.01, iters=5)
    assert adv_x.shape == x.shape
    assert not torch.allclose(adv_x, x)
    assert adv_x.max() <= 1.0 and adv_x.min() >= 0.0

def test_cw_attack():
    model = DummyModel()
    x = torch.rand(2, 3, 32, 32)
    y = torch.tensor([0, 1])

    adv_x = cw_generate(model, x, y, c=1e-4, kappa=0, iters=5, lr=0.01)
    assert adv_x.shape == x.shape
    assert not torch.allclose(adv_x, x)
    assert adv_x.max() <= 1.0 and adv_x.min() >= 0.0

def test_eval_harness():
    model = DummyModel()
    x = torch.rand(4, 3, 32, 32)
    y = torch.tensor([0, 1, 0, 1])
    dataloader = [(x, y)]

    acc_clean = evaluate_robustness(model, dataloader, attack='clean')
    acc_fgsm = evaluate_robustness(model, dataloader, attack='fgsm', eps=0.1)

    assert 0.0 <= acc_clean <= 1.0
    assert 0.0 <= acc_fgsm <= 1.0
