import torch

from adv_lab.attacks.fgsm import generate as fgsm_generate
from adv_lab.attacks.pgd import generate as pgd_generate


def evaluate_robustness(model, dataloader, attack='fgsm', eps=0.3):
    model.eval()
    correct = 0
    total = 0
    
    for x, y in dataloader:
        x, y = x.to(next(model.parameters()).device), y.to(next(model.parameters()).device)
        
        if attack == 'fgsm':
            adv_x = fgsm_generate(model, x, y, eps)
        elif attack == 'pgd':
            adv_x = pgd_generate(model, x, y, eps, alpha=eps/4, iters=10)
        else:
            adv_x = x # clean
            
        with torch.no_grad():
            outputs = model(adv_x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
    return correct / total
