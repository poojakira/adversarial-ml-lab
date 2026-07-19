import torch

from adv_lab.attacks.pgd import generate as pgd_generate


def train_epoch(model, dataloader, optimizer, eps=0.3, alpha=0.01, iters=40):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    for x, y in dataloader:
        x, y = x.to(next(model.parameters()).device), y.to(next(model.parameters()).device)
        
        # Generate adversarial examples
        model.eval()
        adv_x = pgd_generate(model, x, y, eps, alpha, iters)
        model.train()
        
        optimizer.zero_grad()
        outputs = model(adv_x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
        
    return total_loss / total, correct / total
