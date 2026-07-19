import torch


def generate(model, x, y, eps, alpha=0.01, iters=40, criterion=torch.nn.CrossEntropyLoss()):
    model.eval()
    x_adv = x.clone().detach().to(x.device)
    # Random start
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0, 1)
    
    for _ in range(iters):
        x_adv.requires_grad = True
        outputs = model(x_adv)
        loss = criterion(outputs, y)
        
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            delta = torch.clamp(x_adv - x, min=-eps, max=eps)
            x_adv = torch.clamp(x + delta, min=0, max=1)
    return x_adv.detach()
