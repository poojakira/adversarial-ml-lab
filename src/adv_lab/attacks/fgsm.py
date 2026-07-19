import torch


def generate(model, x, y, eps, criterion=torch.nn.CrossEntropyLoss()):
    model.eval()
    x = x.clone().detach().to(x.device)
    x.requires_grad = True
    
    outputs = model(x)
    loss = criterion(outputs, y)
    
    model.zero_grad()
    loss.backward()
    
    adv_x = x + eps * x.grad.sign()
    adv_x = torch.clamp(adv_x, 0, 1)
    return adv_x.detach()
