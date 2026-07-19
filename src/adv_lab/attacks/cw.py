import torch


def generate(model, x, y, c=1e-4, kappa=0, iters=1000, lr=0.01):
    # A simplified Carlini-Wagner L2 attack
    model.eval()
    x = x.clone().detach().to(x.device)
    w = torch.zeros_like(x, requires_grad=True).to(x.device)
    
    optimizer = torch.optim.Adam([w], lr=lr)
    
    for _ in range(iters):
        adv_x = 0.5 * (torch.tanh(w) + 1)
        outputs = model(adv_x)
        
        # L2 distance
        l2_loss = torch.sum((adv_x - x) ** 2, dim=[1,2,3])
        
        # f(x)
        one_hot_y = torch.nn.functional.one_hot(y, num_classes=outputs.shape[-1]).to(x.device)
        real = torch.sum(one_hot_y * outputs, dim=1)
        other = torch.max((1 - one_hot_y) * outputs - (one_hot_y * 10000), dim=1)[0]
        f_x = torch.clamp(real - other + kappa, min=0)
        
        loss = torch.sum(l2_loss + c * f_x)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return (0.5 * (torch.tanh(w) + 1)).detach()
