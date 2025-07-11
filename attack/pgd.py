import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

lower_limit, upper_limit = 0, 1
def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def attack_pgd(model, preprocessing, X, target, prompter=None, add_prompter=None, alpha=0.25,
               attack_iters=20, epsilon=2.0, norm="l_inf",train_trades=False):
    alpha=alpha/255.
    epsilon=epsilon/255.
    
    delta = torch.zeros_like(X).cuda()
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    
    for _ in range(attack_iters):
        _images = preprocessing(X + delta)
        if prompter is not None:
            prompted_images = prompter(_images)
        else:
            prompted_images = _images
        token_prompter = add_prompter() if add_prompter is not None else None
        
        if prompter is not None or token_prompter is not None:
            output, _ = model(prompted_images, token_prompter)
        else: 
            output = model(prompted_images)
        if train_trades:
            output_clean=model(preprocessing(X))
            loss=F.kl_div(F.log_softmax(output, dim=1), F.softmax(output_clean, dim=1), reduction='batchmean')
            
        else:    
            loss = F.cross_entropy(output, target)

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()


    return delta
