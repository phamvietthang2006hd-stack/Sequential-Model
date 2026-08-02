import torch 
from torch import Tensor

def mse(pred: Tensor, target: Tensor) -> float:
    return torch.mean((pred - target)**2).item() 

def rmse(pred: Tensor, target: Tensor) -> float:
    return torch.sqrt(torch.mean((pred - target)**2)).item()

def mae(pred: Tensor, target: Tensor, eps: float = 1e-8) -> float:
    return (torch.mean(torch.abs((target - pred) / (target + eps))) * 100.0).item()
