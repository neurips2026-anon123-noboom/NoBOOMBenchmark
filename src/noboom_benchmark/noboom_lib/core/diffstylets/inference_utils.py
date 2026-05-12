import torch
import numpy as np

def normalize (t: torch.Tensor):
    """
    Normalizes a torch.tensor of shape (B, 1, T) using z-score standardization.
    
    Args:
        t: A torch.tensor of shape (B, 1, T).
        
    Returns:
        A tuple (normalized_t, mean, std), where:
        - normalized_t: The normalized tensor.
        - mean: The calculated mean.
        - std: The calculated standard deviation.
    """
    mean = t.mean(dim=2, keepdim=True)
    std = t.std(dim=2, keepdim=True)
    epsilon = 1e-8
    
    normalized_t = (t - mean) / (std + epsilon)
    
    return normalized_t, {"mean":mean, "std":std}

def denormalize (t: torch.Tensor, stats: dict):
    """
    Reverses the z-score standardization to denormalize a tensor.

    The denormalization formula is: x = z * std + mean.
    
    Args:
        t: The normalized torch.tensor (z).
        mean: The original mean (mu) used during normalization.
        std: The original standard deviation (sigma) used during normalization.
        
    Returns:
        The denormalized tensor (x), returned to its original scale.
    """
    denormalized_t = (t * stats["std"]) + stats["mean"]
    
    return denormalized_t

def convert_to_tensor(t, device):
    """
    Cast to tensor if needed

    Args:
        t: torch.tensor/np.array
        device: torch.device
    Returns:
        torch.tensor
    """
    if isinstance(t, np.ndarray):
        t = torch.tensor(t)
        
    if t.ndim == 1:
        # single inference
        t = t.unsqueeze(0).unsqueeze(0).float()
        
    elif t.ndim == 2: 
        # batched inference
        t = t.unsqueeze(1).float()
    
     
    t = t.to(device)
    return t