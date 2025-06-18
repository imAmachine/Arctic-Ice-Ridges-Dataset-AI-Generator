from typing import Optional
from torch import log, stack, ones_like
import torch


class FractalAnalyzerGPU:
    @staticmethod
    def box_counting(tensor: torch.Tensor, min_size: int = 2, max_size: int = 64, threshold: Optional[float] = None):
        if tensor.dim() == 4:
            tensor = tensor.view(-1, *tensor.shape[-2:])
        if tensor.dim() == 3:
            tensor = tensor.mean(dim=0)
        
        if threshold is None:
            threshold = tensor.mean() + tensor.std().item()
            
        binary = (tensor > threshold).float()
        
        H, W = binary.shape
        sizes, counts = [], []
        
        for size in range(min_size, max_size + 1):
            if size > H or size > W:
                break
            n_h, n_w = H // size, W // size
            cropped = binary[:n_h*size, :n_w*size]
            patches = cropped.unfold(0, size, size).unfold(1, size, size)
            patches = patches.reshape(-1, size*size)
            count = (patches.sum(dim=1) > 0).sum().item()
            sizes.append(size)
            counts.append(count)
        
        sizes = torch.tensor(sizes, dtype=torch.float32, device=binary.device)
        counts = torch.tensor(counts, dtype=torch.float32, device=binary.device)
        return sizes, counts

    @staticmethod
    def calculate_fractal_dimension(sizes: torch.Tensor, counts: torch.Tensor):
        sizes, idx = torch.sort(sizes)
        counts = counts[idx]
        
        log_sizes = log(sizes)
        log_counts = log(counts + 1e-8)
        
        X = stack([log_sizes, ones_like(log_sizes)], dim=1)
        Y = log_counts.view(-1, 1)
        
        sol = torch.linalg.lstsq(X, Y).solution
        fractal_dim = sol[0].item()
        return fractal_dim