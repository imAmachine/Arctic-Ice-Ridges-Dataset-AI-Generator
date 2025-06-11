from torch import log, stack, float32, ones_like, linalg
import torch
from scipy.stats import linregress
import numpy as np

class FractalAnalyzer:
    @staticmethod
    def box_counting(binary_image):
        sizes = []
        counts = []

        min_side = min(binary_image.shape)
        for size in range(2, min_side // 2 + 1, 2):
            count = 0
            for x in range(0, binary_image.shape[0] + 1, size):
                for y in range(0, binary_image.shape[1] + 1, size):
                    if np.any(binary_image[x:x+size, y:y+size] > 0):
                        count += 1
            
            sizes.append(size)
            counts.append(count)

        return sizes, counts

    @staticmethod
    def calculate_fractal_dimension(sizes, counts, epsilon=1e-10):
        log_sizes = np.log(np.array(sizes))
        log_counts = np.log(np.array(counts) + epsilon)
        slope, intercept, r_value, p_value, std_err = linregress(log_sizes, log_counts)
        return np.abs(slope) # type: ignore


class FractalAnalyzerGPU:
    @staticmethod
    def box_counting(tensor, min_size=2, max_size=64):
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        if tensor.dim() == 3:
            tensor = tensor.mean(dim=0)
        
        binary = (tensor > 0.5).float()
        H, W = binary.shape
        
        sizes = []
        counts = []
        
        for size in reversed(range(min_size, max_size + 1)):
            if size > H or size > W:
                continue
                
            # Разбиение на блоки
            n_h = H // size
            n_w = W // size
            cropped = binary[:n_h*size, :n_w*size]
            
            patches = cropped.view(n_h, size, n_w, size)
            patches = patches.permute(0, 2, 1, 3).contiguous()
            patches = patches.view(-1, size, size)
            
            count = (patches.sum(dim=(1,2)) > 0).sum().item()
            
            sizes.append(size)
            counts.append(count)
        
        sizes_tensor = torch.tensor(sizes, dtype=float32, device=binary.device).clone().detach()
        counts_tensor = torch.tensor(counts, dtype=float32, device=binary.device).clone().detach()
        
        return sizes_tensor, counts_tensor

    @staticmethod
    def calculate_fractal_dimension(sizes, counts):
        log_sizes = log(sizes)
        log_counts = log(counts + 1e-8)
        
        X = stack([log_sizes, ones_like(log_sizes)], dim=1)
        Y = log_counts.view(-1, 1)
        
        solution = linalg.lstsq(X, Y).solution
        return -solution[0].item()