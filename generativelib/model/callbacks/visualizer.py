import os
from matplotlib import pyplot as plt
from torch import Tensor

from generativelib.model.enums import ExecPhase


class Visualizer:
    """Handles saving grids of input, generated, and target samples."""
    def __init__(self, output_path: str):
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

    def save(self, inp: Tensor, target: Tensor, gen: Tensor, phase: ExecPhase, samples: int = 3) -> None:
        cols = min(samples, inp.size(0), 5)
        plt.figure(figsize=(12, 12), dpi=300)
        for row_idx, batch in enumerate((inp, gen, target)):
            for col_idx in range(cols):
                img = batch[col_idx].cpu().squeeze()
                ax = plt.subplot(3, cols, row_idx * cols + col_idx + 1)
                ax.imshow(img.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
                ax.set_title(f"{['Input', 'Gen', 'Target'][row_idx]} {col_idx+1}")
                ax.axis('off')
        plt.suptitle(f"Phase: {phase.value}", y=1.02)
        plt.tight_layout(pad=3)
        path = os.path.join(self.output_path, f"{phase.value}.png")
        plt.savefig(path)
        plt.close()