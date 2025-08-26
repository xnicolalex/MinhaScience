# utils/metrics.py
import torch
from torchmetrics.image import FID, SSIM

class FlowMatchingMetrics:
    def __init__(self, device="cuda"):
        self.fid = FID(normalize=True).to(device)
        self.ssim = SSIM(data_range=1.0).to(device)

    def update(self, reals: torch.Tensor, fakes: torch.Tensor):
        """Update metric states with a batch of images (in [0,1])."""
        assert reals.shape == fakes.shape, "Input shapes must match"
        self.fid.update(reals * 255, real=True)    # FID expects [0,255]
        self.fid.update(fakes * 255, real=False)
        self.ssim.update(reals, fakes)             # SSIM expects [0,1]

    def compute(self) -> dict:
        return {
            "fid": self.fid.compute().item(),
            "ssim": self.ssim.compute().item()
        }

    def reset(self):
        self.fid.reset()
        self.ssim.reset()