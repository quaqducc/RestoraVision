import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchmetrics.functional as metrics_fn
import lpips


class EvaluationMetrics:
    def __init__(self, device):
        self.device = device

        # Define loss functions
        self.mse_criterion = nn.MSELoss().to(device)

        # Initialize LPIPS model (AlexNet by default)
        self.lpips_criterion = lpips.LPIPS(net='alex').to(device)

    def calculate_mse(self, sr, hr):
        """Calculate MSE loss between SR and HR images."""
        mse = self.mse_criterion(sr, hr)
        return mse

    def calculate_psnr(self, sr, hr):
        """Calculate PSNR (Peak Signal-to-Noise Ratio) between SR and HR images."""
        # sr_norm = torch.clamp(sr, 0, 1)
        # hr_norm = torch.clamp(hr, 0, 1)
        mse = self.calculate_mse(sr, hr)
        psnr = 10. * torch.log10(1. / mse)
        return psnr

    def calculate_ssim(self, sr, hr):
        """Calculate SSIM (Structural Similarity Index) between SR and HR images."""
        # Ensure input tensors are normalized between [0, 1]
        # sr_norm = torch.clamp(sr, 0, 1)
        # hr_norm = torch.clamp(hr, 0, 1)
        ssim = metrics_fn.structural_similarity_index_measure(sr, hr)
        return ssim

    def calculate_lpips(self, sr, hr):
        """Calculate LPIPS (Learned Perceptual Image Patch Similarity) between SR and HR images."""
        # Ensure inputs are normalized to [-1, 1] for LPIPS
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        sr_norm = normalize(sr).to(self.device)
        hr_norm = normalize(hr).to(self.device)

        lpips_value = self.lpips_criterion(sr_norm, hr_norm).item()
        return lpips_value

    def evaluate_metrics(self, val_dataset):
        n_data = len(val_dataset)
        psnr = 0
        ssim = 0
        mse = 0
        lpips_val = 0

        for idx, (hr, gt) in enumerate(val_dataset):
            hr = hr.to(self.device, dtype=torch.float32).unsqueeze(0)
            gt = gt.to(self.device, dtype=torch.float32).unsqueeze(0)
            psnr += self.calculate_psnr(hr, gt)
            ssim += self.calculate_ssim(hr, gt)
            mse += self.calculate_mse(hr, gt)
            lpips_val += self.calculate_lpips(hr, gt)

        psnr /= n_data
        ssim /= n_data
        mse /= n_data
        lpips_val /= n_data

        return psnr, ssim, mse, lpips_val
