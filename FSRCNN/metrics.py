import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from torchvision import models
from torchmetrics.functional import structural_similarity_index_measure as ssim

class EvaluationMetrics:
    def __init__(self, device=None):
        """Initialize the Image Quality Metrics calculator."""
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_image(self, image_path):
        """Load an image from a file path."""
        image = Image.open(image_path)
        return image

    def image_to_tensor(self, image):
        """Convert an image to a PyTorch tensor."""
        transform = transforms.Compose([transforms.ToTensor()])
        image_tensor = transform(image).unsqueeze(0).to(self.device)  # Add batch dimension
        return image_tensor

    @staticmethod
    def normalize_to_01(tensor):
        """Clamp the input tensor to the range [0, 1]."""
        return torch.clamp(tensor, 0, 1)

    def calculate_psnr(self, sr_image: torch.Tensor, hr_image: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Peak Signal-to-Noise Ratio (PSNR) for a batch of images.

        Args:
            sr_image (torch.Tensor): Super-resolved image tensor of shape (N, C, H, W).
            hr_image (torch.Tensor): High-resolution image tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: A tensor containing PSNR values for each image in the batch.
        """
        sr_image = self.normalize_to_01(sr_image)
        hr_image = self.normalize_to_01(hr_image)
        mse = torch.mean((sr_image - hr_image) ** 2, dim=(1, 2, 3))  # Compute MSE for each image
        psnr = 10 * torch.log10(1.0 / mse)  # Compute PSNR
        return psnr


    def calculate_ssim(self, sr_image: torch.Tensor, hr_image: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Structural Similarity Index Measure (SSIM) for a batch of images.

        Args:
            sr_image (torch.Tensor): Super-resolved image tensor of shape (N, C, H, W).
            hr_image (torch.Tensor): High-resolution image tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: A tensor containing SSIM values for each image in the batch.
        """
        sr_image = self.normalize_to_01(sr_image)
        hr_image = self.normalize_to_01(hr_image)

        ssim_values = []
        for i in range(sr_image.size(0)):  # Loop through batch
            ssim_value = ssim(sr_image[i].unsqueeze(0), hr_image[i].unsqueeze(0))  # Compute SSIM per image
            ssim_values.append(ssim_value)
        
        return torch.stack(ssim_values)  # Return as a tensor


    def evaluate(self, sr_image_path, hr_image_path, hr_grayscale=False):
        """Evaluate PSNR and SSIM for a pair of SR and HR images."""
        # Load images
        sr_image = self.load_image(sr_image_path)
        hr_image = self.load_image(hr_image_path)

        # Convert HR to grayscale if needed
        if hr_grayscale:
            hr_image = hr_image.convert('L')

        # Convert to tensors
        sr_image_tensor = self.image_to_tensor(sr_image)
        hr_image_tensor = self.image_to_tensor(hr_image)

        # Ensure SR and HR have the same shape
        assert sr_image_tensor.shape == hr_image_tensor.shape, "SR and HR images must have the same dimensions!"

        # Calculate metrics
        psnr_value = self.calculate_psnr(sr_image_tensor, hr_image_tensor)
        ssim_value = self.calculate_ssim(sr_image_tensor, hr_image_tensor)

        return {
            'psnr': psnr_value,
            'ssim': ssim_value
        }

# Example usage
if __name__ == "__main__":
    # Paths to images
    sr_image_path = 'D:/HUST/_Intro to DL/FSRCNN-PyTorch/results/fsrcnn_x4/fsrcnn_x4/img_001_x4.png'
    hr_image_path = 'D:/HUST/_Intro to DL/FSRCNN-PyTorch/data/Set5/GTmod12/img_001.png'

    # Initialize the metrics calculator
    metrics_calculator = EvaluationMetrics()

    # Evaluate metrics
    results = metrics_calculator.evaluate(sr_image_path, hr_image_path, hr_grayscale=True)
    psnr = metrics_calculator.calculate_psnr()

    # Print results
    print(f"PSNR: {results['psnr']} dB")
    print(f"SSIM: {results['ssim']}")
