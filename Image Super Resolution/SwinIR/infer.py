import torch
import torch.nn as nn 
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import os
import argparse
from torchvision.transforms import ToTensor, ToPILImage

# Import model here
from models.network_swinir import SwinIR

model_swinir = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1.,
               depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
               mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
model_path_swinir = 'model_zoo/model_weight_X4_swinir.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path, model, device):
    param_key_g = "params"
    pretrained_dict = torch.load(model_path)
    model.load_state_dict(pretrained_dict[param_key_g] if param_key_g in pretrained_dict.keys() else pretrained_dict, strict=True)
    model.to(device)
    model.eval()

load_model(model_path_swinir, model_swinir, DEVICE)

def preprocess_image(image, device):
    """
    Preprocess an image for inference.
    Args:
        image_path (str): Path to the image.
        device (torch.device): Device to process the image.
        scale_factor (int): Factor to upscale the image.
    Returns:
        lr_image (torch.Tensor): Preprocessed low-resolution image tensor.
    """

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to [0, 1] range
    ])
    lr_image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    return lr_image

def postprocess_image(output_tensor):

    """
    Post-process the model output tensor into an image.
    Args:
        output_tensor (torch.Tensor): Output tensor from the model.
    Returns:
        np.ndarray: Output image in RGB format.
    """
    output_image = output_tensor.squeeze(0).clamp(0, 1).cpu().numpy()
    output_image = np.transpose(output_image, (1, 2, 0))  # Convert to HWC format
    output_image = (output_image * 255).astype(np.uint8)  # Convert to [0, 255]
    return output_image


def infer(img_path):
    data = Image.open(img_path)
    
    transform_data = preprocess_image(data, DEVICE)
    
    # Inference
    with torch.no_grad():
        prediction = model_swinir(transform_data)

    pred_img = postprocess_image(prediction)  
    pred_img = Image.fromarray(pred_img)  
    # Save the result in 'output' directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)  # Create 'output' directory if it doesn't exist
    output_file_name = f'swinir_{os.path.basename(img_path)}'  # Only the image name
    output_path = os.path.join(output_dir, output_file_name)
    

    pred_img.save(output_path)
    print(f"Segmented image saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on an image")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()
    
    infer(args.image_path)