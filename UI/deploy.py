import torch
import torch.nn as nn 
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import streamlit as st 
from torchvision.transforms import ToTensor, ToPILImage
import time

# Import model here
from Image_Super_Resolution.SwinIR.models.network_swinir import SwinIR
from Image_Super_Resolution.FSRCNN.models import FSRCNN



model_swinir = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1.,
               depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
               mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')

model_srgan = model_swinir
model_fsrcnn = model_swinir
model_diffIR_SR = model_swinir
model_diffIR_DB = model_swinir
model_nafnet = model_swinir


# model_path = 'model_zoo/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth'
model_path_swinir = 'Image_Super_Resolution/SwinIR/model_zoo/model_weight_X4_swinir.pth'
model_path_srgan = model_path_swinir
model_path_fsrcnn = model_path_swinir
model_path_diffIR_SR = model_path_swinir
model_path_diffIR_DB = model_path_swinir
model_path_nafnet = model_path_swinir


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path, model, device):
    param_key_g = "params"
    pretrained_dict = torch.load(model_path)
    model.load_state_dict(pretrained_dict[param_key_g] if param_key_g in pretrained_dict.keys() else pretrained_dict, strict=True)
    model.to(device)
    model.eval()

load_model(model_path_swinir, model_swinir, DEVICE)
load_model(model_path_srgan, model_srgan, DEVICE)
load_model(model_path_fsrcnn, model_fsrcnn, DEVICE)
load_model(model_path_diffIR_SR, model_diffIR_SR, DEVICE)
load_model(model_path_diffIR_DB, model_path_diffIR_DB, DEVICE)
load_model(model_path_nafnet, model_nafnet, DEVICE)



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

# Streamlit app
st.title("Image Processing with SwinIR")

# Task selection
task = st.selectbox("Select a Task", ("Resolution Enhancement", "Deblurring"))

# Model selection based on task
if task == "Resolution Enhancement":
    model_choice = st.selectbox("Select a Model", ("Swin IR", "SRGAN", "FSRCNN", "DiffIR"))
    if model_choice == "Swin IR":
        model = model_swinir
    elif model_choice == "SRGAN":
        model = model_srgan # Replace by SR_GAN model
    elif model_choice == "DiffIR":
        model = model_diffIR_SR
    else: 
        model = model_fsrcnn
else:
    model_choice = st.selectbox("Select a Model", ("DiffIR", "NAF Net"))
    model = model_diffIR_DB if model_choice == "DiffIR" else model_nafnet

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display uploaded image
    before = time.time()
    input_image = Image.open(uploaded_file).convert("RGB")
    st.image(input_image, caption="Uploaded Image", use_column_width=False)

    # Process image
    if st.button("Process Image"):
        with st.spinner("Processing..."):
            
            lr_torch = preprocess_image(input_image, DEVICE)
            with torch.no_grad():
                sr_torch = model(lr_torch)
                sr_image = postprocess_image(sr_torch)
        st.success(f"Processing complete in {time.time() - before:.4f}s!")
        
        # Display the result
        st.image(sr_image, caption="Processed Image", use_column_width=False)
