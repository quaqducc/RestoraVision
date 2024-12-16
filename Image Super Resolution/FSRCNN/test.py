import argparse
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import FSRCNN
from utils import convert_ycbcr_to_rgb, preprocess
from metrics import EvaluationMetrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, default='model/fsrcnn_x4.pth', help="Path to pretrained FSRCNN weights.")
    parser.add_argument('--input-folder', type=str, default='dataset/set5', help="Path to input folder containing LR images.")
    parser.add_argument('--scale', type=int, default=4, help="Upscale factor.")
    parser.add_argument('--result-folder', type=str, default='results', help="Path to save output images and metrics.")
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = FSRCNN(scale_factor=args.scale).to(device)
    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
    model.eval()

    # Initialize metrics
    metrics_calculator = EvaluationMetrics(device)

    # Prepare result folder
    result_folder = Path(args.result_folder)
    result_folder.mkdir(parents=True, exist_ok=True)
    output_log_path = result_folder / 'test_log_set12.txt'

    # Initialize metrics log
    with open(output_log_path, 'w') as log_file:
        log_file.write("Image PSNR SSIM Time\n")

    total_psnr = 0
    total_ssim = 0
    total_time = 0
    num_images = 0

    # Process images in input folder
    input_folder = Path(args.input_folder)
    for image_path in input_folder.glob("*.*"):
        try:
            # Load image
            image = pil_image.open(image_path).convert('RGB')

            # Prepare HR and LR images
            image_width = (image.width // args.scale) * args.scale
            image_height = (image.height // args.scale) * args.scale
            hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
            lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)

            # Save original image to results folder
            hr.save(result_folder / image_path.name)

            lr_tensor, _ = preprocess(lr, device)
            hr_tensor, _ = preprocess(hr, device)

            # Super-resolve
            start_time = time.time()
            with torch.no_grad():
                preds = model(lr_tensor).clamp(0.0, 1.0)
            elapsed_time = time.time() - start_time

            # Calculate metrics
            psnr_value = metrics_calculator.calculate_psnr(preds, hr_tensor).item()
            ssim_value = metrics_calculator.calculate_ssim(preds, hr_tensor).item()

            total_psnr += psnr_value
            total_ssim += ssim_value
            total_time += elapsed_time
            num_images += 1

            # Save SR image
            preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
            bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
            _, ycbcr = preprocess(bicubic, device)

            output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
            output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
            output_image = pil_image.fromarray(output)
            output_image.save(result_folder / f"{image_path.stem}_fsrcnn_x{args.scale}.png")

            # Log metrics
            with open(output_log_path, 'a') as log_file:
                log_file.write(f"{image_path.name} {psnr_value:.2f} {ssim_value:.4f} {elapsed_time:.2f}s\n")

            print(f"Processed {image_path.name} | PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}, Time: {elapsed_time:.2f}s")

        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")

    # Log overall metrics
    avg_psnr = total_psnr / num_images if num_images > 0 else 0
    avg_ssim = total_ssim / num_images if num_images > 0 else 0
    avg_time = total_time / num_images if num_images > 0 else 0

    with open(output_log_path, 'a') as log_file:
        log_file.write(f"\nAverage PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.4f}, Average Time: {avg_time:.2f}s\n")

    print(f"\nAverage PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.4f}, Average Time: {avg_time:.2f}s")
