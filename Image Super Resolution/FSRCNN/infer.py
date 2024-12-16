import argparse
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import FSRCNN
from utils import convert_ycbcr_to_rgb, preprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, default='model/fsrcnn_x4.pth', help="Path to pretrained FSRCNN weights.")
    parser.add_argument('--input-folder', type=str, default='dataset/set5', help="Path to input folder containing LR images.")
    parser.add_argument('--scale', type=int, default=4, help="Upscale factor.")
    parser.add_argument('--result-folder', type=str, default='results', help="Path to save output images.")
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

    # Prepare result folder
    result_folder = Path(args.result_folder)
    result_folder.mkdir(parents=True, exist_ok=True)

    # Process images in input folder
    input_folder = Path(args.input_folder)
    for image_path in input_folder.glob("*.*"):
        try:
            # Load and preprocess the image
            image = pil_image.open(image_path).convert('RGB')

            # Resize to make dimensions divisible by scale
            image_width = (image.width // args.scale) * args.scale
            image_height = (image.height // args.scale) * args.scale
            lr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)

            lr_tensor, _ = preprocess(lr, device)

            # Super-resolve the image
            with torch.no_grad():
                preds = model(lr_tensor).clamp(0.0, 1.0)

            # Post-process and save the SR image
            preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
            bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
            _, ycbcr = preprocess(bicubic, device)

            output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
            output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
            output_image = pil_image.fromarray(output)
            output_image.save(result_folder / f"{image_path.stem}_fsrcnn_x{args.scale}.png")

            print(f"Processed {image_path.name} -> Saved as {image_path.stem}_fsrcnn_x{args.scale}.png")

        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
