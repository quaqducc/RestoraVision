import numpy as np
import torch
import torch.optim as optim
import argparse
import os
from model.model import Generator
from utils.utils import load_checkpoint, save_image
from utils import config
from PIL import Image
import torchvision


def parse_args():
    parser = argparse.ArgumentParser(description='SRGAN Inference')
    parser.add_argument('--img_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='infer_results', help='Output directory')
    parser.add_argument('--checkpoint_path', type=str, default=config.CHECKPOINT_GEN,
                        help='Path to generator checkpoint')
    return parser.parse_args()


def prepare_image(image_path):
    image = Image.open(image_path)
    return config.test_transform(image=np.asarray(image))["image"].unsqueeze(0).to(config.DEVICE)


def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize generator and move to device
    gen = Generator(in_channels=3).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))

    # Load checkpoint
    load_checkpoint(
        args.checkpoint_path,
        gen,
        opt_gen,
        config.LEARNING_RATE
    )

    # Set generator to evaluation mode
    gen.eval()

    # Prepare input image
    input_image = prepare_image(args.img_path).to(config.DEVICE)

    # Generate super-resolution image
    with torch.no_grad():
        upscaled_image = gen(input_image)

    # Save the result
    output_filename = os.path.join(args.output_dir,
                                   f"SR_{os.path.basename(args.img_path)}")
    save_image(upscaled_image * 0.5 + 0.5, output_filename)
    print(f"SR Image saved to: {output_filename}")


if __name__ == "__main__":
    main()
