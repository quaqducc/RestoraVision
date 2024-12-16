import torch
import os
import numpy as np
from PIL import Image
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import config


def gradient_penalty(critic, real, fake, device):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake.detach() * (1 - alpha)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_pen = torch.mean((gradient_norm - 1) ** 2)
    return gradient_pen


def save_checkpoint(model, optimizer, epoch=None, filename="my_checkpoint.pth.tar"):
    save_noti = "-> Saving Checkpoint"
    if epoch:
        save_noti = save_noti + f' --- Epoch: {epoch}'
    print(save_noti)
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("-> Loading Checkpoint")
    # Default: weights_only = False
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this -> it will just have learning rate of old checkpoint -> many hour of debugging
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def plot_examples(val_lr_folder, output_folder, gen, n_examples=-1):
    files = os.listdir(val_lr_folder)

    if n_examples <= 0:
        n_examples = len(files)

    gen.eval()
    for idx, file in enumerate(files):
        if idx >= n_examples:
            break
        image = Image.open(os.path.join(val_lr_folder, file))
        with torch.no_grad():
            upscaled_img = gen(
                config.test_transform(image=np.asarray(image))["image"]
                .unsqueeze(0)
                .to(config.DEVICE)
            )

        save_image(upscaled_img * 0.5 + 0.5, os.path.join(output_folder, file))
    gen.train()


def show_example(lr_path, sr_path, gt_path, epoch=None):
    img_paths = [lr_path, sr_path, gt_path]
    titles = ["LR Image", "SR Image", "GT Image"]

    # Create a figure with 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Iterate over the axes, image paths, and titles
    for ax, img_path, title in zip(axes, img_paths, titles):
        # Load and display the image
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis('off')  # Hide axes for cleaner display
        ax.set_title(title, fontsize=12, pad=10)  # Add a title with padding

    # Add a heading for the entire plot
    plt.suptitle(f"Epoch: {epoch}", fontsize=16, y=1.02)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


def val(val_dataset, evaluator, epoch=None):
    n_data = len(val_dataset)
    psnr, ssim, mse, lpips = evaluator.evaluate_metrics(val_dataset)
    log_metrics = {f"Epoch": epoch,
                   f"PSNR/{n_data}": psnr,
                   f"SSIM/{n_data}": ssim,
                   f"MSE/{n_data}": mse,
                   f"LPIPS/{n_data}": lpips}
    return log_metrics
