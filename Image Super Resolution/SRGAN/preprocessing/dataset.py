import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from copy import deepcopy
import utils.config as config


class MyImageFolder(Dataset):
    def __init__(self, root_gt, root_lr, transform=False):
        super(MyImageFolder, self).__init__()
        self.transform = transform
        self.is_test = False
        self.root_lr = root_lr
        self.root_gt = root_gt

        self.lr_files = os.listdir(root_lr)
        self.gt_files = os.listdir(root_gt)

        paired_data = []
        for lr_file in self.lr_files:
            tmp = deepcopy(lr_file)
            paired_data.append((lr_file, tmp.replace(f'x{config.SCALE}.png', '.png')))
        self.data = paired_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        lr_name, gt_name = self.data[idx]
        lr_path = os.path.join(self.root_lr, lr_name)
        gt_path = os.path.join(self.root_gt, gt_name)

        lr_image, gt_image = np.array(Image.open(lr_path)), np.array(Image.open(gt_path))
        if self.transform:
            # Apply random crop and transformations to GT image
            gt_mask = np.zeros_like(gt_image)[:, :, 0]  # Single channel mask
            gt_transforms = config.gt_transform(image=gt_image)
            gt_image = gt_transforms["image"]

            # Get crop coordinates from the transform params
            crop_params = gt_transforms["replay"]["transforms"][0]["params"]
            x_gt_start, y_gt_start, x_gt_end, y_gt_end = crop_params["crop_coords"]

            # Get Transformation probability params
            horizontal_flip = gt_transforms["replay"]["transforms"][1]["applied"]

            # Create a binary mask for the GT image
            gt_mask[y_gt_start:y_gt_end, x_gt_start:x_gt_end] = 1

            # Resize the mask to LR dimensions
            lr_h, lr_w = lr_image.shape[:2]

            lr_mask = Image.fromarray(gt_mask.astype(np.uint8))
            lr_mask = lr_mask.resize((lr_w, lr_h), Image.NEAREST)
            lr_mask = np.array(lr_mask)

            # Find the coordinates in the LR image
            cropped_lr = None
            lr_coords = np.where(lr_mask == 1)
            if len(lr_coords[0]) > 0:
                y_lr_start, x_lr_start = lr_coords[0].min(), lr_coords[1].min()
                y_lr_end, x_lr_end = lr_coords[0].max() + 1, lr_coords[1].max() + 1
                cropped_lr = lr_image[y_lr_start:y_lr_end, x_lr_start:x_lr_end]

            # Crop and transform the LR image
            lr_transforms = config.lr_transform(horizontal_flip)(image=cropped_lr)
            lr_image = lr_transforms["image"]
        else:
            gt_transforms = config.val_transform(image=gt_image)
            gt_image = gt_transforms["image"]

            lr_transforms = config.val_transform(image=lr_image)
            lr_image = lr_transforms["image"]

        return lr_image, gt_image
