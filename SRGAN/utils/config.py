import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN = "E:\\Workspaces\\My Projects\\SRGAN-from-scratch\\experiments\\checkpoint\\gen.pth.tar"
CHECKPOINT_DISC = "E:\\Workspaces\\My Projects\\SRGAN-from-scratch\\experiments\\checkpoint\\disc.pth.tar"
ROOT_GT = "/kaggle/input/df2kdata/DF2K_train_HR"
ROOT_LR = "/kaggle/input/df2kdata/DF2K_train_LR_bicubic"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1  # 100000
BATCH_SIZE = 16
NUM_WORKERS = 4
SCALE = 4  # values in [2, 3, 4]
HIGH_RES = 128
IMG_CHANNELS = 3

LOW_RES = HIGH_RES // SCALE
if SCALE == 4:
    ROOT_LR = ROOT_LR + "/X4"
elif SCALE == 3:
    ROOT_LR = ROOT_LR + "/X3"
elif SCALE == 2:
    ROOT_LR = ROOT_LR + "/X2"

LOAD_MODEL = False
SAVE_MODEL = True
ROOT_GT = "/kaggle/input/df2kdata/DF2K_train_HR"  # Edit
ROOT_LR = "/kaggle/input/df2kdata/DF2K_train_LR_bicubic"  # Edit
ROOT_VAL_GT = "/kaggle/input/super-resolution-benchmarks/Set14/Set14/GTmod12"  # Edit
ROOT_VAL_LR = "/kaggle/input/super-resolution-benchmarks/Set14/Set14/LRbicx4"  # Edit
OUTPUT_VAL_HR = "E:\\Workspaces\\My Projects\\SRGAN-from-scratch\\experiments\\test"
CHECKPOINT_GEN = "E:\\Workspaces\\My Projects\\SRGAN-from-scratch\\experiments\\checkpoint\\gen.pth.tar"
CHECKPOINT_DISC = "E:\\Workspaces\\My Projects\\SRGAN-from-scratch\\experiments\\checkpoint\\disc.pth.tar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100  # 100000
WARMUP_EPOCHS = 10
BATCH_SIZE = 16
NUM_WORKERS = 4
SCALE = 4  # values in [2, 3, 4]
HIGH_RES = 96
IMG_CHANNELS = 3

LOW_RES = HIGH_RES // SCALE
if SCALE == 4:
    ROOT_LR = ROOT_LR + "/X4"
elif SCALE == 3:
    ROOT_LR = ROOT_LR + "/X3"
elif SCALE == 2:
    ROOT_LR = ROOT_LR + "/X2"

gt_transform = A.ReplayCompose(
    [
        A.RandomCrop(width=HIGH_RES, height=HIGH_RES),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ]
)


def lr_transform(HorizontalFlip=None):
    p_hf = 1 if HorizontalFlip else 0

    transform = A.Compose(
        [
            A.HorizontalFlip(p=p_hf),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            ToTensorV2()
        ]
    )

    return transform


# Apply before upscale an image
test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2()
    ]
)

# Apply in show_img
val_transform = A.Compose(
    [
        ToTensorV2()
    ]
)

# Make sure gt_transform returns transformation parameters
gt_transform = A.ReplayCompose(gt_transform.transforms)
