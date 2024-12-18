# DiffIR-SR

This repository contains the implementation of **DiffIR-SR**, a diffusion-based model for single-image super-resolution. The model supports multiple scaling factors (**1x**, **2x**, and **4x**) with pre-trained weights available.

---

## Installation

1. **Install dependencies**:
   Ensure you have `torch`, `torchvision`, and other dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **Uninstall any conflicting library**:
   ```bash
   pip uninstall ldm
   ```

---

## Fixing the Import Issue

When you first run the following command:

```bash
python infer.py
```

You should encounter an error similar to the following:

```
ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'
```

The error can look somehow like this:
![Alt text](https://github.com/quaqducc/RestoraVision/blob/main/Image%20Super%20Resolution/DiffIR-SR/figs/infer-instruction.png)

This error occurs because line 8 in the file `degradations.py` in the `basicsr\data\` package has an outdated import statement. To fix it:

1. Open the file by Ctrl+Click on link to the file 
   ```bash
   basicsr/data/degradations.py
   ```
2. Delete line 8:
   ```python
   from torchvision.transforms.functional tensor import rgb_to_grayscale
   ```
3. Replace it with the correct import statement:
   ```python
   from torchvision.transforms._functional_tensor import rgb_to_grayscale
   ```

After making these changes, rerun the command:

```bash
python infer.py
```

---

## Usage

To perform inference using the **DiffIR-SR** model, use the following command:

```bash
python infer.py --scale <scale_factor> --model_path <path_to_model>
```

### Arguments:
- `--scale`: The upscaling factor. Supported values are `1`, `2`, and `4`.
- `--model_path`: Path to the pre-trained model weights for the specified scale.

---

## Pre-trained Model Paths

Below are the pre-trained model paths corresponding to each scaling factor:

| Scale Factor | Model Path                                 |
|--------------|--------------------------------------------|
| 1x           | `./DiffIR/weights/RealworldSR-DiffIRS1x.pth` |
| 2x           | `./DiffIR/weights/RealworldSR-DiffIRS2x.pth` |
| 4x           | `./DiffIR/weights/RealworldSR-DiffIRS4x.pth` |

---

## Example Commands

1. **Scale 1x**:
   ```bash
   python infer.py --scale 1 --model_path ./DiffIR/weights/RealworldSR-DiffIRS1x.pth
   ```

2. **Scale 2x**:
   ```bash
   python infer.py --scale 2 --model_path ./DiffIR/weights/RealworldSR-DiffIRS2x.pth
   ```

3. **Scale 4x**:
   ```bash
   python infer.py --scale 4 --model_path ./DiffIR/weights/RealworldSR-DiffIRS4x.pth
   ```

---

## Results

The model takes a low-resolution image as input and outputs a high-resolution image based on the specified scale.

---
