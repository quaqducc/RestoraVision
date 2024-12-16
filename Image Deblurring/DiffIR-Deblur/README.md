# DiffIR-Deblur

A diffusion-based image restoration model for motion deblurring tasks. This repository contains the implementation and setup instructions to run the model.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Setup Instructions](#setup-instructions)
- [Troubleshooting](#troubleshooting)
- [Run Inference](#run-inference)
- [Results](#results)

---

## Overview
DiffIR-Deblur is a motion deblurring model based on diffusion processes. It takes a blurry input image and produces a sharp, restored version. 

---

## Requirements

Ensure you have the following prerequisites installed:

- Python 3.8 or later
- PyTorch 1.10+ (with CUDA if GPU is available)
- torchvision 0.11+

Install dependencies:
```bash
pip install -r requirements.txt
```
Run this uninstall command to ensure no conflict happened:
```bash
pip uninstall ldm
```

---

## Setup Instructions

When you first run the model using:
```bash
python infer.py
```

you should encounter the following error:
```
ModuleNotFoundError: No module named 'torchvision.transforms.functional tensor'
```
The error can look somehow like this:
![Alt text](https://github.com/quaqducc/RestoraVision/blob/main/Image%20Deblurring/DiffIR-Deblur/figs/infer-instruction.png)
### **Cause**
This error occurs due to an outdated import statement in `basicsr/data/degradations.py`.

### **Fix**
Follow these steps to resolve the issue:
1. Open the file by Ctrl+Click on link to the file:
   ```bash
   basicsr/data/degradations.py
   ```
2. Locate **line 8** and delete it:
   ```python
   from torchvision.transforms.functional.tensor import rgb_to_grayscale
   ```
3. Replace it with the updated import statement:
   ```python
   from torchvision.transforms._functional_tensor import rgb_to_grayscale
   ```

Once the changes are made, save the file.

---

## Run Inference

After fixing the import error, run the model inference again:
```bash
python infer.py
```

### Input
- Place your blurry images in the input folder test\input.

### Output
- The deblurred images will be saved to the output folder test\output.

---


