import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from basicsr.utils import  img2tensor
from torch.utils import data as data
import cv2
from basicsr.utils.img_util import tensor2img
from DiffIR.archs.S2_arch import DiffIRS2
from basicsr.utils import  img2tensor
import argparse
from torch.nn import functional as F

def pad_test(lq, window_size): 
    mod_pad_h, mod_pad_w = 0, 0
    _, _, h, w = lq.size()
    if h % window_size != 0:
        mod_pad_h = window_size - h % window_size
    if w % window_size != 0:
        mod_pad_w = window_size - w % window_size
    lq = F.pad(lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return lq,mod_pad_h,mod_pad_w

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--model_path', type=str, default='./DiffIR/weights/Deblurring-DiffIRS2.pth')
    parser.add_argument('--im_path', type=str, default='./test/input/')
    parser.add_argument('--res_path', type=str, default='./test/output/')
    args = parser.parse_args()

    os.makedirs(args.res_path, exist_ok=True)
    model = DiffIRS2( n_encoder_res = 5, dim = 48, num_blocks = [3,5,6,6], num_refinement_blocks = 4, heads = [1,2,4,8], ffn_expansion_factor = 2,LayerNorm_type= "WithBias")
    loadnet = torch.load(args.model_path, map_location=torch.device('cpu'))
    model.load_state_dict(loadnet['params_ema'], strict=True)
    model.to('cuda:0')
    model.eval()

    im_list = os.listdir(args.im_path)
    im_list.sort()
    im_list = [name for name in im_list if name.endswith('.png') or name.endswith('.jpg')]

    for name in im_list:
        path = os.path.join(args.im_path, name)
        im = cv2.imread(path)
        im = img2tensor(im)
        im = im.unsqueeze(0).cuda(0)/255.
        lq,mod_pad_h,mod_pad_w= pad_test(im, 4)
        with torch.no_grad():
            sr = model(lq)
        _, _, h, w = sr.size()
        sr = sr[:, :, 0:h - mod_pad_h * args.scale, 0:w - mod_pad_w * args.scale]
        im_sr = tensor2img(sr, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1))
        save_path = os.path.join(args.res_path, name.split('.')[0]+'_out'+name[-4:])
        cv2.imwrite(save_path, im_sr)
        print(save_path)
