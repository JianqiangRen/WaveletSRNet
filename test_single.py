# coding=utf-8
# summary:
# author: Jianqiang Ren
# date:

import argparse
from networks import NetSR
import argparse
import torch
import glob
import os
import cv2
import torchvision.transforms as transforms
import numpy as np
import torchvision.transforms.functional as F
from networks import *


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--upscale', type=int, default=2, help='resize scale')
    parser.add_argument('--num_layers_res', type=int, help='number of the layers in residual block', default=2)
    parser.add_argument('--model', type=str)
    parser.add_argument('--img', type=str)
    opt = parser.parse_args()
    
    wavelet_rec = WaveletTransform(scale=opt.upscale, dec=False)  # wavelet recomposition
    
    
    img = cv2.imread(opt.img)
    
    srnet = NetSR(opt.upscale, num_layers_res=opt.num_layers_res)
    srnet.eval()
    
    print("=> loading model '{}'".format(opt.model))
    weights = torch.load(opt.model,map_location='cpu' )
    pretrained_dict = weights['model'].state_dict()
    model_dict = srnet.state_dict()
    # print(model_dict)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    srnet.load_state_dict(model_dict)
    print("=>model loaded'{}'".format(opt.model))
    
    input_transform = transforms.ToTensor()
    
    a = F.to_pil_image(img)
    a=input_transform(a)
    a = torch.unsqueeze(a, 0)
    wavelets = srnet(a)
    
    prediction = wavelet_rec(wavelets)
    
    img = prediction.cpu()
    im = img.data.numpy().astype(np.float32)
    
    im = im.transpose(0, 2, 3, 1)