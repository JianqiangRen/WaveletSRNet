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
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torchvision.transforms.functional as F
from networks import *
from dataset import *

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--upscale', type=int, default=2, help='resize scale')
    parser.add_argument('--num_layers_res', type=int, help='number of the layers in residual block', default=2)
    parser.add_argument('--model', type=str)
    parser.add_argument('--img', type=str)
    opt = parser.parse_args()
    
    wavelet_rec = WaveletTransform(scale=opt.upscale, dec=False)  # wavelet recomposition
 
    srnet = NetSR(opt.upscale, num_layers_res=opt.num_layers_res)

    
    print("=> loading model '{}'".format(opt.model))
    weights = torch.load(opt.model,map_location='cpu')
    pretrained_dict = weights['model'].state_dict()
    model_dict = srnet.state_dict()
    # print(model_dict)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    srnet.load_state_dict(model_dict)
    print("=>model loaded")
    print(srnet)
 
    input_transform = transforms.ToTensor()

    img = Image.open(opt.img)
    img = img.convert("RGB")
    img_lr = img.resize((int(img.size[0] / 4), int(img.size[1] / 4)), Image.BICUBIC)
    img_lr.save("lr.png")
    img_bic_hr = img_lr.resize((int(img_lr.size[0]*4), int(img_lr.size[1]*4)), Image.BICUBIC)

    img_lr = input_transform(img_lr)
    img_lr = torch.unsqueeze(img_lr, 0)
    print(img_lr.shape)
 
    srnet.eval()
    wavelets = srnet(img_lr)
 
    prediction = wavelet_rec(wavelets)
 
    img = prediction.cpu()
    im = img.data.numpy().astype(np.float32)
    
    im = im.transpose(0, 2, 3, 1)
    print(im.shape)
    im = im[0] * 255
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    img_bic_hr.save("res_bicubic.png")
    cv2.imwrite("res_wsrnet.png", im)