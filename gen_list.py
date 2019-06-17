# coding=utf-8
# summary: 生成图像的list
# author: Jianqiang Ren
# date:

import argparse
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str,  help='fold containing images')
parser.add_argument('--file', type=str,  help='file list')
args = parser.parse_args()

img_paths = glob.glob(os.path.join(args.path, "*.jpg"))

with open(args.file, "w") as f:
    for idx,  p in  enumerate(img_paths):
        if idx != len(img_paths) -1:
            f.write(os.path.basename(p) + "  0\n")
        else:
            f.write(os.path.basename(p) + "  0")