#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 21:15:34 2020

@author: sqtang
"""
import sys
import torch
import pandas as pd
import warnings
from sklearn import model_selection
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore")

import numpy as np
from os import listdir
from os.path import join, isdir
from glob import glob
import cv2
import timeit

# number of channels of the dataset image, 3 for color jpg, 1 for grayscale img
# you need to change it to reflect your dataset
CHANNEL_NUM = 3


def cal_dir_stat(root):
    cls_dirs = [d for d in listdir(root) if isdir(join(root, d))]
    pixel_num = 0 # store all pixel number in the dataset
    channel_sum = np.zeros(CHANNEL_NUM)
    channel_sum_squared = np.zeros(CHANNEL_NUM)

    for idx, d in enumerate(cls_dirs):
        print("#{} class".format(idx))
        im_pths = glob(join(root, d, "*.jpeg"))

        for path in im_pths:
            im = cv2.imread(path) # image in M*N*CHANNEL_NUM shape, channel in BGR order
            im = im/255.0
            pixel_num += (im.size/CHANNEL_NUM)
            channel_sum += np.sum(im, axis=(0, 1))
            channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

    bgr_mean = channel_sum / pixel_num
    bgr_std = np.sqrt(channel_sum_squared / pixel_num - np.square(bgr_mean))
    
    # change the format from bgr to rgb
    rgb_mean = list(bgr_mean)[::-1]
    rgb_std = list(bgr_std)[::-1]
    
    return rgb_mean, rgb_std

# The script assumes that under train_root, there are separate directories for each class
# of training images.
train_root = '/home/sqtang/Documents/DR_2020challenge/Data/Kaggle2015+2019_prep_train_sigma10'
start = timeit.default_timer()
mean, std = cal_dir_stat(train_root)
end = timeit.default_timer()
print("elapsed time: {}".format(end-start))
print("MEAN = {}\nSTD = {}".format(mean, std))

#CONFIG = {
#    'DATA_PATH': './Data',
#    'PRETRAINED_PATH': None,
#    'LEARNING_RATE': 3e-3,
#    'MILESTONES': [150, 220],
#    'GAMMA': 0.1,
#    'MOMENTUM': 0.9,
#    'WEIGHT_DECAY' : 0.0005,
#    'INPUT_SIZE': 112,
#    'BATCH_SIZE': 100,
#    'EPOCHS': 300,
#    'NUM_WORKERS': 8,
#    'NUM_GPU': 2,
#    'LOSS_FUNC': 'CrossEntropyLoss', # or 'MSELoss'
#    
#    'DATA_AUGMENTATION': {
#        'scale': (1 / 1.15, 1.15),
#        'stretch_ratio': (0.7561, 1.3225),  # (1/(1.15*1.15) and 1.15*1.15)
#        'ratation': (-180, 180),
#        'translation_ratio': (40 / 112, 40 / 112),  # 40 pixel in the report
#        'sigma': 0.5
#    }
#}
# 
#
#train_dataset, test_dataset = generate_stem_dataset(CONFIG['DATA_PATH'],
#                                                CONFIG['INPUT_SIZE'],
#                                                CONFIG['DATA_AUGMENTATION'],
#                                                cv=False)
#train_loader = DataLoader(train_dataset,
#                      batch_size=10,
#                      sampler=None,
#                      num_workers=10,
#                      drop_last=False,
#                      shuffle=False)
#
#
#mean = 0.
#std = 0.
#for images, y in train_loader:
#    batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
#    images = images.view(batch_samples, images.size(1), -1)
#    mean += images.mean(2).sum(0)
#    std += images.std(2).sum(0)
#
#mean /= len(train_loader.dataset)
#std /= len(train_loader.dataset)
