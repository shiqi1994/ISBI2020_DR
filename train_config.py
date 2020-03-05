#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 16:26:13 2020

@author: sqtang
"""

TRAIN_CONFIG = {
    'DATA_PATH': './Data',
    'SAVE_PATH': './Results',
    'MODEL_NAME': 'efficientnet-b4', # efficientnet-b3 b4 b5
    
    'PRETRAINED_PATH': None,
#    'PRETRAINED_PATH': '/home/sqtang/Documents/DR_2020challenge/Results/2020_02_19_14_18_14/efficientnet-b3_best_kappa.pth',
#    'PRETRAINED_PATH': '/home/sqtang/Documents/DR_2020challenge/Results/2020_02_20_00_37_54/efficientnet-b4_best_kappa.pth',
#    'PRETRAINED_PATH': '/home/sqtang/Documents/DR_2020challenge/Results/2020_02_20_12_37_40/efficientnet-b5_best_kappa.pth',

    'CROSS_VALIDATION': False,
    'CV_SEED': 17,
    'LEARNING_RATE': 5e-6,
    'MILESTONES': [20, 30, 40, 50, 100, 150, 200, 250],
    'GAMMA': 0.5,
    'OPTIMIZER': 'ADAM', #'SGD' or 'ADAM'
    'MOMENTUM': 0.9,
    'WEIGHT_DECAY' : 0,
    'BETAS': (0.9,0.999),
    'EPS': 1e-08,
    'INPUT_SIZE': 512,
    'BATCH_SIZE': 32,
    'EPOCHS': 50,
    'NUM_WORKERS': 8,
    'GPU_NUM': [2,0,1,3],
    'LOSS_FUNC': 'MSELoss', # 'CrossEntropyLoss' or 'MSELoss'
    
    'DATA_AUGMENTATION': {
        'scale': (1 / 1.15, 1.15),
        'stretch_ratio': (0.7561, 1.3225),  # (1/(1.15*1.15) and 1.15*1.15)
        'ratation': (-180, 180),
        'translation_ratio': ( 100 / 512, 100 / 512),  # 100 pixel 
        'sigma': 0.5
    }
}