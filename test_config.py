#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:07:19 2020

@author: sqtang
"""

TEST_CONFIG = {
    'DATA_PATH': './Data',
    'SAVE_PATH': './tr2019_Evaluate_Results',
    'MODEL_NAME': 'efficientnet-b3', # efficientnet-b3 b4 b5
    
#    'PRETRAINED_PATH': None,
    
# pretrained model on Kaggle2015
#    'PRETRAINED_PATH': '/home/sqtang/Documents/DR_2020challenge/Results/2020_02_19_14_18_14/efficientnet-b3_best_kappa.pth',
#    'PRETRAINED_PATH': '/home/sqtang/Documents/DR_2020challenge/Results/2020_02_20_00_37_54/efficientnet-b4_best_kappa.pth',
#    'PRETRAINED_PATH': '/home/sqtang/Documents/DR_2020challenge/Results/2020_02_20_12_37_40/efficientnet-b5_best_kappa.pth',

# pretrained model on Kaggle2015 and finetuned on ISBI2020
    #b5 fold1-5
    'PRETRAINED_PATH': '/home/sqtang/Documents/DR_2020challenge/Results/2020_03_03_18_23_12/fold2_best_kappa.pth',

    'INPUT_SIZE': 512,
    'BATCH_SIZE': 20,
    'NUM_WORKERS': 8,
    'GPU_NUM': [2,0,1,3],

    'DATA_AUGMENTATION': {
        'scale': (1 / 1.15, 1.15),
        'stretch_ratio': (0.7561, 1.3225),  # (1/(1.15*1.15) and 1.15*1.15)
        'ratation': (-180, 180),
        'translation_ratio': ( 100 / 512, 100 / 512),  # 100 pixel 
        'sigma': 0.5
    }
}