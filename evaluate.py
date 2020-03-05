#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 12:56:13 2020

@author: sqtang
"""
import os
import time
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import model_selection

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from data_utils import generate_stem_dataset, ScheduledWeightedSampler
from utils import print_msg
from metrics import classify, accuracy, quadratic_weighted_kappa

def evaluate(CONFIG):
    
     #creat result folder
    if not os.path.isdir(CONFIG['SAVE_PATH']):
        os.makedirs(CONFIG['SAVE_PATH'])
        
    # creat dataset    
    test_dataset = generate_stem_dataset(CONFIG['DATA_PATH'],
                                         CONFIG['INPUT_SIZE'],
                                         CONFIG['DATA_AUGMENTATION'],
                                         cv=False,
                                         mode='evaluate')
    
    # creat dataloader
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE'],
                             num_workers=CONFIG['NUM_WORKERS'],
                             shuffle=False)
    # define model
    model_name = CONFIG['MODEL_NAME']
    model = EfficientNet.from_pretrained(model_name)
    feature = model._fc.in_features
    model._fc = nn.Linear(in_features=feature, out_features=1, bias=True)
    
    #multi-gpu setting
    torch.cuda.set_device(CONFIG['GPU_NUM'][0])
    model = torch.nn.DataParallel(model, device_ids=CONFIG['GPU_NUM']).to(device=torch.device('cuda'))
    
    # load pretrained weights
    if CONFIG['PRETRAINED_PATH']:
        state_dict = torch.load(CONFIG['PRETRAINED_PATH'])
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module' not in k:
                k = 'module.'+k
            else:
                k = k.replace('features.module.', 'module.features.')
            new_state_dict[k]=v
        model.load_state_dict(new_state_dict)
    
    # evaluate 
    model.eval()
    torch.set_grad_enabled(False)
    
    correct = 0
    total = 0
    
    all_targ = torch.tensor([]).to(dtype=torch.int64).cuda()
    all_pred = torch.tensor([]).to(dtype=torch.int64).cuda()
    logit_pred_y = [] 
    logit_targ_y = []
    for test_data in test_loader:
        X, y = test_data
        X, y = X.cuda(), y.cuda()
        
        y_pred = model(X)
        
        y_pred_classified = y_pred.view(-1).clone()
        for i in range(len(y_pred)):
            y_pred_classified[i] = classify(y_pred[i])
        
        all_pred = torch.cat((all_pred, y_pred_classified.to(torch.int64)))
        all_targ = torch.cat((all_targ, y.to(torch.int64)))
        
        total += y.size(0)
        correct += accuracy(y_pred.cpu(), y.cpu().float()) * y.size(0)
        
        logit_pred_y += list(y_pred.view(-1).cpu().numpy())
        logit_targ_y += list(y.cpu().float().numpy())
        
    acc = round(correct / total, 4)
    c_matrix, kappa = quadratic_weighted_kappa(all_targ.cpu().numpy(), all_pred.cpu().numpy())
    
    ks_dataframe = pd.DataFrame({'pred':logit_pred_y,
                                'targ':logit_targ_y})
    ks_dataframe.to_csv(os.path.join(CONFIG['SAVE_PATH'],model_name+'_eval_results.csv'),index=False,sep=',')    
        
    print('==============================')
    print('Test acc: {}'.format(acc))
    print('Confusion Matrix:\n{}'.format(c_matrix))
    print('quadratic kappa: {}'.format(kappa))
    print('==============================')