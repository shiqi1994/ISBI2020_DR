#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:28:14 2020

@author: sqtang
"""
import os
import time
import json
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

def train(CONFIG):
    
    #creat result folder and save config as txt file
    t = time.strftime('%Y_%m_%d_%H_%M_%S')
    results_dir = os.path.join(CONFIG['SAVE_PATH'],t)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    with open(os.path.join(results_dir,'Settings.txt'), 'w') as file:
        file.write(json.dumps(CONFIG))
    
    # creat train dataset
    train_dataset, test_dataset = generate_stem_dataset(CONFIG['DATA_PATH'],
                                          CONFIG['INPUT_SIZE'],
                                          CONFIG['DATA_AUGMENTATION'],
                                          cv=False)
        
    #define dynamic weighted resampler
    train_targets = [sampler[1] for sampler in train_dataset.samples]
    weighted_sampler = ScheduledWeightedSampler(len(train_dataset), train_targets, True)
        
    #creat dataloader
    train_loader = DataLoader(train_dataset,
                          batch_size=CONFIG['BATCH_SIZE'],
                          sampler=weighted_sampler,
                          num_workers=CONFIG['NUM_WORKERS'],
                          drop_last=False,
                          shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE'],
                             num_workers=CONFIG['NUM_WORKERS'],
                             shuffle=False)
        
    # define model
    model_name = CONFIG['MODEL_NAME']
    model = EfficientNet.from_pretrained(model_name)
    feature = model._fc.in_features
    model._fc = nn.Linear(in_features=feature, out_features=1, bias=True)
    
    torch.cuda.set_device(CONFIG['GPU_NUM'][0])
    model = torch.nn.DataParallel(model, device_ids=CONFIG['GPU_NUM']).to(device=torch.device("cuda"))
        
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

        
    # define loss and optimizer
    if CONFIG['LOSS_FUNC'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif CONFIG['LOSS_FUNC'] == 'MSELoss':
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError
        
    if CONFIG['OPTIMIZER'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=CONFIG['LEARNING_RATE'],
                                    momentum=CONFIG['MOMENTUM'],
                                    nesterov=True,
                                    weight_decay=CONFIG['WEIGHT_DECAY'])
    elif CONFIG['OPTIMIZER'] == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=CONFIG['LEARNING_RATE'],
                                     betas=CONFIG['BETAS'],
                                     eps=CONFIG['EPS'],
                                     weight_decay=CONFIG['WEIGHT_DECAY'])
    else:
        raise NotImplementedError
            
    # learning rate decay
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=CONFIG['MILESTONES'],
                                                        gamma=CONFIG['GAMMA'])
    # train
    max_kappa = 0
    record_epochs, accs, losses = [], [], []
    for epoch in range(1, CONFIG['EPOCHS']+1):
            
        # resampling weight update
        if weighted_sampler:
            weighted_sampler.step()
            
        # learning rate update
        if lr_scheduler:
            lr_scheduler.step()
            if epoch in lr_scheduler.milestones:
                print_msg('Learning rate decayed to {}'.format(lr_scheduler.get_lr()[0]))
                            
        epoch_loss = 0
        correct = 0
        total = 0
        progress = tqdm(enumerate(train_loader))
        for step, train_data in progress:
            X, y = train_data # X.dtype is torch.float32, y.dtype is torch.int64
            X, y = X.cuda(), y.float().cuda()

            # forward
            y_pred = model(X)
            loss = criterion(y_pred.view(-1), y.float())

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # metrics
            epoch_loss += loss.item()
            total += y.size(0)
            correct += accuracy(y_pred.cpu(), y.cpu().float()) * y.size(0)
            avg_loss = epoch_loss / (step + 1)
            avg_acc = correct / total
            
            progress.set_description('Epoch: {}/{}, loss: {:.6f}, acc: {:.4f}'.format(epoch, CONFIG['EPOCHS'], avg_loss, avg_acc))
            
        # save model and kappa score&confusion matrix 
        acc, c_matrix, kappa, all_pred = _eval(model, test_loader, CONFIG)
        print('validation accuracy: {}, kappa: {}'.format(acc, kappa))
        if kappa > max_kappa:
            torch.save(model.state_dict(), os.path.join(results_dir,model_name+'_best_kappa.pth'))
            max_kappa = kappa
            print_msg('Best kappa model save at {}'.format(results_dir))
            print_msg('Confusion matrix with best kappa is:\n', c_matrix)
            np.savetxt(os.path.join(results_dir,'confusion_matrix.csv'), np.array(c_matrix), delimiter = ',')
            with open(os.path.join(results_dir,'kappa_score.txt'), 'w') as f:
                f.write('Best kappa: {}'.format(kappa))

    # record
    record_epochs.append(epoch)
    accs.append(acc)
    losses.append(avg_loss)
    print('\nBest validation kappa score for {}:\n {}'.format(model_name,max_kappa))
    return record_epochs, accs, losses
  

def _eval(model, dataloader, CONFIG):
    model.eval()
    torch.set_grad_enabled(False)
    
    correct = 0
    total = 0
    
    all_targ = torch.tensor([]).to(dtype=torch.int64).cuda()
    all_pred = torch.tensor([]).to(dtype=torch.int64).cuda()
    
    for test_data in dataloader:
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
    acc = round(correct / total, 4)
    c_matrix, kappa = quadratic_weighted_kappa(all_targ.cpu().numpy(), all_pred.cpu().numpy())
    model.train()
    torch.set_grad_enabled(True)
    return acc, c_matrix, kappa, all_pred.cpu().numpy()
