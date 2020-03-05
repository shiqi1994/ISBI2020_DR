#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 12:35:41 2020

@author: sqtang
"""
import os
import torch
from PIL import Image
from torch.utils.data.sampler import Sampler
from torchvision import transforms, datasets

# The following functions have been taken from YijinHuang's github repository
# https://github.com/YijinHuang/pytorch-DR

#For kaggle2015+2019
#MEAN = [0.5526258586659438, 0.5358703745005223, 0.5261124241401273]
#
#STD = [0.15279997373926116, 0.14627116174744156, 0.11912972190411034]
#
#BALANCE_WEIGHTS = torch.tensor([38785/27614, 38785/2812,
#                                38785/6290, 38785/1066,
#                                38785/1003], dtype=torch.double)
#For Kaggle2015
#MEAN = [0.5526258586659438, 0.5358703745005223, 0.5261124241401273]
#
#STD = [0.15279997373926116, 0.14627116174744156, 0.11912972190411034]
#
#BALANCE_WEIGHTS = torch.tensor([1.3609453700116234, 14.378223495702006,
#                                6.637566137566138, 40.235967926689575,
#                                49.612994350282484], dtype=torch.double)

##For processed ISBI2020
#MEAN = [0.5555938, 0.53418064, 0.5213337]
#
#STD = [0.16183352, 0.1411609, 0.10492506]
#
#U = torch.tensor([[-0.41850981, -0.67490248, 0.6077468],
#                  [-0.59750913, -0.29935143, -0.74388948],
#                  [-0.68398273, 0.67445931, 0.27797889]], dtype=torch.float32)
#
#EV = torch.tensor([0.05315017, 0.00269258, 0.00128307], dtype=torch.float32)
#
#BALANCE_WEIGHTS = torch.tensor([1200/540, 1200/140,
#                                1200/234, 1200/214,
#                                1200/72], dtype=torch.double)
#FINAL_WEIGHTS = torch.as_tensor([1, 2, 2, 2, 2], dtype=torch.double)

##For processed kaggle2019
MEAN = [0.42150313, 0.22500901, 0.07526358]

STD = [0.2766358, 0.15143834, 0.0826507]

U = torch.tensor([[-0.12569453, -0.8564691, 0.50066113],
                  [-0.46428955, -0.39520053, -0.79262334],
                  [-0.87671894, 0.33208015, 0.3479751]], dtype=torch.float32)

EV = torch.tensor([0.09841777, 0.00683931, 0.00103468], dtype=torch.float32)

BALANCE_WEIGHTS = torch.tensor([3662/1805, 3662/370,
                                3662/999, 3662/193,
                                3662/295], dtype=torch.double)

FINAL_WEIGHTS = torch.as_tensor([1, 2, 2, 2, 2], dtype=torch.double)

def generate_stem_dataset(data_path, input_size, data_aug, cv, mode=None):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            size=input_size,
            scale=data_aug['scale'],
            ratio=data_aug['stretch_ratio']
        ),
        transforms.RandomAffine(
            degrees=data_aug['ratation'],
            translate=data_aug['translation_ratio'],
            scale=None,
            shear=None
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(tuple(MEAN), tuple(STD))])
#        KrizhevskyColorAugmentation(sigma=data_aug['sigma'])
#    ])

    test_transform = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        transforms.Normalize(tuple(MEAN), tuple(STD))
    ])

    def load_image(x):
        return Image.open(x)
    if not mode:
        if cv:
            return generate_dataset_cv(data_path, load_image, ('jpg', 'jpeg','png'), train_transform, test_transform)
        else:
            return generate_dataset(data_path, load_image, ('jpg', 'jpeg','png'), train_transform, test_transform)
    else:
        return generate_dataset(data_path, load_image, ('jpg', 'jpeg','png'), train_transform, test_transform, mode)


def generate_dataset_cv(data_path, loader, extensions, train_transform, test_transform):
#    train_path = os.path.join(data_path, 'ISBI2020_prep_Mix_sigma10')
#    train_path = os.path.join(data_path, 'ISBI2020_prep_Train_sigma10')
    train_path = os.path.join(data_path, 'Kaggle2019_prep_train_sigma10')  
    train_dataset = datasets.DatasetFolder(train_path, loader, extensions, transform=train_transform)

    return train_dataset

def generate_dataset(data_path, loader, extensions, train_transform, test_transform, mode=None):
    if not mode:
    #    train_path = os.path.join(data_path, 'Kaggle2015+2019_prep_train_sigma10')  
        train_path = os.path.join(data_path, 'Kaggle2015_prep_train_sigma10')  
    
#        train_path = os.path.join(data_path, 'ISBI2020_prep_Train_sigma10')
        test_path = os.path.join(data_path, 'ISBI2020_prep_Test_sigma10')
    
        train_dataset = datasets.DatasetFolder(train_path, loader, extensions, transform=train_transform)
        test_dataset = datasets.DatasetFolder(test_path, loader, extensions, transform=test_transform)
        
        return train_dataset, test_dataset
    
    else:
        test_path = os.path.join(data_path, 'ISBI2020_prep_Test_sigma10')
        test_dataset = datasets.DatasetFolder(test_path, loader, extensions, transform=test_transform)

        return test_dataset

class ScheduledWeightedSampler(Sampler):
    def __init__(self, num_samples, train_targets, initial_weight=BALANCE_WEIGHTS,
                 final_weight=FINAL_WEIGHTS, replacement=True):
        self.num_samples = num_samples
        self.train_targets = train_targets
        self.replacement = replacement

        self.epoch = 0
        self.w0 = initial_weight
        self.wf = final_weight
        self.train_sample_weight = torch.zeros(len(train_targets), dtype=torch.double)

    def step(self):
        self.epoch += 1
        factor = 0.975**(self.epoch - 1) # r=0.975 here is a hyperparameter
        self.weights = factor * self.w0 + (1 - factor) * self.wf
        for i, _class in enumerate(self.train_targets):
            self.train_sample_weight[i] = self.weights[_class]
    def __iter__(self):
        return iter(torch.multinomial(self.train_sample_weight, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples
    
    
    
class KrizhevskyColorAugmentation(object):
    def __init__(self, sigma=0.5):
        self.sigma = sigma
        self.mean = torch.tensor([0.0])
        self.deviation = torch.tensor([sigma])

    def __call__(self, img, color_vec=None):
        sigma = self.sigma
        if color_vec is None:
            if not sigma > 0.0:
                color_vec = torch.zeros(3, dtype=torch.float32)
            else:
                color_vec = torch.distributions.Normal(self.mean, self.deviation).sample((3,))
            color_vec = color_vec.squeeze()

        alpha = color_vec * EV
        noise = torch.matmul(U, alpha.unsqueeze(dim=1))
        noise = noise.view((3, 1, 1))
        return img + noise

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={})'.format(self.sigma)