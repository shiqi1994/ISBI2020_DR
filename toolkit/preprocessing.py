#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 18:17:27 2020

@author: sqtang

This scripts has been taken from :
    https://www.kaggle.com/ratthachat/aptos-eye-preprocessing-in-diabetic-retinopathy#2.-Try-Ben-Graham's-preprocessing-method.

The following functions perform effective DR data preprocessing:
    1. Reducing lighting-condition effects
    2. Cropping uninformative area and resize
and saved preprocessed images to specific folder.  

The argument 'sigmaX' in function 'load_ben_color' can be modified. 
Here we set 'sigmaX' to 10 since Ben got better performance on that setting. 
(maybe other values could be better, like 30)

"""
import os
import cv2
import numpy as np

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img
    
def load_ben_color(path, IMG_SIZE, sigmaX=10):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
        
    return image

if __name__ == '__main__':
    
    # modify to the specific dictionary and img_size
    ori_dir = '/home/sqtang/Downloads/aptos2019-blindness-detection/train_images'
    tar_dir = '/home/sqtang/Downloads/Kaggle2019_prep_sigma10'
    IMG_SIZE = 512
    
    if not os.path.exists(tar_dir):
        os.mkdir(tar_dir)
    img_name = os.listdir(ori_dir)
    for num, item in enumerate(img_name):
        ori_img_path = os.path.join(ori_dir, item)
        proc_img = load_ben_color(ori_img_path, IMG_SIZE)
        tar_img_path = os.path.join(tar_dir, item)
        BGR_img = cv2.cvtColor(proc_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(tar_img_path, BGR_img)
        print('{}/{}'.format(num+1, len(img_name)))
