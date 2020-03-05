#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 16:44:29 2020

@author: sqtang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 21:06:34 2020
@author: sqtang

This script rearrange the files of datasets in this way:
    
    root/class_x/1.jpg
    root/class_x/2.jpg
    ...
    root/class_y/a.jpg
    root/class_y/b.jpg
    ...

which is the generic data arrangment for torchvision.dataset.DatasetFolder.

In this script, number of class has been set to 5 according to 5 levels of Diabetic Retinopathy.
"""

import csv
import shutil
import os

#####  modify the following values if you need
label_dir = '/home/sqtang/Documents/DR_2020challenge/toolkit/Label/Kaggle2015_trainLabels.csv'
source_root = '/home/sqtang/Downloads/Kaggle2015_prep_sigma10/'
target_dir = '/home/sqtang/Documents/DR_2020challenge/Data/Kaggle2015_prep_train_sigma10/'
#####

if not os.path.exists(target_dir):
    os.mkdir(target_dir[:-1])
    
for i in range(5):
    if not os.path.exists(target_dir+str(i)):
        os.mkdir(target_dir+str(i))
    
with open(label_dir,'r') as csvfile:
    reader = csv.reader(csvfile)
    reader.__next__()
    img_name_label = [[row[0],row[1]] for row in reader] 
# img_name is list in format of ['10_left', '10_right', '13_left'...]
for step, item in enumerate(img_name_label):
    file_name = item[0]+'.jpeg' # string in form of '10_left.jpg' for example
    
    if item[1] == '0':
        shutil.copy(source_root+file_name, target_dir+'0/'+file_name)
    if item[1] == '1':
        shutil.copy(source_root+file_name, target_dir+'1/'+file_name)
    if item[1] == '2':
        shutil.copy(source_root+file_name, target_dir+'2/'+file_name)
    if item[1] == '3':
        shutil.copy(source_root+file_name, target_dir+'3/'+file_name)
    if item[1] == '4':
        shutil.copy(source_root+file_name, target_dir+'4/'+file_name)
        
    print('Processing: {}/{}'.format(step+1, len(img_name_label)))