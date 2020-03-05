#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 19:35:33 2020

@author: sqtang
"""
import argparse
from train_config import TRAIN_CONFIG
from test_config import TEST_CONFIG
from train import train
from evaluate import evaluate
from train_cv import train_cv

def main():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--MODE', type=str, default = None)
    args = parser.parse_args()
    if args.MODE == 'TRAIN':
        if TRAIN_CONFIG['CROSS_VALIDATION']:
            record_epochs, accs, losses = train_cv(TRAIN_CONFIG)
        else:
            record_epochs, accs, losses = train(TRAIN_CONFIG)
            
    if args.MODE == 'TEST':
        evaluate(TEST_CONFIG)
        
        
if __name__ == '__main__':
    main()
