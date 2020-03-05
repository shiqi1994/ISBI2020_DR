#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 13:55:34 2020

@author: sqtang
"""
import os
import csv
import numpy as np
import pickle as pk
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import cohen_kappa_score

from utils import get_BJ_time
from metrics import quadratic_weighted_kappa

def boost_generate_data(folder,qk_np):
    
    root = './Results'
    model_name = ['efficientnet-b3',
                  'efficientnet-b4',
                  'efficientnet-b5']
    
    # creat train dataset
    b3_x_train = []
    for m in range(1,6):
        logits_dir = os.path.join(root,folder[0],model_name[0]+'_fold'+str(m)+'_test_results.csv')
        f_pred = []
        with open(logits_dir,'r') as csvfile:
            reader = csv.reader(csvfile)
            reader.__next__()
            for item in reader: 
                f_pred.append(item[0])
        b3_x_train += f_pred
        
    b4_x_train = []
    for m in range(1,6):
        logits_dir = os.path.join(root,folder[1],model_name[1]+'_fold'+str(m)+'_test_results.csv')
        f_pred = []
        with open(logits_dir,'r') as csvfile:
            reader = csv.reader(csvfile)
            reader.__next__()
            for item in reader: 
                f_pred.append(item[0])
        b4_x_train += f_pred
        
    b5_x_train = [] 
    y_train = []
    for m in range(1,6):
        logits_dir = os.path.join(root,folder[2],model_name[2]+'_fold'+str(m)+'_test_results.csv')
        f_pred, f_targ = [], []
        with open(logits_dir,'r') as csvfile:
            reader = csv.reader(csvfile)
            reader.__next__()
            for item in reader: 
                f_pred.append(item[0])
                f_targ.append(item[1])
        b5_x_train += f_pred
        y_train += f_targ
    
    X_train = np.concatenate([np.array(b3_x_train,dtype=np.float32).reshape(-1,1),
                              np.array(b4_x_train,dtype=np.float32).reshape(-1,1),
                              np.array(b5_x_train,dtype=np.float32).reshape(-1,1)], axis=1)
    Y_train = np.array(y_train,dtype=np.float32)
    
    # creat test dataset
    test_root = './Evaluate_Results'
    b3_x_test = []
    
    for m in range(1,6):
        logits_dir = os.path.join(test_root,model_name[0]+'_fold'+str(m)+'_eval_results.csv')
        f_pred = []
        with open(logits_dir,'r') as csvfile:
            reader = csv.reader(csvfile)
            reader.__next__()
            for item in reader: 
                f_pred.append(item[0])
        b3_x_test.append(f_pred)
   
    b4_x_test = []
    for m in range(1,6):
        logits_dir = os.path.join(test_root,model_name[0]+'_fold'+str(m)+'_eval_results.csv')
        f_pred = []
        with open(logits_dir,'r') as csvfile:
            reader = csv.reader(csvfile)
            reader.__next__()
            for item in reader: 
                f_pred.append(item[0])
        b4_x_test.append(f_pred)

    b5_x_test, y_test = [], []
    for m in range(1,6):
        logits_dir = os.path.join(test_root,model_name[0]+'_fold'+str(m)+'_eval_results.csv')
        f_pred, f_targ = [], []
        with open(logits_dir,'r') as csvfile:
            reader = csv.reader(csvfile)
            reader.__next__()
            for item in reader: 
                f_pred.append(item[0])
                f_targ.append(item[1])
        b5_x_test.append(f_pred) 
        if m == 1:
            y_test += f_targ
    
    b3_x_test_avg = np.average(np.array(b3_x_test,dtype=np.float32), axis=0)  
    b4_x_test_avg = np.average(np.array(b4_x_test,dtype=np.float32), axis=0)  
    b5_x_test_avg = np.average(np.array(b5_x_test,dtype=np.float32), axis=0)  
    
    X_test = np.concatenate([b3_x_test_avg.reshape(-1,1),
                             b4_x_test_avg.reshape(-1,1),
                             b5_x_test_avg.reshape(-1,1)], axis=1)

    Y_test = np.array(y_test,dtype=np.float32)

    return X_train, Y_train, X_test, Y_test


#====================== LightGBM    
if __name__ == '__main__':
    SEED = 2020
    current_time = get_BJ_time()
    print(current_time)
    def qk_np(y, y_pred):
        k = cohen_kappa_score(np.round(y_pred), y, weights='quadratic')
        return k
    folder = ['2020_02_27_15_03_05', '2020_02_27_16_29_02', '2020_02_27_17_55_56']
    X_train, Y_train, X_test, Y_test = boost_generate_data(folder,qk_np)
    
    score = make_scorer(qk_np, greater_is_better=True)
    estimator = lgb.LGBMRegressor(random_state=SEED)
    
    param_grid = {
        'max_depth': [1,2,3],
#        'max_depth': [3],  
        'num_leaves': [2, 5, 10],
#        'learning_rate': [0.08], 
        'learning_rate': [0.01, 0.05, 0.1],
#        'feature_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
        'colsample_bytree': [0.4,0.5,0.6],
#        'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
        'subsample': [0.4,0.5,0.6],
#        'bagging_freq': [8],
#        'bagging_freq': [5, 6, 8],
#        'lambda_l1': [0, 0.1, 0.4],
#        'lambda_l1': [0],
#        'lambda_l2': [15],
#        'lambda_l2': [0, 10, 15, 20],
#        'cat_smooth': [1],
#        'cat_smooth': [1, 10, 15],
    }
    
    gbm = GridSearchCV(estimator, param_grid, cv=5, n_jobs=-1, scoring=score, verbose=1)
    gbm.fit(X_train, Y_train)
    
    print('Best parameters found by grid search are:', gbm.best_params_)
    #    gbm.cv_results_
    print(gbm.best_score_, qk_np(Y_train, gbm.predict(X_train)))
    model_save_name = "lightgbm-{}".format(current_time)
    
    with open(os.path.join('./boost_model', model_save_name+".pkl"), "wb") as f:
        pk.dump(gbm.best_estimator_, f)
    print(model_save_name)
    
    y_pred = gbm.predict(X_test)
    y_pred = np.round(y_pred)
    
    print(qk_np(Y_test, y_pred))


##====================== XGBoost    
#    estimator_xgb = xgb.XGBRegressor(n_jobs=8, random_state=SEED)
#    parameters = {
#                  'max_depth': [3],
#                  'learning_rate': [0.1],
#    #               'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
#                  'min_child_weight': [20],
#    #               'min_child_weight': [0, 2, 5, 10, 20],
#                  'max_delta_step': [2],
#    #               'max_delta_step': [0, 0.2, 0.6, 1, 2],
#                  'subsample': [0.8],
#    #               'subsample': [0.6, 0.7, 0.8, 0.85, 0.95],
#                  'colsample_bytree': [0.7],
#    #               'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
#                  'reg_alpha': [0],
#    #               'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
#                  'reg_lambda': [0.6],
#    #               'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],
#                  'scale_pos_weight': [0.8]
#    #               'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1]
#    }
#    xlf = GridSearchCV(estimator_xgb, parameters, cv=5, n_jobs=16, scoring=score, verbose=1)
#    xlf.fit(X_train, Y_train)
#    print('Best parameters found by grid search are:', xlf.best_params_)
#    print(xlf.best_score_, qk_np(Y_train, xlf.predict(X_train)))
#    model_save_name = "xgboost-{}".format(current_time)
#
#    with open(os.path.join(root, model_save_name+".pkl"), "wb") as f:
#        pk.dump(xlf.best_estimator_, f)
#    print(model_save_name)
#
#
##====================== SVR
#    estimator_svr = SVR()
#    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                         'C': [1, 10, 100, 1000]},
#                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
#    
#    svr = GridSearchCV(estimator_svr, tuned_parameters, cv=5, n_jobs=16, scoring=score, verbose=1)
#    svr.fit(X_train, Y_train)
#    print('Best parameters found by grid search are:', svr.best_params_)
#    print(svr.best_score_, qk_np(Y_train, svr.predict(X_train)))
#    model_save_name = "svr-{}".format(current_time)
#
#    with open(os.path.join(root, model_save_name+".pkl"), "wb") as f:
#    # with open(os.path.join(deployment_dir, "svr-0903_05-26-03.pkl"), "wb") as f:
#        pk.dump(svr.best_estimator_, f)    
#    #     pk.dump(svr, f)
#    
#    print(model_save_name)
#    
##====================== CatBoost    
#    estimator_cb = CatBoostRegressor(random_seed=SEED)
#    params = {
#              'depth':[3,1,2,6,4,5],
#    #           'iterations':[500],
#              'iterations':[250,500,750,1000],
#    #           'learning_rate':[0.2], 
#              'learning_rate':[0.01,0.1,0.2,0.3], 
#              'l2_leaf_reg':[3,1,5,10],
#              'border_count':[100,128, 200, 254, 300]
#             }
#    cb = GridSearchCV(estimator_cb, params, cv=5, n_jobs=16, scoring=score, verbose=1)
#    cb.fit(X_train, Y_train)
#    print('Best parameters found by grid search are:', cb.best_params_)
#    print(cb.best_score_, qk_np(Y_train, cb.predict(X_train)))
#    model_save_name = "cb-{}".format(current_time)
#
#    with open(os.path.join(root, model_save_name+".pkl"), "wb") as f:
#        pk.dump(cb.best_estimator_, f)    
#    
#    print(model_save_name)
    
    
