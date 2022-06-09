"""
prepare data, train and related experiments
"""
import logging
from pathlib import Path
import time 
import itertools
from itertools import combinations, product
import numpy as np
import pandas as pd 
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, roc_auc_score
import torch 
from torch_geometric.utils import structured_negative_sampling
# import pickle
import gc
import json
import random

# model 
import xgboost as xgb
# set 
np.random.seed(2021)
random.seed(2021)

##### data preparation
def split_mask(df, masks = ['train_mask', 'val_mask', 'test_mask']):
    """
    mini function to split df based on the mask
    """
    res = []
    for mask in masks:
        mask_c = df[mask].to_numpy()
        df_mask = df[mask_c]
        res.append(df_mask)
    return res

def create_negatives(src_list, dst_list, size):
    """
    mini function to create negative pairs, similar to  tgn
    """
    src_set = np.unique(src_list)
    dst_set = np.unique(dst_list)
    
    src_index = np.random.randint(0, len(src_set), size)
    dst_index = np.random.randint(0, len(dst_set), size)
    
    return src_set[src_index], dst_set[dst_index]


def get_features(author_ls, feat_np, author_ls2 = None):
    """
    mini function to create features based on author_ls
    """
    if author_ls2 is None:
        feats = np.take(feat_np, author_ls, axis = 0)
    else:
        feats1 = np.take(feat_np, author_ls, axis = 0)
        feats2 = np.take(feat_np, author_ls2, axis = 0)
        feats = np.concatenate((feats1, feats2), axis =1)
    return feats


def merge_pn(pos_ls, neg_ls, pos_val = 1, neg_val = 0, pos_ls2 = None, neg_ls2 = None, seed = 2021):
    """
    insert the negative list into the postive list while keep in the relative order of the postive list
    
    """
    random.seed(seed)
    assert len(pos_ls) == len(neg_ls)
    
    merged_ls = pos_ls 
    merged_label = [pos_val] * len(pos_ls)
    if (pos_ls2 is None and neg_ls2 is None):
        for i in range(len(neg_ls)):
            indx = random.randint(0, len(merged_ls))
            merged_ls = np.insert(merged_ls, obj = indx, values = neg_ls[i])
            merged_label = np.insert(merged_label, obj = indx, values = neg_val)
            
        return merged_ls, merged_label 
    else:
        assert len(pos_ls) == len(neg_ls) == len(pos_ls2) == len(neg_ls2)
        merged_ls2 = pos_ls2 
        for i in range(len(neg_ls2)):
            indx = random.randint(0, len(merged_ls))
            merged_ls = np.insert(merged_ls, obj = indx, values = neg_ls[i])
            merged_label = np.insert(merged_label, obj = indx, values = neg_val)
            merged_ls2 = np.insert(merged_ls2, obj = indx, values = neg_ls2[i])
    
        return merged_ls, merged_label, merged_ls2


def prepare_training(df, split_on_mask = True, get_negs = True, merge = True, get_feats = True):
    pass
    

#### training related  
def create_log(args):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    Path(args.savepath +"log/").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler( args.savepath + 'log/{}.log'.format(str(time.time())))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)
    
    return logger


def train_epoch_oneShot(logger, epochs, X_train, y_train, X_val, y_val, savepath, \
                        params = {  'eta':0.002,
                                    'eval_metric': ['logloss', 'auc'],
                                    'objective':'binary:logistic',
                                    'process_type': 'default',
                                    'refresh_leaf': True,
                                    'tree_method': 'gpu_hist',
                                    'max_depth': 6}, patience_trees = 15):
    """
    X_train, y_train, X_val, y_val need to be numpy array
    """
    one_shot_model_itr = None
    best_auc, best_ap, best_epoch = 0., 0., 0.
   
    train = xgb.DMatrix(X_train, y_train)
    valid = xgb.DMatrix(X_val, y_val)
    
    for i in range(epochs):
        start_epoch = time.time()
        one_shot_model_itr = xgb.train(params, 
                                       dtrain= train, 
                                       xgb_model= one_shot_model_itr,
                                       evals = [(train, 'train'), (valid, 'valid')],
                                       early_stopping_rounds = patience_trees)
        
        y_pr = one_shot_model_itr.predict(xgb.DMatrix(X_val))
        auc = roc_auc_score(y_val, y_pr)
        ap = average_precision_score(y_val, y_pr)
        epoch_time = time.time() - start_epoch
        
        logger.info('epoch {}: took  {:.2f}s, val AUC {}, val AP {} '.format(i, epoch_time, auc, ap))
        if ap > best_ap:
            best_ap = ap 
            best_auc = auc
            best_epoch = i
            one_shot_model_itr.save_model(savepath +'best_bsl.json')
            
 
    logger.info('best val AUC {}, val AP {}, found at epoch: {}'.format(best_auc, best_ap, best_epoch)) 
    
    model_xgb = xgb.Booster()
    model_xgb.load_model(savepath +'best_bsl.json')
    return one_shot_model_itr, model_xgb




def predictions_oneShot(logger, model, X_test, y_test):
    y_pred =  model.predict(xgb.DMatrix(X_test))
    test_auc = roc_auc_score(y_test, y_pred)
    test_ap = average_precision_score(y_test, y_pred) 
    logger.info('test AUC {}, test AP {}'.format(test_auc, test_ap))
    logging.shutdown()
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            
            
def easy_train(logger, clf, X_train, y_train, X_val, y_val, path = '20192020_pubs'):
    start_train = time.time()
    clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)])
    val_preds = clf.predict(X_val)
    total_train = time.time() - start_train
    auc = roc_auc_score(y_val, val_preds)
    ap = average_precision_score(y_val, val_preds)
    clf.save_model(path + 'bsl.json')
    logger.info('training took  {:.2f}s, val AUC {}, val AP {} '.format(total_train, auc, ap))
    
    return clf

def easy_predictions(logger, clf, X_test, y_test):
    start_test = time.time()
    test_preds = clf.predict(X_test)
    total_test = time.time() - start_test
    test_auc = roc_auc_score(y_test, test_preds)
    test_ap = average_precision_score(y_test, test_preds) 
    logger.info('test took {:.2f}s, test AUC {}, test AP {}'.format(total_test, test_auc, test_ap))
    logging.shutdown()
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()

# maybe do some batch training as well, let's see






######PART2. for lightGCN basline ##########################
# function which random samples a mini-batch of positive and negative samples
def sample_mini_batch(batch_size, i, edge_index):
    """Randomly samples indices of a minibatch given an adjacency matrix

    Args:
        batch_size (int): minibatch size
        edge_index (torch.Tensor): 2 by N list of edges

    Returns:
        tuple: user indices, positive item indices, negative item indices
    """
    edges = structured_negative_sampling(edge_index)
    edges = torch.stack(edges, dim=0)
    indices = list(range(batch_size * i,  min(batch_size* (i+1), edge_index.shape[1])))
    batch = edges[:, indices]
    user1_indices, pos_user2_indices, neg_user2_indices = batch[0], batch[1], batch[2]
    return user1_indices, pos_user2_indices, neg_user2_indices