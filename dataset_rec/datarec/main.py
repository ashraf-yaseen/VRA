#general 
import os
import argparse
import pickle
import dill

#you cannot live without 
from tqdm import trange
import pandas as pd
import numpy as np
import time
#import matplotlib.pyplot as plt
import random
from termcolor import colored
from sklearn.feature_extraction.text import TfidfVectorizer

#pip install transformers
#pytorch related
import torch
import torch.nn as nn
import torch.nn.functional as F

#bert related
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, BertConfig
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

#self-defined
from dataProcessing_bert import DataProcess
import utils_bert as ut 
from clfbert import clfModel
from eval_metrics import Metrics

torch.autograd.set_detect_anomaly(True)

#let's see for all in a simple file 
def main():
    
    #for calling the file from terminal 
    parser = argparse.ArgumentParser(description = 'BERT model for GEO to paper recommendation')
    
    #uncomment this when running on terminal, and comment those below
    parser.add_argument('-data_path', type = str, default = 'data/', 
                        help = 'complete path to the training data [default:data/]')
    parser.add_argument('-load_pretrained', type = bool, default = False,
                        help = 'whether to load pretrained embeddings & tokenizer [default:False]')
    parser.add_argument('-load_path', type = str, default = 'model_save_v2/', 
                        help = '''path where fine-tuned (on our task) embeddings & tokenizer  
                               are saved [default:model_save_v2/]''')
    parser.add_argument('-split', type = bool, default = True, 
                        help = '''whether to split the data (for training) or not (for prediction)[default: True]''')
    parser.add_argument('-newSplit', type = bool, default = True, 
                        help = '''whether to split the data for recommendation metrics calculation[default: True]''')
    parser.add_argument('-cuda_device', type = int, default = 0, 
                        help = 'if has cuda, device index to be used [default:0]')
    parser.add_argument('-learning_rate', type = float, default = 2e-5, 
                        help = 'learning rate of optimizer [default:2e-5]')
    parser.add_argument('-epsilon', type = float, default = 1e-8, 
                        help = 'epsilon of optimizer [default:1e-8]')
    parser.add_argument('-train_epochs', type = int, default = 4, 
                        help = 'fine tune epoch numbers [default: 4]')
    parser.add_argument('-plot_train', type = bool, default = True, 
                        help = 'Whether to plot training stats [default: True]')
    args = parser.parse_args()
    
    """
    #do aruguments here when not calling from terminal/inside jupyter notebook 
    args = parser.parse_args([])
    args.data_path = 'data/'
    args.load_pretrained = False
    args.load_path= 'model_save_v2/'
    args.split = True
    args.newSplit= True
    args.cuda_device = 0
    args.learning_rate = 2e-5
    args.epsilon = 1e-8
    args.train_epochs = 4 
    args.plot_train = True
    """
    #make sure results are replicable
    seed_val = 1234
    ut.set_seed(seed_val)
    
    #load dataloader
    dp =  DataProcess(path= args.data_path,
          load_pretrained = args.load_pretrained, 
          load_path = args.load_path,
          split = args.split,
          newSplit = args.newSplit)
    dp.dataframize_()
    train_loader, _, valid_loader, test_loader = dp.dataloaderize_() #dataloader right here, len of records 83512, 10816, 25016 
    
    #check device
    if torch.cuda.is_available():
        use_cuda = torch.device('cuda:' + str(args.cuda_device))
    else:
        use_cuda = torch.device('cpu')
        
    #load model for bert 
    model = clfModel(load_pretrained = args.load_pretrained, load_path = args.load_path).model
    model.to(use_cuda)

    """ 
    some sanity check for debugging, can be ignored
    print(len(train_loader)* dp.batch_size, len(valid_loader)*dp.batch_size, len(test_loader)*dp.batch_size)
    print(dp.df.iloc[dp.train_idx,:].pmid.nunique())
    print(dp.df.iloc[dp.valid_idx,:].pmid.nunique())
    print(dp.df.iloc[dp.test_idx,:].pmid.nunique())
    """

    #optimizer and scheduler
    optimizer = AdamW(model.parameters(),
                      lr = args.learning_rate,
                      eps = args.epsilon)

    # Create the learning rate scheduler.
    total_steps = len(train_loader) * args.train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    
    #train and valid 
    training_stats = ut.train(epochs = args.train_epochs, 
                                     model = model,
                                     train_loader = train_loader, 
                                     valid_loader = valid_loader, 
                                     optimizer = optimizer, 
                                     scheduler = scheduler, 
                                     use_cuda = use_cuda,
                                     args = args)
    
    #plot
    if args.plot_train:
        ut.plot_train(training_stats, args.load_path)
        
    #prediction on test
    combine_predictions, combine_true_labels = ut.predictions(model = model, 
                                                              test_loader = test_loader, 
                                                              use_cuda = use_cuda, 
                                                              path = args.load_path)
    
    citation_df = dp.df.iloc[dp.test_idx,:]
    similarity_dict, max_leng = ut.create_smilarity_dict(citation_df = citation_df, 
                                                         combine_predictions = combine_predictions, 
                                                        save_path = args.load_path)
    print(max_leng)
    #metrics
    print('MRR:')
    print(Metrics(dp.citation, leng = max_leng).calculate_mrr(similarity_dict)) #mrr

    print('recall@1, recall@10:')
    print(Metrics(dp.citation, leng = max_leng).calculate_recall_at_k(similarity_dict, 1))
    print(Metrics(dp.citation, leng = max_leng).calculate_recall_at_k(similarity_dict, 10))

    print('precision@1, precision@10:')
    print(Metrics(dp.citation,leng = max_leng).calculate_precision_at_k(similarity_dict, 1))        
    print(Metrics(dp.citation,leng = max_leng).calculate_precision_at_k(similarity_dict, 10))

    print('MAP:')
    print(Metrics(dp.citation,leng = max_leng).calculate_MAP_at_k(similarity_dict))
    
    
    
    
if __name__ == '__main__':
    main()