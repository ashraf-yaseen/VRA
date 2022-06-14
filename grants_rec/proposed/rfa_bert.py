#general 
import os
import argparse
import pickle
import dill

#you cannot live without 
from tqdm import trange
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from termcolor import colored
from sklearn.feature_extraction.text import TfidfVectorizer
#from gensim import corpora

#pip install transformers
#pytorch related
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

#bert related
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, BertConfig, PretrainedConfig 
from transformers import AdamW


torch.autograd.set_detect_anomaly(True)

class GrantModel:
    def __init__(self, load_pretrained = False, load_path = 'model_save_v6/'):
        super(GrantModel, self).__init__()
        self.load_pretrained = load_pretrained
        self.load_path = load_path 
        #self.bert = BertModel(config)
        if not self.load_pretrained:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') #keep this the same
            self.model = BertForSequenceClassification.from_pretrained( "bert-base-uncased", # Use the 12-layer BERT
                                               num_labels = 2, # --2 for binary classification.  
                                               output_attentions = False, # Whether attentions weights, 
                                               output_hidden_states = True) # returns hidden-states. yes!

        else:
            
            self.tokenizer = BertTokenizer.from_pretrained(self.load_path)
            self.model = BertForSequenceClassification.from_pretrained(self.load_path, output_hidden_states = True)