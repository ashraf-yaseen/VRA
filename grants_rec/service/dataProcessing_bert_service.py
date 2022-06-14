#let's do the predictions
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri June 12 15:13:27 2020
@author: Ginnyzhu
for RFA and publication data processing:
1.converting pairs to datafram,
2.preparing sentences, labels lists
3.converting them into dataloader(of tensors)
"""

#general
import os
import re
import fnmatch
import datetime 
import itertools
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#torch and bert
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from  transformers import BertTokenizer

#self-defined
from utils_bert_service import get_corpus_and_dict, get_corpus_and_dict2
from CVProcessing import CVProcessor as CVP


def findfiles(which, where='.'):
    '''Returns list of filenames from `where` path matched by 'which'
       shell pattern. Matching is case-insensitive.'''
    
    rule = re.compile(fnmatch.translate(which), re.IGNORECASE)
    return [name for name in os.listdir(where) if rule.match(name)]



class RFADataProcessForPred:
    def __init__(self, logger, path1 = '../newdata/', path2 = '../../part2_CollaRec/service/',
                 load_pretrained = False, load_path = '../model_uq/',
                 f_name = 'Hulin', m_name = '', l_name = 'Wu'):
        # Reading training and testing datasets
        # and details for pub & rfas
        self.path = path1
        self.path2 = path2
        self.f_name = f_name
        self.m_name = m_name
        self.l_name = l_name 
        self.logger = logger
        if self.m_name.strip() =='':
            self.name = self.f_name + '_' + self.l_name 
        else:
            self.name = self.f_name + '_' + self.m_name + '_' + self.l_name 
        self.outpath = self.f_name.lower() + self.l_name.lower() + '/'
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)
        self.load_pretrained = load_pretrained
        self.load_path = load_path
        self.rfas = pd.read_csv(self.path + 'processed_nih_grants_only.csv')
        self.rfa_ls = pickle.load(open(self.path  + 'rfa_ls.ls', 'rb'))
        self.df = None
        if not self.load_pretrained:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') #do_lower_case = True:default
        else:
            self.tokenizer = BertTokenizer.from_pretrained(self.load_path)
        self.tokenizer.padding_side = 'left'
        self.batch_size = 8
        self.sep_len = 256
        self.ab_len = 256 #if too big, do the 384 or 256
        self.process_pubs()
    
    def process_pubs(self):
        if not (os.path.exists(self.outpath  + 'pubDetails') and \
         os.path.exists(self.outpath  + 'pmids')):  
            # load pdf and produce pubdetails and pmids
            cv = findfiles(self.l_name + '*.pdf', where= self.path2)[0] #get the title form
            cv = self.path2 + cv
            """
            last part to modify
            """
            final_data = CVP(self.logger).process(cv, self.outpath, self.f_name, self.l_name, self.m_name)
            final_data['researcher_name'] = self.name
            with open(self.outpath +  'outwname.json', 'w') as f:
                json.dump(final_data, f, indent =4) 
        # else it's already there we dont need to do anything, just load 
        self.pubs = pickle.load(open(self.outpath  + 'pubDetails', 'rb')) 
        self.pmids = pickle.load(open(self.outpath  + 'pmids', 'rb'))        
    
    def dataframize_(self, col_names = ['pmid', 'rfaid']):

        if os.path.exists(self.outpath +'df.csv'):
            self.df = pd.read_csv(self.outpath +'df.csv')
            print('pairs dataframe loaded')
            # output = (self.df,)
        else:
            #create pairs and save it 
            pairs = list(itertools.product(self.pmids, self.rfa_ls))
            self.df = pd.DataFrame(pairs, columns =['pmid','rfaid'])  
            self.df.to_csv(self.outpath + 'df.csv', index = False)  
        output = (self.df,)
        return output

    
    def tensorize_sep(self, content_ls):
        '''
        tensorize each sentences separately
        '''
        encoded_dict = [self.tokenizer.encode_plus(piece, add_special_tokens=True, 
                                              max_length = self.sep_len, 
                                              #truncation = True, 
                                              #padding = True,
                                              #padding = 'longest',#
                                              pad_to_max_length = True, 
                                              return_attention_mask = True, return_token_type_ids = True, 
                                              return_tensors = 'pt') for piece in content_ls] 
        pr = [row['input_ids'] for row in encoded_dict]
        mask = [row['attention_mask'] for row in encoded_dict]
        type_id = [row['token_type_ids'] for row in encoded_dict]
        pr = torch.cat(pr, dim=0)
        mask = torch.cat(mask, dim = 0)
        type_id = torch.cat(type_id, dim = 0)
        return pr, mask, type_id
    
    
    def tensorize_AB(self, content_ls, content_ls2):
        '''
        tensorize sentences AB together 
        '''
        encoded_dict = [self.tokenizer.encode_plus(piece[0], piece[1], add_special_tokens=True, 
                                              max_length = self.ab_len,  #truncation = True, 
                                              #padding = True, 
                                              pad_to_max_length = True,
                                              #padding = 'longest',pad_to_max_length = True,
                                              return_attention_mask = True, return_token_type_ids = True,
                                              return_tensors = 'pt') for piece in zip(content_ls, content_ls2)]
                
        pr = [row['input_ids'] for row in encoded_dict]
        mask = [row['attention_mask'] for row in encoded_dict]
        type_id = [row['token_type_ids'] for row in encoded_dict]
        pr = torch.cat(pr, dim=0)
        mask = torch.cat(mask, dim = 0)
        type_id = torch.cat(type_id, dim = 0)
        return pr, mask, type_id


    def dataloaderize_(self, strategy = 'together'):
   
        # the authors recommend a batch size of 16 or 32
        if os.path.exists(self.outpath+  'pub_corpus.pickle') and \
           os.path.exists(self.outpath+  'pub_corpus_dict.pickle') and \
           os.path.exists(self.outpath+  'rfa_corpus.pickle') and \
           os.path.exists(self.outpath+  'rfa_corpus_dict.pickle'):
            
            pub_corpus = pickle.load(open( self.outpath+ 'pub_corpus.pickle', 'rb'))
            pub_corpus_dict = pickle.load(open(self.outpath+  'pub_corpus_dict.pickle', 'rb'))
            rfa_corpus = pickle.load(open(self.outpath+  'rfa_corpus.pickle', 'rb'))
            rfa_corpus_dict = pickle.load(open( self.outpath+ 'rfa_corpus_dict.pickle', 'rb'))
            print('corpus loaded')
        else:
            pub_corpus, pub_corpus_dict = get_corpus_and_dict(df= self.df,id_col = 'pmid',
                                                              filepickle= self.pubs, field1= 'ptitle', field2 ='pabstract',
                                                              out_addr = self.outpath, name1 = 'pubs_corpus',
                                                              name2 = 'pub_corpus_dict')
            rfa_corpus, rfa_corpus_dict = get_corpus_and_dict2(df= self.df, id_col = 'rfaid',
                                                                  filecsv = self.rfas, file_id_col =  'funding_opportunity_number',
                                                                  field1= 'processed_funding_opportunity_title', 
                                                                  field2 ='processed_description',
                                                                  out_addr = self.outpath, name1 = 'rfa_corpus', 
                                                                  name2 = 'rfa_corpus_dict')

        test_pub_corpus = pub_corpus 
        test_rfa_corpus = rfa_corpus
        
        #now splits
        if strategy  == 'separate':
            test_pub, test_pub_mask,_ = self.tensorize_sep(test_pub_corpus)
            test_rfa, test_rfa_mask, _  = self.tensorize_sep(test_rfa_corpus)            
            test_data = TensorDataset(test_pub, test_pub_mask, test_rfa, test_rfa_mask, test_target)
                            
        else:
            #both setences input together
            test_pr, test_mask, test_type_id = self.tensorize_AB(test_pub_corpus, test_rfa_corpus)
            test_data = TensorDataset(test_pr, test_mask, test_type_id)# test_target)            


        #loader 
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.batch_size, drop_last = True)

        return  test_dataloader, test_pr #train_dataloader, valid_dataloader, test_dataloader, train_pr#