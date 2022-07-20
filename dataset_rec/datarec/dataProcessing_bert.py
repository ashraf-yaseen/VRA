#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#try this preparation 
"""
Created on Tue Mar 24 15:13:27 2020
@author: Ginnyzhu
for publication & geo data processing:
1.converting pairs to datafram,
2.split for training and nonsplit for prediction
3.preparing sentences, labels tensors
4.converting them into dataloader(of tensors)
"""

#general
import os
import pickle
import random 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#torch and bert
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from  transformers import BertTokenizer, PreTrainedTokenizer, AutoTokenizer 

#self-defined
from bert.utils_bert import get_corpus_and_dict4

class DataProcess:
    def __init__(self, path = 'data/', load_pretrained = False, load_path = 'model_save_v2/', 
                 split = True, newSplit = True):
        # Reading training and testing datasets
        # and details for pub & geos 
        self.path = path
        self.load_pretrained = load_pretrained
        self.load_path = load_path
        self.truth =  pickle.load(open(self.path + 'true_pairs.pkl', 'rb'))
        self.random_false = pickle.load(open(self.path + 'random_false_pairs.pkl', 'rb'))
        self.pubs = pickle.load(open(self.path +'pub_dataset.pickle', 'rb'))
        self.dataset = pickle.load(open(self.path +'geo_dataset.pickle', 'rb'))
        self.df = None
        if not self.load_pretrained:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') #do_lower_case = True:default
        else:
            self.tokenizer = BertTokenizer.from_pretrained(self.load_path)
        self.tokenizer.padding_side = 'left'
        self.batch_size = 8
        self.sep_len = 256
        self.ab_len = 256 #if too big, do the 384 or 256
        self.split = split
        self.newsplit = newSplit
        self.citation = pickle.load(open(self.path + 'citation_data.dict', 'rb'))
    
    def dataframize_(self, col_names = ['pmid', 'geoid']):
        if os.path.exists(self.path+'pairs_mixed.csv'):
            self.df = pd.read_csv(self.path+'pairs_mixed.csv')
            print('mixed pairs dataframe loaded')
            output = (self.df,)

        else:
            #first to dataframes, the pairs will have 'match' columns to indicate whether match or not
            df_false = pd.DataFrame(self.random_false, columns = col_names) 
            df_false['match'] = 0
            df_true = pd.DataFrame(self.truth, columns =col_names) 
            df_true['match'] = 1
            df= pd.concat([df_true, df_false], ignore_index=True)
            print('newly created dataframe here:\n')
            print(df.head())
            print(df.shape) #

            #let's check if all pubs & geo has info in the scrapped pickle file
            pub_df_ls = set(df[col_names[0]].tolist()) #
            pub_ls = set(list(self.pubs.keys())) 
            #make sure the pairs rfas are only the subset
            print('checking whether all the pairs have information in pubs file:\n' )
            print(pub_df_ls.issubset(pub_ls)) # 
            common_pub_ls = pub_df_ls.intersection(pub_ls)
            print(len(common_pub_ls)) #  
            geo_df_ls = set(df[col_names[1]].tolist()) #
            geo_ls = set(list(self.dataset.keys())) 
            #make sure the pairs rfas are only the subset
            print('checking whether all the pairs have information in geo file:\n' )
            print(geo_df_ls.issubset(geo_ls)) # 
            common_geo_ls = geo_df_ls.intersection(geo_ls)
            print(len(common_geo_ls)) 
            
            if (pub_df_ls.issubset(pub_ls)) and (geo_df_ls.issubset(geo_ls)):
                print('all data involved available')
            else:
                print('take subset of pairs whose info are available')
                df = df[df[col_names[0]].isin(common_pub_ls)]
                df = df[df[col_names[1]].isin(common_geo_ls)]
                
            df = df.copy()
            df = df.sample(frac=1).reset_index(drop=True)
            print('final screened total pairs and shape:\n')
            print(df.shape)
            print('true pairs total:\n')
            print(df['match'].sum()) 
            df.to_csv(self.path+'pairs_mixed.csv', index = False)
            self.df = df
            output = (self.df,)
            
        if self.split and self.newsplit:
            if os.path.exists(self.path+'train_idx.ls') and \
            os.path.exists(self.path+'valid_idx.ls') and \
            os.path.exists(self.path+'test_idx.ls'):
                self.train_idx = pickle.load(open(self.path+'train_idx.ls', 'rb'))
                self.valid_idx = pickle.load(open(self.path+'valid_idx.ls', 'rb'))
                self.test_idx = pickle.load(open(self.path+'test_idx.ls', 'rb'))
            else:
                pmids = self.df.pmid.unique()
                train, test = train_test_split(pmids, random_state=1234, test_size=0.3)
                test, valid  = train_test_split(test, random_state=1234, test_size=0.3)
                self.train_idx = np.where(self.df['pmid'].isin(train))[0]
                self.valid_idx = np.where(self.df['pmid'].isin(valid))[0]
                self.test_idx = np.where(self.df['pmid'].isin(test))[0]
                with open(self.path  + 'train_idx.ls', 'wb') as handle:
                    pickle.dump(self.train_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(self.path  + 'valid_idx.ls', 'wb') as handle:
                    pickle.dump(self.valid_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(self.path  + 'test_idx.ls', 'wb') as handle:
                    pickle.dump(self.test_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)

        elif not self.split:
            #else, no split, just all together for predictions
            self.train_idx = random.sample(list(self.df.index),self.batch_size)
            self.valid_idx = random.sample(list(self.df.index),self.batch_size)
            self.test_idx = list(self.df.index)
        else:
            print('not implemented')
        output = output + (self.train_idx,) + (self.valid_idx,) + (self.test_idx,)    
        return output
   
    """
    def tensorize_sep(self, content_ls):
        '''
        tensorize each sentences separately
        '''
        encoded_dict = [self.tokenizer.encode_plus(piece, add_special_tokens=True, 
                                              max_length = self.sep_len, pad_to_max_length = True, 
                                              return_attention_mask = True, return_token_type_ids = True, 
                                              return_tensors = 'pt') for piece in content_ls] 
        pr = [row['input_ids'] for row in encoded_dict]
        mask = [row['attention_mask'] for row in encoded_dict]
        type_id = [row['token_type_ids'] for row in encoded_dict]
        pr = torch.cat(pr, dim=0)
        mask = torch.cat(mask, dim = 0)
        type_id = torch.cat(type_id, dim = 0)
        return pr, mask, type_id"""
    
    def tensorize_AB(self, content_ls, content_ls2):
        '''
        tensorize sentences AB together 
        '''
        encoded_dict = [self.tokenizer.encode_plus(piece[0], piece[1], add_special_tokens=True, 
                                              max_length = self.ab_len, pad_to_max_length = True,
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
        if os.path.exists(self.path + 'pub_corpus.pickle') and \
           os.path.exists(self.path + 'pub_corpus_idls.pickle') and \
           os.path.exists(self.path + 'geo_corpus.pickle') and \
           os.path.exists(self.path + 'geo_corpus_idls.pickle'):
            
            pub_corpus = pickle.load(open(self.path + 'pub_corpus.pickle', 'rb'))  #119,355
            pub_corpus_ls = pickle.load(open(self.path + 'pub_corpus_idls.pickle', 'rb'))
            geo_corpus = pickle.load(open(self.path + 'geo_corpus.pickle', 'rb'))
            geo_corpus_ls = pickle.load(open(self.path + 'geo_corpus_idls.pickle', 'rb'))
            print('corpus loaded')
            
        else:
            pub_corpus, pub_corpus_dict = get_corpus_and_dict4(df= self.df,id_col = 'pmid',
                                                              filepickle= self.pubs,
                                                              textfield1 ='title', textfield2= 'abstract', 
                                                              out_addr = self.path, name1 ='pub_corpus', name2 ='pub_corpus_idls')
            geo_corpus, geo_corpus_dict = get_corpus_and_dict4(df = self.df, id_col = 'geoid', 
                                                               filepickle =self.dataset, 
                                                               textfield1 ='title', textfield2= 'summary', 
                                                               out_addr = self.path, name1 ='geo_corpus', name2 ='geo_corpus_idls')

        targets = self.df['match'].tolist()

        train_pub_corpus, train_geo_corpus, train_targets = list(np.array(pub_corpus)[self.train_idx]), \
        list(np.array(geo_corpus)[self.train_idx]), list(np.array(targets)[self.train_idx])
        valid_pub_corpus, valid_geo_corpus, valid_targets = list(np.array(pub_corpus)[self.valid_idx]), \
        list(np.array(geo_corpus)[self.valid_idx]), list(np.array(targets)[self.valid_idx]) 
        test_pub_corpus, test_geo_corpus, test_targets = list(np.array(pub_corpus)[self.test_idx]), \
        list(np.array(geo_corpus)[self.test_idx]), list(np.array(targets)[self.test_idx])   
            
        train_target = torch.tensor(train_targets) #has to be: longtensor
        valid_target = torch.tensor(valid_targets) #dtype = torch.long)    
        test_target = torch.tensor(test_targets)# dtype = torch.float32)

        #both setences input together
        train_pr, train_mask, train_type_id = self.tensorize_AB(train_pub_corpus, train_geo_corpus)
        valid_pr, valid_mask, valid_type_id = self.tensorize_AB(valid_pub_corpus, valid_geo_corpus)
        test_pr, test_mask, test_type_id = self.tensorize_AB(test_pub_corpus, test_geo_corpus)


        train_data = TensorDataset(train_pr, train_mask, train_type_id, train_target)
        valid_data = TensorDataset(valid_pr, valid_mask, valid_type_id, valid_target)
        test_data = TensorDataset(test_pr, test_mask, test_type_id, test_target)            


        #loader
        train_sampler = RandomSampler(train_data) 
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size= self.batch_size, drop_last= True)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=self.batch_size, drop_last= True)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.batch_size, drop_last = True)

        return train_dataloader, train_pr, valid_dataloader, test_dataloader