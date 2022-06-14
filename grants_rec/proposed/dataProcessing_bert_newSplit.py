#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:13:27 2020
@author: Ginnyzhu
for RFA and publication data processing:
1.converting pairs to datafram,
2.preparing sentences, labels lists
3.converting them into dataloader(of tensors)
"""

#general
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#torch and bert
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from  transformers import BertTokenizer

#self-defined
from utils_bert import get_corpus_and_dict, get_corpus_and_dict2

class RFADataProcess:
    def __init__(self, path = 'newdata/', load_pretrained = False, load_path = 'model_save/', newSplit = True):
        # Reading training and testing datasets
        # and details for pub & rfas
        self.path = path
        self.load_pretrained = load_pretrained
        self.load_path = load_path
        self.truth =  pickle.load(open(self.path + 'true_pairs_1_2.pickle', 'rb'))
        self.random_false = pickle.load(open(self.path + 'random_false_pairs_1_2.pickle', 'rb'))
        self.pubs = pickle.load(open(self.path +'processed_pubs.pickle', 'rb'))
        self.rfas = pd.read_csv(self.path + 'processed_nih_grants_only.csv')
        self.df = None
        if not self.load_pretrained:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') #do_lower_case = True:default
        else:
            self.tokenizer = BertTokenizer.from_pretrained(self.load_path)
        self.tokenizer.padding_side = 'left'
        self.batch_size = 8
        self.sep_len = 256
        self.ab_len = 256 #if too big, do the 384 or 256
        self.newsplit = newSplit
        self.citation = pickle.load(open(self.path + 'citation_data.dict', 'rb'))
    
    def dataframize_(self, col_names = ['pmid', 'rfaid']):
        if os.path.exists(self.path+'pairs_mixed.csv'):
            self.df = pd.read_csv(self.path+'pairs_mixed.csv')
            print('mixed pairs dataframe loaded')
            output = (self.df,)
            if self.newsplit:
                pmids = self.df.pmid.unique()
                train, test = train_test_split(pmids, random_state=1234, test_size=0.3)
                test, valid  = train_test_split(test, random_state=1234, test_size=0.3)
                self.train_idx = np.where(self.df['pmid'].isin(train))[0]
                self.valid_idx = np.where(self.df['pmid'].isin(valid))[0]
                self.test_idx = np.where(self.df['pmid'].isin(test))[0]
                with open(self.path  + 'train_idx.ls', 'wb') as handle:
                    pickle.dump(self.train_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(self.path  + 'valdi_idx.ls', 'wb') as handle:
                    pickle.dump(self.valid_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(self.path  + 'test_idx.ls', 'wb') as handle:
                    pickle.dump(self.test_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
                output = output + (self.train_idx,) + (self.valid_idx,) + (self.test_idx,)
        else:
            #first to dataframes, the pairs will have 'matcgh columns to indicate whether match or not
            df_false = pd.DataFrame(self.random_false, columns = col_names) 
            df_false['match'] = 0
            df_true = pd.DataFrame(self.truth, columns =col_names) 
            df_true['match'] = 1
            df= pd.concat([df_true, df_false], ignore_index=True)
            print('newly created dataframe here:\n')
            print(df.head())
            print(df.shape) #3528282 

            #let's check if the rfas have all the pairs info
            rfa_df_ls = set(df[col_names[-1]].tolist()) #
            rfa_ls = set(self.rfas['funding_opportunity_number'].tolist()) 
            #make sure the pairs rfas are only the subset
            print('checking whether all the pairs have information in rfa file:\n' )
            print(rfa_df_ls.issubset(rfa_ls)) # 
            common_ls = rfa_df_ls.intersection(rfa_ls)
            print(len(common_ls)) #

            new_df = df[df[col_names[-1]].isin(common_ls)]
            df = new_df.copy()
            #mix data all together
            df = df.sample(frac=1).reset_index(drop=True)
            print('final screened total pairs and shape:\n')
            #print(df.head())
            print(df.shape)
            print('true pairs total:\n')
            print(df['match'].sum()) #
            df.to_csv(self.path+'pairs_mixed.csv', index = False)
            self.df = df
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
        if os.path.exists(self.path + 'pub_corpus.pickle') and \
           os.path.exists(self.path + 'pub_corpus_dict.pickle') and \
           os.path.exists(self.path + 'rfa_corpus.pickle') and \
           os.path.exists(self.path + 'rfa_corpus_dict.pickle'):
            
            pub_corpus = pickle.load(open(self.path + 'pub_corpus.pickle', 'rb'))
            pub_corpus_dict = pickle.load(open(self.path + 'pub_corpus_dict.pickle', 'rb'))
            rfa_corpus = pickle.load(open(self.path + 'rfa_corpus.pickle', 'rb'))
            rfa_corpus_dict = pickle.load(open(self.path + 'rfa_corpus_dict.pickle', 'rb'))
            print('corpus loaded')
        else:
            pub_corpus, pub_corpus_dict = get_corpus_and_dict(df= self.df,id_col = 'pmid',
                                                              filepickle= self.pubs, field1= 'ptitle', field2 ='pabstract',
                                                              out_addr = self.path, name1 ='pub_corpus', name2 ='pub_corpus_dict')
            rfa_corpus, rfa_corpus_dict = get_corpus_and_dict2(df= self.df, id_col = 'rfaid',
                                                                  filecsv = self.rfas, file_id_col =  'funding_opportunity_number',
                                                                  field1= 'processed_funding_opportunity_title', 
                                                                  field2 ='processed_description',
                                                                  out_addr = self.path, name1 ='rfa_corpus', name2 ='rfa_corpus_dict')

        targets = self.df['match'].tolist()
        if not self.newsplit:
            train_pub_corpus, valid_pub_corpus, train_rfa_corpus, valid_rfa_corpus, \
            train_targets, valid_targets = train_test_split(pub_corpus, rfa_corpus, targets,
                                                                    random_state=1234, test_size=0.3)
            test_pub_corpus, valid_pub_corpus, test_rfa_corpus, valid_rfa_corpus, \
            test_targets, valid_targets = train_test_split(valid_pub_corpus, valid_rfa_corpus, valid_targets,
                                                                    random_state=1234, test_size=0.3)
        else:
            train_pub_corpus, train_rfa_corpus, train_targets = list(np.array(pub_corpus)[self.train_idx]), \
            list(np.array(rfa_corpus)[self.train_idx]), list(np.array(targets)[self.train_idx])
            valid_pub_corpus, valid_rfa_corpus, valid_targets = list(np.array(pub_corpus)[self.valid_idx]), \
            list(np.array(rfa_corpus)[self.valid_idx]), list(np.array(targets)[self.valid_idx]) 
            test_pub_corpus, test_rfa_corpus, test_targets = list(np.array(pub_corpus)[self.test_idx]), \
            list(np.array(rfa_corpus)[self.test_idx]), list(np.array(targets)[self.test_idx])
            
            
        train_target = torch.tensor(train_targets) #has to be: longtensor
        valid_target = torch.tensor(valid_targets) #dtype = torch.long)    
        test_target = torch.tensor(test_targets)# dtype = torch.float32)
        
        #now splits
        if strategy  == 'separate':
            train_pub, train_pub_mask, _ = self.tensorize_sep(train_pub_corpus)
            valid_pub, valid_pub_mask,_ = self.tensorize_sep(valid_pub_corpus)
            test_pub, test_pub_mask,_ = self.tensorize_sep(test_pub_corpus)
    
            train_rfa, train_rfa_mask, _= self.tensorize_sep(train_rfa_corpus)
            valid_rfa, valid_rfa_mask, _  = self.tensorize_sep(valid_rfa_corpus)
            test_rfa, test_rfa_mask, _  = self.tensorize_sep(test_rfa_corpus)  
            
            # Create an iterator of our data with torch DataLoader. 
            # unlike a for loop, with an iterator the entire dataset does not need to be loaded into memory
            train_data = TensorDataset(train_pub, train_pub_mask, train_rfa, train_rfa_mask, train_target)
            valid_data = TensorDataset(valid_pub, valid_pub_mask, valid_rfa, valid_rfa_mask, valid_target)
            test_data = TensorDataset(test_pub, test_pub_mask, test_rfa, test_rfa_mask, test_target)
                            
        else:
            #both setences input together
            train_pr, train_mask, train_type_id = self.tensorize_AB(train_pub_corpus, train_rfa_corpus)
            valid_pr, valid_mask, valid_type_id = self.tensorize_AB(valid_pub_corpus, valid_rfa_corpus)
            test_pr, test_mask, test_type_id = self.tensorize_AB(test_pub_corpus, test_rfa_corpus)
            

            train_data = TensorDataset(train_pr, train_mask, train_type_id, train_target)
            valid_data = TensorDataset(valid_pr, valid_mask, valid_type_id, valid_target)
            test_data = TensorDataset(test_pr, test_mask, test_type_id, test_target)            


        #loader
        train_sampler = RandomSampler(train_data) #to sequential instead, cuz it won't matter, and also easier for tfidf_weights
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size= self.batch_size, drop_last= True)
        valid_sampler = RandomSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=self.batch_size, drop_last= True)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.batch_size, drop_last = True)

        return train_dataloader, valid_dataloader, test_dataloader, train_pr#train_pr for all encoded training corpus