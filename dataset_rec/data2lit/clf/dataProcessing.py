
"""
Created on Tue Mar 24 15:13:27 2020
@author: Ginnyzhu
for datset and publication data processing:
1.converting pairs to datafram,
2.preparing sentences, labels lists
3.converting them into dataloader(of tensors)
"""

#general
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

#torch and bert
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from  transformers import BertTokenizer

#self-defined
from utils import get_corpus_and_dict

class GEOLiTDataProcess:
    def __init__(self, path = 'resources/use4Maybe/'):
        # Reading training and testing datasets
        # and details for pub & rfas
        self.path = path
        self.df = pd.read_csv(self.path + 'pairs_mixed.csv') #total 152066, geo_id, pmid, match
        self.pubs_title = pickle.load(open(self.path +'new_article_title.pickle', 'rb'))
        self.pubs_abs = pickle.load(open(self.path +'new_article_abstract.pickle', 'rb'))
        self.geo_title = pickle.load(open(self.path +'new_geo_title.pickle', 'rb'))
        self.geo_sum = pickle.load(open(self.path +'new_geo_summary.pickle', 'rb'))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') #do_lower_case = True:default
        self.tokenizer.padding_side = 'left'
        self.batch_size = 8
        self.ab_len = 256 #if too big, do the 384 or 256
    
    def dataframize_(self, col_names = ['pmid', 'rfaid']):
        #first to dataframes, the pairs will have 'matcgh columns to indicate whether match or not

        return self.df
   
    
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
        return pr, mask, type_id
    
    
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
        #to lists and save
        pub_corpus, pub_corpus_dict = get_corpus_and_dict(df= self.df,id_col = 'pmid',
                                                          filepickle1= self.pubs_title, filepickle2=self.pubs_abs, 
                                                          out_addr = self.path, name1 ='pub_corpus', name2 ='pub_corpus_dict')
        geo_corpus, geo_corpus_dict = get_corpus_and_dict (df= self.df, id_col = 'geo_id',
                                                              filepickle1 = self.geo_title, filepickle2=self.geo_sum,
                                                              out_addr = self.path, name1 ='geo_corpus', name2 ='geo_corpus_dict')

        targets = self.df['match'].tolist()
       
        train_pub_corpus, valid_pub_corpus, train_geo_corpus, valid_geo_corpus, \
        train_targets, valid_targets = train_test_split(pub_corpus, geo_corpus, targets,
                                                                random_state=1234, test_size=0.3)
        test_pub_corpus, valid_pub_corpus, test_geo_corpus, valid_geo_corpus, \
        test_targets, valid_targets = train_test_split(valid_pub_corpus, valid_geo_corpus, valid_targets,
                                                                random_state=1234, test_size=0.3)
        
        train_target = torch.tensor(train_targets) #has to be: longtensor
        valid_target = torch.tensor(valid_targets) #dtype = torch.long)    
        test_target = torch.tensor(test_targets)# dtype = torch.float32)
        
        #now splits
        if strategy  == 'separate':
            train_pub, train_pub_mask, _ = self.tensorize_sep(train_pub_corpus)
            valid_pub, valid_pub_mask,_ = self.tensorize_sep(valid_pub_corpus)
            test_pub, test_pub_mask,_ = self.tensorize_sep(test_pub_corpus)
    
            train_rfa, train_geo_mask, _= self.tensorize_sep(train_geo_corpus)
            valid_rfa, valid_geo_mask, _  = self.tensorize_sep(valid_geo_corpus)
            test_rfa, test_geo_mask, _  = self.tensorize_sep(test_geo_corpus)  
            
            # Create an iterator of our data with torch DataLoader. 
            # unlike a for loop, with an iterator the entire dataset does not need to be loaded into memory
            train_data = TensorDataset(train_pub, train_pub_mask, train_geo, train_geo_mask, train_target)
            valid_data = TensorDataset(valid_pub, valid_pub_mask, valid_geo, valid_geo_mask, valid_target)
            test_data = TensorDataset(test_pub, test_pub_mask, test_geo, test_geo_mask, test_target)
                            
        else:
            #both setences input together
            train_pr, train_mask, train_type_id = self.tensorize_AB(train_pub_corpus, train_geo_corpus)
            valid_pr, valid_mask, valid_type_id = self.tensorize_AB(valid_pub_corpus, valid_geo_corpus)
            test_pr, test_mask, test_type_id = self.tensorize_AB(test_pub_corpus, test_geo_corpus)
            

            train_data = TensorDataset(train_pr, train_mask, train_type_id, train_target)
            valid_data = TensorDataset(valid_pr, valid_mask, valid_type_id, valid_target)
            test_data = TensorDataset(test_pr, test_mask, test_type_id, test_target)            


        #loader
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size= self.batch_size)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=self.batch_size)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(valid_data, sampler=test_sampler, batch_size=self.batch_size)

        return train_dataloader, valid_dataloader, test_dataloader