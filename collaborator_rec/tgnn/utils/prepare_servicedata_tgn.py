"""
This acts a fore-runner0 before running the main file 
you need to do somethig like

part0:
from utils.prepare_data import yearly_authors
yrs = list(range(1980, 1990))
authfile = '../../DLrec/newdata/processed_pubs.pickle'
authors = yearly_authors(authfile = authfile, years = yrs, savepath = 'data/') 

part1: 
!python utils/preprocess_data.py --data collab
before running the main notebook or pyfile

for service: remember to add engineered mask data for train, validation and testing
also newly added: node features: to processing the node features using cumsum

"""
import torch
import pickle
import pandas as pd
from torch_geometric.data import Data
import networkx as nx
from scipy.sparse import coo_matrix
from itertools import combinations 
from collections import defaultdict

import math
import random
import os.path as osp
from itertools import chain
import sys
from ast import literal_eval
import argparse

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.csgraph import shortest_path
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torch.nn import ModuleList, Linear, Conv1d, MaxPool1d

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, global_sort_pool
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   train_test_split_edges, k_hop_subgraph,
                                   to_scipy_sparse_matrix)

class yearly_authors():
    def __init__(self, authfile, service_dict, savepath, \
                 l_name = 'wu', f_name = 'hulin', m_name = '', options = 'pubs', val_ratio = 0.2,
                 exclude_users = ''):  
        self.file = pickle.load(open(authfile, 'rb'))
        self.service_dict = service_dict # directly from the service.scrap_pubs()
        # year to extract the graph from
        # self.year = years
        self.l_name = l_name
        self.f_name = f_name
        self.m_name = m_name
        if self.m_name.strip() == '':
            self.name = self.f_name + ' '  + self.l_name 
        else:
            self.name = self.f_name + ' ' +  self.m_name +  ' ' + self.l_name 
        self.val_ratio = val_ratio
        self.data_name = 'collab'
        self.maxLen = 50000 # 35000 for 2010-2011, 50000 for actual service stage
        self.keepauthors = 4
        self.savepath = savepath
        self.edge_indx_coo = None
        self.hasNodeLabel = False
        self.df_thin = None
        self.exclude_users = exclude_users
        self.df_authors()
        self.graphize()       
        self.options = options
        self.feature_x(options = self.options)
        
    def df_authors(self, date_col = 'pubdate'):
        """
        sorting authors' publications by year,
        processing the author list into coupled pair
        processing the format into a list of 
        save mapping pickles & graphs
        """
        # merge with exisiting data
        self.file = dict(list(self.file.items()) + list(self.service_dict.items())) 
        self.file_sorted = {k: v for k, v in sorted(self.file.items(), key=lambda item: str(item[1][date_col]))}
        
        self.file_year = self.file_sorted
        # check the length
        if len(self.file_year) > self.maxLen:
            # get the last
            self.file_year = {k: self.file_year[k] for k in list(self.file_year)[-self.maxLen:]}
        with open(self.savepath + '_.pickle', 'wb') as handle:
            pickle.dump(self.file_year, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.df = pd.DataFrame.from_dict(self.file_year, orient='index')
        self.df['timestamp'] = self.df[date_col].str.split(pat='-', expand = True).iloc[:,0].copy()
        ## or we might be able to incorporate more precise time!"""
        self.df['timestamp2'] = self.df[date_col].str.split(pat='-', expand = True).iloc[:,1].copy().str.rjust(2, '0')
        self.df['timestamp2'] = self.df['timestamp2'].fillna('00')
        self.df['timestamp3'] = self.df[date_col].str.split(pat='-', expand = True).iloc[:,2].copy().str.rjust(2, '0')
        self.df['timestamp3'] = self.df['timestamp3'].fillna('00')
        self.df['timestamp'] = self.df['timestamp'].copy() + self.df['timestamp2'] + self.df['timestamp3']

        self.df['timestamp'] = self.df['timestamp'].astype(int)
        self.df.sort_values(by=['timestamp'], inplace = True, ignore_index=True)
        self.df_thin = self.df[['authors', 'pmid', 'mesh_terms', 'journal', 'affiliations', 'country', 'timestamp']]
        # mesh to pmid
        self.mesh_dict = dict(zip(self.df_thin['pmid'].tolist(), self.df_thin['mesh_terms'].tolist()))
        
        # combinations of tuples
        authortuple = []
        time = []
        pmid = []
        affis = []
        for idx, row in self.df_thin.iterrows():
            #temp = row['authors']
            #temp = list(combinations(row['authors'].rsplit(';'), 2))
            keep = row['authors'].rsplit(';')
            if len(keep) > self.keepauthors:
                keep = keep[:self.keepauthors -1]+ keep[-1:]
            temp = list(combinations(keep, 2))
            authortuple.extend(temp)
            rep = len(temp)
            time.extend([row['timestamp']]* rep)
            pmid.extend([row['pmid']]* rep)
            affis.extend([row['affiliations']]* rep)
        self.df_long = pd.DataFrame({'authors': authortuple, 'timestamp': time, 'pmid': pmid}) #, 'affiliations': affis})
        # we need the state_label and X features
        self.df_long[['author', 'coauthor']] = pd.DataFrame(self.df_long['authors'].tolist(), index= self.df_long.index)
        self.df_long[['author', 'coauthor']] = self.df_long[['author', 'coauthor']].apply(lambda x: x.str.strip())
        self.df_long['state_label'] = pd.Series([0]*self.df_long.shape[0])
        # self.df_long.to_csv(self.savepath + 'df_long.csv', index = False)
      
    
    def graphize(self):
        
        """
        processing the dictionary into what tgn needs
        get mapping
        get graph 

        """
        # create mapping
        people = list(set(self.df_long.author.unique().tolist() + self.df_long.coauthor.unique().tolist()))
        self.num_nodes = len(people)
  
        le = LabelEncoder()
        ids = le.fit_transform(people)
        self.mapping = dict(zip(le.classes_, range(len(le.classes_))))
        self.names = le.classes_
        with open(self.savepath + 'mapping.pickle', 'wb') as handle:
            pickle.dump(self.mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
       
        # mapping for author and coauthor
        def people2idx(key):
            return self.mapping[key]
        self.df_long['new_author'] = self.df_long['author'].apply(people2idx)
        self.df_long['new_coauthor'] = self.df_long['coauthor'].apply(people2idx)
        
        # for later recommendation to produce more human-friendly result showcase
        pubs1 = self.df_long.groupby('new_author')['pmid'].agg(lambda x: list(set(x))).reset_index(name='part1')
        dict1 = pd.Series(pubs1.part1.values, index = pubs1.new_author).to_dict()
        pubs2 = self.df_long.groupby('new_coauthor')['pmid'].agg(lambda x: list(set(x))).reset_index(name='part2')
        dict2 = pd.Series(pubs2.part2.values, index = pubs2.new_coauthor).to_dict()
        for k, v in dict2.items():
            if k in dict1:
                dict1[k].extend(v)
            else:
                dict1[k] = v
        self.authorID2Pubs = dict1
        # new dictionary with k as the author id, then a nested dict of 'pmid_ls' and 'name' as the value
        author_refs = {}
        for k, v in self.authorID2Pubs.items():
            get_name = self.names[int(k)]
            try:
                v.remove('999999')
            except:
                pass
            author_refs[k] = {'name':get_name, 'pmid_ls': v}
        with open(self.savepath+'author_refs.pickle', 'wb') as handle:
            pickle.dump(author_refs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # we need to add 'train, valid and test mask here as well: as the 'mask' column with value [0, 1, 2]
        n_train =  int(self.df_long.shape[0] * (1- self.val_ratio))
        n_val =  self.df_long.shape[0] - n_train
        self.df_long['mask'] = pd.Series([0]*n_train + [1]*n_val) 
        
        # deliberately create testing pairs:
        exist = [v['authors'] for i, v in self.service_dict.items()]
        exist2 = [x.split(';') for x in exist]
        exist_users = [item for sublist in exist2 for item in sublist]
        possible = list(set(people).difference(set(self.exclude_users), set(exist_users)))
        possible_ids = [self.mapping[e] for e in possible]

        ## double check this name, users's actual name in the publication
        serve_id = self.mapping[self.name] #make sure they appear as exactly they are in the pubMed, titled format
        new_author = [serve_id]* len(possible_ids)
        new_coauthor = possible_ids
        timestamp = [self.df_long['timestamp'].iloc[-1] + 1] * len(possible_ids)
        # pmid = ['999999']* len(possible_ids)
        state_label = [0] * len(possible_ids)
        mask = [2] * len(possible_ids)
        
        df_test = pd.DataFrame({'new_author': new_author, 'new_coauthor': new_coauthor,\
                               'timestamp': timestamp, 'state_label':state_label, 'mask': mask})
        # self.df_long = pd.concat([self.df_long, df_test], ignore_index = True)
        cols = ['new_author', 'new_coauthor', 'timestamp', 'state_label', 'mask'] #list(self.df_long.columns)[8:]
        self.df_long = self.df_long[cols]
        self.df_long = pd.concat([self.df_long, df_test], ignore_index = True)
        self.df_long.to_csv(self.savepath + 'collab.csv', index = False)

        
    def feature_x(self, options='mesh'):
        """
        get node features, if we want to use mesh terms or other content (pubs) to represnet the authors
        all using tfidf, max_features 1000 
        feat_x: a compressed sparse row matrix, aggregated based on pmid in self.df_long 
        """
        if options == 'mesh':
            # let's processing mesh terms
            # we need self.df_thin['mesh_terms'] for this, first aggregate then convert
            # aggregate the results at the author level 
            meshlist = []
            for item in self.df_long['pmid'].tolist():
                temp = self.mesh_dict[item]
                meshlist.append(temp)
            
            # do the corpus aggregation 
            mesh = meshlist
            meshlist = [x.split(';') for x in mesh]
            meshlist2 = []
            for z in meshlist:
                temp = [x[x.find(':')+1:] for x in z]
                meshlist2.append(temp)
            # this is for converting a list of list to a big list
            meshlist3 =  [" ".join(x) for x in meshlist2]
            corpus = meshlist3 
        elif options =='pubs':
            # crawl the pubmed articles' title infos first using the orignal authorfile
            # then set a feature representation 
            '''needs to be modified'''
            pubs_dict = {}
            # print(self.file_year)
            for k, v in self.authorID2Pubs.items():
                temp = [self.file_year.get(str(p))['title'] if p != '999999' else '' for p in v]
                # print(temp)
                temp = ' '.join(temp)
                pubs_dict[k] = temp                   
            dict_df = pd.DataFrame({'id': list(pubs_dict.keys()), 'title': list(pubs_dict.values())})
            # sort by id
            corpus_df = dict_df.sort_values(by='id')
            # print(corpus_df.head())
            corpus = corpus_df['title'].tolist()
        else:
            print ('author feature representation not implemented')
        
        vectorizer = TfidfVectorizer(max_features = 172) #control number of features here
        feat_x = vectorizer.fit_transform(corpus)
        outfile = '{}ml_{}_node.npy'.format(self.savepath, self.data_name)
        newrow = np.zeros((1, 172))
        A = np.vstack([newrow, feat_x.toarray()])
        np.save(outfile, A)
