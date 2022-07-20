#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 9 1:49:51 2020
@author: ginnyzhu
* data scrapping for publications, based on 'citation' information in geo dataset pickle file, save in batches(memory and connection issues) 
* merge all scrapped publications data together
* reversed the 'citation' key & values, and use the publication as the key instead
* described statistics of # of geos associated with publications, and took only 99.9% of the citation data where a publication has less than 13 dataset associated
* created pairs out of the true citation, and using batches of geos, create random pairs of pub--geo(cannot remove the true pairs from false pairs here, possibly memory issues?)
* converted the true pairs & random pairs into a dataframe, remove duplicates to see if satisfiying 1:1 ratio. If not, add more random pairs
"""
import glob 
import os
import pickle
import dill
from collections import defaultdict 
import itertools

#you cannot live without 
import pandas as pd
import numpy as np
import time
#import matplotlib.pyplot as plt
import random
from termcolor import colored

class dataPreProcess:
    def __init__(self,datapath = 'IIdata/immunespace.pickle', output = 'IIdata/'):
        self.datapath = datapath 
        self.data = pickle.load(open(self.datapath, 'rb'))
        self.output = output
        self.citation = None
        self.publs = None
        
    #first function to check   
    def load_citation_ls(self):
        if os.path.exists(self.output + 'imspace_citation_data.dict') and \
        os.path.exists(self.output + 'imspacePubs.ls') and \
        os.path.exists(self.output + 'imspaceIncitation.ls'):
            self.citation = pickle.load(open(self.output + 'citation_data.dict', 'rb'))
            self.publs = pickle.load(open(self.output + 'imspacePubs.ls', 'rb'))
            self.datals = pickle.load(open(self.output + 'imspaceIncitation.ls', 'rb'))
        else:
            print('files do not exist!! Please run get_geocitation_notnull() first!')
       
    def get_item(self, somedict):
        '''return a dictionary item'''
        for k,v in somedict.items():
            print (k,v)
            break
        
    def get_geocitation_notnull(self):
        '''original citation and save'''
        citation_dict = {}
        #counter = 0 
        for k,v in self.data.items():
            if (len(str(v['citations'])) !=  0) and (str(v['citations']).strip() != ''):
                citation_dict[k] = v['citations']
        self.citation = citation_dict 
        with open(self.output + 'imspace_citation_data.dict', 'wb') as handle:
            pickle.dump(self.citation, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.keys_fromcitation(name = 'imspace')
    
        #sanity check
        print('total length:', len(self.citation))
        print('sample data structure:')
        self.get_item(self.citation)
        return self.citation 
   
    def reverse_citation(self):
        '''reverse the citation dictionary and save'''
        new_citation_data = defaultdict(list)
        for k,v in self.citation.items():
            papls = str(v).strip().split()
            print(papls)
            for i in range(len(papls)): 
                if papls[i] in new_citation_data.keys():                      
                      new_citation_data[papls[i]].append(k) 
                else:
                    new_citation_data[papls[i]] = [k]
        self.citation = dict(new_citation_data)
        with open(self.output + 'imspace_citation_data.dict', 'wb') as handle:
            pickle.dump(self.citation, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.keys_fromcitation(self, name = 'imspacePubs')
            
        #sanity check
        print('total length:', len(self.citation))
        print('sample data structure:')
        self.get_item(self.citation) 
        return self.citation

    def cut_citation(self, perc = 0.999):
        lengths = [len(v) for v in self.citation.values()]
        num = np.quantile(lengths, [perc])[0]
        print("at %f, the length of citation is %d" % (perc, num))
        
        new_citation = {} 
        for k,v in self.citation.items():
            leng = len(v)
            if leng <= num: #we take 0.999 of all the data available
                new_citation[k] = v
        self.citation = new_citation
        with open(self.output + 'imspace_citation_data.dict', 'wb') as handle:
            pickle.dump(self.citation, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.keys_fromcitation(self, name='imspacePubsCut')
      
        #sanity check
        print('total length:', len(self.citation))
        print('sample data structure:')
        self.get_item(self.citation) 
        return self.citation
        
    def keys_fromcitation(self, name):
        '''write citation list out '''
        Incitation = list(self.citation.keys())
        with open(self.output + name + 'Incitation.ls', 'wb') as handle:
            pickle.dump(Incitation, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def create_pairs(self, ratio = 1, every = 100, error = 10):
        '''create actual pairs from citation and random false pairs
           ratio =  # of false/ # of true'''
        if os.path.exists(self.output + 'imspacePubsIncitation.ls') and \
           os.path.exists(self.output + 'imspaceIncitation.ls'):
            self.publs = pickle.load(open(self.output + 'imspacePubsIncitation.ls', 'rb'))
            self.geols = pickle.load(open(self.output + 'imspaceIncitation.ls', 'rb'))
        else:
            print('files do not exist!! Please run relevant function to extract data first!')
        
        true_pairs = []
        for k,v in self.citation.items():
            k_ls = [k]
            true_pairs.extend(list(itertools.product(k_ls,v)))
        with open(self.output + 'imspace_true_pairs.pkl', 'wb') as handle:
            pickle.dump(true_pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        #sanity check
        print('true pairs created')
        print('total number of true pairs: ', len(true_pairs))
        print('true pairs sample:')
        print(random.sample(true_pairs, 2))
        
        
        ##now batch create false pairs
        #batches through geoIncitation 96,457, with 100 per batch, keep some 103,108/100 in each batch, let's have a total of 103,108
        random.seed(1234)
        final = []
        chunks = (len(self.datals) - 1) // every + 1
        for i in range(chunks):
            batch = datals[i*every:(i+1)*every]
            #batch permutation 
            batch_perm = list(itertools.product(self.publs, batch))
            #print(batch_perm[:10])
            #break 
            choose_perm = random.sample(batch_perm, len(true_pairs)//every + error)
            choose_pairs = choose_perm
            final.extend(choose_pairs)
        #sanity check
        print('total number of random permutations: ',len(final))
        print('sample:')
        print(random.sample(final, 2))

        #some trials and errors:
        false_num = int(len(true_pairs) * ratio)
        total_num = len(true_pairs) + false_num 
        #experiment with thins number until satisfactory 
        final2 = random.sample(final, false_num + error)
        choose_pairs = list(set(final2) - set(true_pairs))
        if len(choose_pairs) >=  false_num:
            final3 = random.sample(choose_pairs, false_num)
        else:
            print('choose a larger error number!')
        #assertion
        col_names = ['a','b']
        df_false = pd.DataFrame(final3, columns = col_names) 
        df_true = pd.DataFrame(true_pairs, columns =col_names) 
        df= pd.concat([df_true, df_false], ignore_index=True)
        df.drop_duplicates(inplace = True)
        assert df.shape[0]== total_num
        
        with open(self.output+'imspace_random_false_pairs.pkl', 'wb') as handle:
            pickle.dump(final3, handle, protocol=pickle.HIGHEST_PROTOCOL)  
        print('random false pairs created and choosen')



def main():
    print('scrap publications, creating citations and pairs')
    x = dataPreProcess(datapath = 'IIdata/immunespace.pickle', output = 'IIdata/')
    #list of things we want to do here, either do everything or selectively 
    _ = x.get_geocitation_notnull()
    _ = x.reverse_citation()
    _ = x.cut_citation(perc = 0.999)
    
    '''
    #or, if we already have scrapped data, just no pairs yet,
    #we can do 
    x = dataPreProcess(geopath = 'GEOMetaDataCollection/geo_dataset.pickle', output = 'data/')
    x.load_citation_ls()
    #and then 
    '''
    x.create_pairs(ratio = 1, every = 100, error = 10)
    
  
   

if __name__ == '__main__':
    main()