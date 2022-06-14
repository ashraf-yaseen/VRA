#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues May 19 15:49:51 2020
@author: ginnyzhu
#we need to unify the length to maximum recommendation length

"""
import numpy as np

class Metrics:
    def __init__(self, citations):
        self.geo_citation_dict = citations


    def mean_reciprocal_rank(self, rs):
        rs = (np.asarray(r).nonzero()[0] for r in rs)
        return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

    def make_citation_list(self, geo_id, pmids):#, leng = 5):
        citations = self.geo_citation_dict.get(geo_id, [])
        result_list = []
        if not len(citations) == 0:
            present = list(set(citations).intersection(set(pmids))) 
            if len(present) == 0:
                result_list = [0] * len(pmids)
            else:
                
                index = pmids.index(present[0]) # why only the first one? cuz most of the geo only have 1 citation? 
                result_list = [0] * len(pmids)
                result_list[index] = 1
                '''
                #this should take considerations of the situations where geo has more than 1 citation (one has 10)
                result_list = [0] * len(pmids)
                for i in range(len(present)):
                    index = pmids.index(present[i])
                    result_list[index] =1  
                    '''
        return result_list

    def calculate_mrr(self, similarity_dict):
        final_result_list = []
        for geo_id in similarity_dict:
            selected_pmids_dict = similarity_dict[geo_id] #pmids with values for a particular geo_id
            result_list = self.make_citation_list(geo_id, list(selected_pmids_dict.keys()))
            if not len(result_list) == 0: 
                final_result_list.append(result_list)
        return self.mean_reciprocal_rank(final_result_list)

    def recall_at_k(self, rs, refLst, k):
        '''
        rs_atK: should be the array of indicators of at k recomendation fpr each pmid 
        rs_atK = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]]);
        3 pmids, each recommendations take top 3, 1 and 0 indicates whether relevant or not 
        reflst; should be the # of matches for each pmid 
        refLst = np.array([[3], [2], [1]]): 3 pmids, each has 3, 2, 1 relevant items with it.
        '''
        assert k >= 1
        b = np.zeros([len(rs),len(max(rs,key = lambda x: len(x)))]) #to count for situaions where leng of recommended varies
        for i,j in enumerate(rs):
            b[i][0:len(j)] = j
        r = b[:,:k]!=0 
        # r = np.asarray(rs)[:,:k] != 0
        #if r.size != k:
            #raise ValueError('Relevance score length < k')
        refLst = np.asarray(refLst)
        row_divide = np.true_divide(np.array(r).sum(axis = 1),refLst)
        recall = np.mean(row_divide)
        return recall 
       
    def calculate_recall_at_k(self, similarity_dict, k):
        ref_Lst = []
        final_result_list = []
        for geo_id in similarity_dict:
            selected_pmids_dict = similarity_dict[geo_id] #pmids with values for a particular geo_id
            result_list = self.make_citation_list(geo_id, list(selected_pmids_dict.keys()))
            citations_len = len(self.geo_citation_dict.get(geo_id, []))
            if not len(result_list) == 0: 
                final_result_list.append(result_list)
                ref_Lst.append(citations_len)
        return self.recall_at_k(final_result_list, ref_Lst, k)


    def precision_at_k(self, r, k):
        """Score is precision @ k
        Relevance is binary (nonzero is relevant).
        >>> r = [0, 0, 1]
        >>> precision_at_k(r, 1)
        0.0
        >>> precision_at_k(r, 2)
        0.0
        >>> precision_at_k(r, 3)
        0.33333333333333331
        >>> precision_at_k(r, 4)
        Traceback (most recent call last):
            File "<stdin>", line 1, in ?
        ValueError: Relevance score length < k
        Args:
            r: Relevance scores (list or numpy) in rank order
                (first element is the first item)
        Returns:
            Precision @ k
        Raises:
            ValueError: len(r) must be >= k
        """
        assert k >= 1
        r = np.asarray(r)[:k] != 0
        #if r.size != k:
            #raise ValueError('Relevance score length < k')
        return np.mean(r)

    def calculate_precision_at_k(self, similarity_dict, k):
        final_result_list = []
        out =[]
        for geo_id in similarity_dict:
            selected_pmids_dict = similarity_dict[geo_id] #pmids with values for a particular geo_id
            result_list = self.make_citation_list(geo_id, list(selected_pmids_dict.keys()))
            if not len(result_list) == 0: 
                final_result_list.append(result_list)
                out.append(self.precision_at_k(result_list, k))
        return np.mean(out)

    def average_precision(self, r, k=False):
        r = np.asarray(r) != 0    
        if not k:
            out = [self.precision_at_k(r, k+1) for k in range(r.size) if r[k]]
        else:
            out = [self.precision_at_k(r, k1+1) for k1 in range(k) if r[k1]]
        if not out:
            return 0.
        return np.mean(out)

    def mean_average_precision(self, rs, k=False):
        """Score is mean average precision
        Relevance is binary (nonzero is relevant).
        >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
        >>> mean_average_precision(rs)
        0.78333333333333333
        >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
        >>> mean_average_precision(rs)
        0.39166666666666666
        Args:
            rs: Iterator of relevance scores (list or numpy) in rank order
                (first element is the first item)
        Returns:
            Mean average precision
        """
        return np.mean([self.average_precision(r, k) for r in rs])
    
    def calculate_MAP_at_k(self, similarity_dict):
        final_result_list = []
        for geo_id in similarity_dict:
            selected_pmids_dict = similarity_dict[geo_id] #pmids with values for a particular geo_id
            result_list = self.make_citation_list(geo_id, list(selected_pmids_dict.keys()))
            if not len(result_list) == 0: 
                final_result_list.append(result_list)
        return self.mean_average_precision(final_result_list) 
