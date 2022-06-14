"""
by ginny 01/31/2022
modified from Braja's processing codes
"""

import os
import pickle
import sys
from textblob import TextBlob
from datetime import datetime
import operator
from collections import Counter
import math
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel


# some local
import sys
sys.path.append('../../')
from preparation.clustering.dpmmgs import GSDPMM
from preparation.weight.year_weight import YearWeight
from preparation.pubmed_data_extraction.pubmed_query import PubCollection
from preparation.preprocessing.similar_words import SimilarWordsNormalize
from preparation.preprocessing.preprocessing import PreProcessing
# from RFOrecsys.embeddings.tf_idf import TFIDF


class CVProcessor:

    def __init__(self, logger):
        self.email = 'jzhu8@uth.edu'
        self.tool = 'GrantRecSys'
        self.word2vec = '../../RFOrecsys/resources/similarity/word2vec.model'
        self.thrshld_cluster_size = 2
        self.pub_collection = PubCollection(self.email, self.tool)
        self.weight = YearWeight()
        self.pp = PreProcessing()
        self.swn = SimilarWordsNormalize(self.word2vec)
        self.logger = logger

    def clustering_pub(self, publications):
        '''This clusters the publications into many clusters using GSDPMM
        Parameters:
                   publications (list of str): List of all processed publications.
                   Each publication contains title and abstract.
        Returns:
                   clusters_pubindex (dict of list): This contains the clusters of publication indexes
                   (indexes are from publications list)
        '''
        beta = float(10/math.sqrt(len(publications)))
        # beta = 2.5 # This is for Xing only
        self.logger.error('beta value is {}'.format(beta))
        max_cluster_size = len(publications)
        gsdmm = GSDPMM(publications, max_cluster_size, beta)
        gsdmm.initialize()
        gsdmm.gibbs_sampling()
        cluster_ids = gsdmm.z_c
        clusters_pubindex = {}
        for i, clust in enumerate(cluster_ids):
            if clust in clusters_pubindex:
                temp = clusters_pubindex[clust]
                temp.append(i)
                clusters_pubindex[clust] = temp
            else:
                clusters_pubindex[clust] = [i]
        # for cluster in clusters_pubindex:
        #    print(cluster)
        #    for pub_indx in clusters_pubindex[cluster]:
        #        print(publications[pub_indx])
        return clusters_pubindex

    def remove_small_cluster(self, clusters_pubindex):
        '''This removes the small clusters with less than minimum number
        (thrshld_cluster_size) of desired cluster size.
        Parameters:
                   clusters_pubindex (dict of list): This contains the clusters of publication indexes
                   (indexes are from publications list)
        Returns:
                   clusters_pubindex (list of lists): This contains the clusters of publication indexes after removing
                   cluster with minimum size (indexes are from publications list)
        '''
        new_clusters_pubindex = {}
        for cluster in clusters_pubindex:
            pub_indxs = clusters_pubindex[cluster]
            if len(pub_indxs) >= self.thrshld_cluster_size:
                new_clusters_pubindex[cluster] = pub_indxs
        return new_clusters_pubindex
    """
    def process_tf_idf(self, clustered_pubs, pub_years):
        self.model = TFIDF(self.conf.model_path_tfidf)
        rfo_ids = self.model.i_ds
        rfo_vecs = self.model.tfidf_vecs
        final_tfidf_vec = None
        flag = 0
        for text, year in zip(clustered_pubs, pub_years):
            vec = self.model.vectorizer.transform([text])
            if flag == 0:
                final_tfidf_vec = self.weight.calculate_weight(vec, year)
                flag = 1
            else:
                final_tfidf_vec += self.weight.calculate_weight(vec, year)
        final_tfidf_vec = final_tfidf_vec/len(pub_years)
        similarity = linear_kernel(final_tfidf_vec, rfo_vecs).flatten()
        similarity_dict = dict(zip(rfo_ids, similarity))
        return sorted(similarity_dict.items(), key=operator.itemgetter(1), reverse=True)
    """
    
    def format_op_cluster(self, clustered_pmids, pub_details):
        c_data = {}
        list_papers = []
        all_text = ''
        pub_text = []
        for pmid in clustered_pmids:
            dict_paper = {}
            pub_detail = pub_details[pmid]
            dict_paper['pmid'] = pmid
            dict_paper['link'] = pub_detail['link']
            dict_paper['title'] = pub_detail['title']
            dict_paper['abstract'] = pub_detail['abstract']
            dict_paper['year'] = pub_detail['year']
            all_text += pub_detail['title'] + ' '
            all_text += pub_detail['abstract'] + ' '
            temp = pub_detail['ptitle'] + pub_detail['pabstract']
            pub_text.extend(temp.split())
            list_papers.append(dict_paper)
        c_data['cluster_papers'] = list_papers
        c_data['keywords'] = self.extract_plain_keywords(all_text)
        # c_data['recommended_rfos'] = self.rfo_op(top_similar_rfos, pub_text)
        return c_data
    """
    def rfo_op(self, top_similar_rfos, pub_text):
        rank = 0
        list_rfos = []
        for i_d in top_similar_rfos:
            rfo_details = self.processed_rfos[i_d]
            dict_rfo = dict()
            dict_rfo['id'] = i_d
            dict_rfo['link'] = rfo_details['URL']
            dict_rfo['title'] = rfo_details['Funding Opportunity Title']
            dict_rfo['purpose'] = rfo_details['Funding Opportunity Purpose']
            dict_rfo['release_date'] = rfo_details['Release_Date']
            dict_rfo['expired_date'] = rfo_details['Expired_Date']
            dict_rfo['Activity_Code'] = rfo_details['Activity_Code']
            dict_rfo['Organization'] = rfo_details['Organization']
            dict_rfo['Clinical_Trials'] = rfo_details['Clinical_Trials']
            temp = rfo_details['ptitle'] + ' ' + rfo_details['pfop'] + ' ' + rfo_details['pfta']
            matched_text = set(pub_text).intersection(set(temp.split()))
            dict_rfo['matched_words'] = ' '.join(matched_text)
            dict_rfo['rank'] = rank
            rank += 1
            dict_rfo['score'] = top_similar_rfos[i_d]
            dict_rfo['category'] = 'NIH'
            list_rfos.append(dict_rfo)
        return list_rfos
        """

    def format_final_op(self, success, data, desc):
        final_dict = dict()
        final_dict['success'] = success
        final_dict['data'] = data
        final_dict['description'] = desc
        return final_dict

    def process_pubs_from_name(self, cv_path, outpath, f_name, l_name, m_name=''):
        '''This process the name in the PubMed, then filters the collected publications from
        PubMed using the text from CV.
        Parameters:
                   cv_path (str): path to cv
        '''
        pubs_output = outpath + 'pubDetails'
        self.logger.error('Collecting publications from PubMed for {}'.format(f_name+l_name))
        time_pubmed = datetime.now()
        pub_details = self.pub_collection.process_pub_collection(cv_path, f_name, l_name, m_name)
        time_elapsed = datetime.now() - time_pubmed
        self.logger.error('Time taken to collect {} publications from PubMed {}'.format(len(pub_details), time_elapsed))
        pickle.dump(pub_details, open(pubs_output, 'wb'))

        return pub_details

    def process(self, cv_path, outpath, f_name, l_name, m_name=''):
        self.logger.error('Processing publications for {} {} {}'.format(f_name, m_name, l_name))
        pub_details = self.process_pubs_from_name(cv_path, outpath, f_name, l_name, m_name)
        self.logger.error('Total number of aricle found for this author = {}'.format(len(pub_details)))

        if len(pub_details) == 0:
            self.logger.error('No article found this author')
            desc = 'We did not find any publication in PubMed using your name: {} {}' \
            'please use the text only box to find relevant rfos'.format(f_name, l_name)
            return self.format_final_op(False, {}, desc)
        original_publications = []
        publications = []
        pmids, years = [], []

        time_now = datetime.now()
        for pmid in pub_details:
            pub_detail = pub_details[pmid]
            words = pub_detail['ptitle'] + ' ' + pub_detail['pabstract']
            original_publications.append(words)
            publications.append(' '.join(self.swn.process_text(words)))
            # publications.append(words)
            years.append(pub_detail['year'])
            pmids.append(pmid)
        # print('Time taken to getting similar words = {}'.format(datetime.now()-time_now))       
        #with open(processed_path + f_name + l_name + '_pubs', 'wb') as f:
            #pickle.dump(original_publications, f)
        with open(outpath + 'pmids', 'wb') as f:
            pickle.dump(pmids, f)    
        with open(outpath + 'pubYrs', 'wb') as f:
            pickle.dump(years, f)    
        
        time_now = datetime.now()
        cluster_pubindex = self.clustering_pub(publications)
        with open(outpath + 'clusteredPubs', 'wb') as f:
            pickle.dump(cluster_pubindex, f)
        self.logger.error('Total clusters = {}'.format(len(cluster_pubindex)))
        self.logger.error('Time taken to clustering = {}'.format(datetime.now() - time_now))
        cluster_pubindex = self.remove_small_cluster(cluster_pubindex)
        self.logger.error('After removing small clusters, now number = {}'.format(len(cluster_pubindex)))
        all_rfos = []
        i_d = 0
        for cluster in cluster_pubindex:
            indxs = cluster_pubindex[cluster]
            # clustered_pmids = [pmids[indx] for indx in indxs]
            # ccp.append(clustered_pmids)
            clustered_pubs = [original_publications[indx] for indx in indxs]
            clustered_pmids = [pmids[indx] for indx in indxs]
            pub_years = [years[indx] for indx in indxs]
            #top_similar_items = None
            #top_similar_items = self.process_tf_idf(clustered_pubs, pub_years)
            #top_similar_rfos = dict(top_similar_items[:self.max_number_rfos])
            c_rfo = self.format_op_cluster(clustered_pmids, pub_details)#, top_similar_rfos)
            c_rfo['id'] = i_d
            i_d += 1
            all_rfos.append(c_rfo)
        desc = 'We found {} publications in pubmed'.format(len(pub_details))
        return self.format_final_op(True, all_rfos, desc)


    def extract_plain_keywords(self, text):
        '''This method extracts the keywords (mainly noun phrases) using TextBlob. 
        Parameters: 
                   text (str): plain text for identifying the keywords. 
                   final_list (list): list of keywords/noun phrases identified using textblob. 
        '''
        from nltk import word_tokenize
        final_list = []
        for token in word_tokenize(text):
            if len(token) == 1:
                continue
            if token.isdigit():
                continue
            elif token.isupper():
                final_list.append(token)
        blob = TextBlob(text)
        nps = blob.noun_phrases
        for word in nps:
            if word.upper() in final_list:
                final_list.append(word.upper())
            else:
                final_list.append(word)
        return dict(sorted(dict(Counter(final_list)).items(), key=operator.itemgetter(1), reverse=True))