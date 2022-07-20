from gensim.summarization.bm25 import BM25
import operator
import os
import pickle

class BM25Class:

    def __init__(self, model_path):
        #self.rnw_pickle = RNWPickle()
        #self.rnw = ReadWrite()
        self.top_threshold = 10
        self.bm_25_object = None
        self.average_idf = None 
        self.pmids = None
        self.model_path = model_path
        self.check_model_existance()

    def check_model_existance(self):
        if os.path.isfile(self.model_path + 'bm25') and \
           os.path.isfile(self.model_path + 'avg_idf') and \
            os.path.isfile(self.model_path + 'pmids'):
            self.load_all_models()

    def load_all_models(self):
        print('Loading Existing models')
        self.bm_25_object = pickle.load(open(self.model_path + 'bm25', 'rb'))
        self.average_idf = pickle.load(open(self.model_path + 'avg_idf', 'rb'))
        self.pmids = pickle.load(open(self.model_path + 'pmids', 'rb'))
        print('Completed loading') 
        
    def create_model(self, text_dict_article):
        corpus = []
        self.pmids = []
        for pmid in text_dict_article:
            self.pmids.append(pmid)
            corpus.append(text_dict_article[pmid])
        print('Creating new model for BM25')
        self.bm_25_object = BM25(corpus)
        self.average_idf = sum(float(val) for val in self.bm_25_object.idf.values()) / len(self.bm_25_object.idf)
        print('Created model for BM25')
        filename =  self.model_path + 'bm25'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        pickle.dump(self.bm_25_object, open(self.model_path + 'bm25', 'wb'), protocol=4)
        filename =  self.model_path + 'avg_idf'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        pickle.dump(self.average_idf, open(self.model_path + 'avg_idf', 'wb'), protocol=4)
        filename =  self.model_path + 'pmids'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        pickle.dump(self.pmids, open( self.model_path + 'pmids', 'wb'), protocol=4)
        print('Saved all models for BM25')

    
    def get_score(self, geo_details):
        scores = self.bm_25_object.get_scores(geo_details)
        final_dict = dict(zip(self.pmids, scores))
        sorted_dict = sorted(final_dict.items(), key=operator.itemgetter(1), reverse=True)
        return dict(sorted_dict[:self.top_threshold])

    

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from string import punctuation
punc = set(punctuation)


def clean(str):
    temp = []
    for word in str.split():
        if word in stop_words or word in punc:
            continue
        temp.append(word)
    return temp

