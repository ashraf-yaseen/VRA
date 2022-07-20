import gensim
from gensim.test.utils import get_tmpfile
import multiprocessing
import os
import pickle


from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from string import punctuation
punc = set(punctuation)
import operator


def clean(str):
    temp = []
    for word in str.split():
        if word in stop_words or word in punc:
            continue
        temp.append(word)
    return temp


class Doc2Vec:

    def __init__(self, model_path):
        self.dimension = 300
        self.epochs = 100
        self.no_of_steps = 200
        self.top_threshold = 10#?
        self.model_path = model_path
        self.doc2vec_model = None
        self.check_existance()
        
    def check_existance(self):
        flag = False
        if os.path.exists(self.model_path + 'doc2vec.model'):
            flag = True 
            self.load_model()
        return flag
        
       
    def load_model(self):
        self.doc2vec_model = gensim.models.Doc2Vec.load(self.model_path + 'doc2vec.model')

    def training(self, sentenct_dict):
        #the sentences need to a list of tokens not just the string of words
        tagged_docs = []
        for i_d in sentenct_dict:
            tagged_docs.append(gensim.models.doc2vec.TaggedDocument(clean(sentenct_dict[i_d]), [i_d]))
        cores = int(multiprocessing.cpu_count())
        self.doc2vec_model = gensim.models.Doc2Vec(dm=1, vector_size=self.dimension, 
                                                   window=8, min_count=2, epochs=self.epochs, 
                                                   workers=cores)
        print('Building vocab on {} documents'.format(len(tagged_docs)))
        self.doc2vec_model.build_vocab(tagged_docs)
        print('Built Vocab on {} words'.format(len(self.doc2vec_model.wv.vocab)))
        print('Training Model')
        self.doc2vec_model.train(tagged_docs, total_examples=self.doc2vec_model.corpus_count,
                                 epochs=self.doc2vec_model.iter)
        print('Built Model')
        filename =  self.model_path+ 'doc2vec.model'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.doc2vec_model.save(filename)
        print('Saved Model ',filename)

    def train_new_vec(self, sentence): 
        #here the input should be a list of tokens not the string of words 
        tokens = clean(sentence)
        infered_vec = self.doc2vec_model.infer_vector(doc_words=tokens, alpha=0.1, min_alpha=0.01, 
                                                      steps=self.no_of_steps)

        return infered_vec

    def similar_vec(self, vector):
        #vector now!
        listoftuples = self.doc2vec_model.docvecs.most_similar(positive=[vector], topn=self.top_threshold)
        #with ids and their score
        sim_dict = dict(listoftuples)
        return sim_dict

    def get_vector(self, vector_id):
        return self.doc2vec_model[vector_id]
