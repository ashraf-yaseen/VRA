import multiprocessing
from gensim.models import word2vec
from gensim import corpora
import pickle
import numpy as np
import os
from gensim.matutils import softcossim
from gensim.test.utils import common_texts
from gensim.corpora import Dictionary
from gensim.models import Word2Vec, WordEmbeddingSimilarityIndex
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix


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


class Word2Vec:

    def __init__(self, model_path):
        self.dimension = 200
        self.top_threshold =10
        self.model_name = model_path + 'word2vec.model'
        self.bow_dict_path = model_path + 'bow_dict.model'
        self.dict_path = model_path + 'dict.model'
        self.sim_matrix_path = model_path + 'sim.model'
        self.vec_dict_path = model_path + 'vec_dict.model'
        self.word2vec_model = None
        self.dictionary = None
        self.bow_dict = None
        self.similarity_matrix = None
        self.check_existance()

    def check_existance(self):
        flag = False
        if os.path.exists(self.model_name) and \
           os.path.exists(self.bow_dict_path) and \
           os.path.exists(self.dict_path) and \
           os.path.exists(self.sim_matrix_path):
            flag = True 
            self.load_models()
        return flag
        

    def load_models(self):
        print('Loading all models in word2vec')
        self.word2vec_model = word2vec.Word2Vec.load(self.model_name)
        self.bow_dict = pickle.load(open(self.bow_dict_path, 'rb'))
        print(len(self.bow_dict), ' size of the bow dict')
        self.dictionary = pickle.load(open(self.dict_path, 'rb'))
        self.similarity_matrix = pickle.load(open(self.sim_matrix_path, 'rb'))
        self.docsim_index = SoftCosineSimilarity(list(self.bow_dict.values()), self.similarity_matrix, num_best=self.top_threshold)
        print('Loded all the models in word2vec')

    def training(self, sentence_dict):
        sentences = []
        for i_d in sentence_dict:
            sentences.append(clean(sentence_dict[i_d]))
        #print(sentences)
        #cores = multiprocessing.cpu_count()
        cores =4 #let's just try this instead 
        print('Building word2vec model on {} documents'.format(len(sentences)))
        self.word2vec_model = word2vec.Word2Vec(sentences, size=self.dimension, 
                                                   window=8, min_count=5, #epochs=self.epochs, 
                                                   workers=cores)
        print('Built Model')
        self.word2vec_model.save(self.model_name)
        print('Saved Word2Vec Model')
        self.dictionary = corpora.Dictionary(sentences)
        #corpus = [dictionary.doc2bow(document) for sentence in sentences]
        self.bow_dict = {}
        for i_d in sentence_dict:
            self.bow_dict[i_d] = self.dictionary.doc2bow(clean(sentence_dict[i_d]))
        pickle.dump(self.bow_dict, open(self.bow_dict_path, 'wb'))
        print('Saved Bag of word dictionary')
        pickle.dump(self.dictionary, open(self.dict_path, 'wb'))
        print('Saved Dictionary')
        self.similarity_matrix = self.word2vec_model.wv.similarity_matrix(self.dictionary)
        pickle.dump(self.similarity_matrix, open(self.sim_matrix_path, 'wb'))
        print('Saved similarity matrix')

    def create_new_vec(self, tokens):
        if not self.word2vec_model:
            self.load_models()
        final_vec = np.zeros(self.dimension)
        hits = 0
        for word in tokens:
            if word not in self.word2vec_model:
                continue
            final_vec += self.word2vec_model[word]
            hits += 1
        if not hits == 0:
            final_vec /= hits
        return final_vec

    def similar_vec_soft_cosine_two(self, tokens1, tokens2):
        if not self.word2vec_model:
            self.load_models()
        new_sent1 = self.dictionary.doc2bow(tokens1)
        new_sent2 = self.dictionary.doc2bow(tokens2)
        return softcossim(new_sent1, new_sent2, self.similarity_matrix)

    def similar_vec_soft_cosine(self, words):
        if not self.word2vec_model:
            self.load_models()
        tokens = clean(words)
        get = self.dictionary.doc2bow(tokens)
        #termsim_index = WordEmbeddingSimilarityIndex(self.word2vec_model.wv)
        sims = self.docsim_index[get] 
        sims_dict = dict(sims)
        new_keys  = list(self.bow_dict.keys())
        new_sims_dict = {new_keys[key]: sims_dict[key] for key in list(sims_dict.keys())}
        return new_sims_dict
        #return sim_dict

    def create_vectors(self, sentence_dict):
        if not self.word2vec_model:
            self.load_models()
        vec_dict = {}
        for i_d in sentence_dict:
            vec_dict[i_d] = self.create_new_vec(clean(sentence_dict[i_d]))
        return vec_dict

    def create_vectors_from_publication(self, sentence_dict):
        if not os.path.exists(self.vec_dict_path):
            vec_dict = self.create_vectors(sentence_dict)
            pickle.dump(vec_dict, open(self.vec_dict_path, 'wb'))
            return vec_dict
        else:
            return pickle.load(open(self.vec_dict_path, 'rb'))
        

def main():
    path = '../resources/word2vec_models/Plain/'
    x = Word2Vec(path)
    words = 'barely las for vesion test newr'
    sent = {}
    sent['100'] = words
    sent['101'] = words
    x.training(sent)


if __name__ == "__main__":
    main()
