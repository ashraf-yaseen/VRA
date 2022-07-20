from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import os
import pickle

class LSA:

    def __init__(self, model_path):
        self.vectorizer = None
        self.lsa = None
        self.lsa_dict = None
        self.model_paths = [model_path + 'vectorizer.pickle', model_path + 'lsa.pickle', model_path + 'lsa_dict.pickle']
        self.check_model_existance()
        self.ngram_range = (1,1)
        self.min_df = 2 
        self.sdv_size = 300  # for bigger size matrix, I am keeping the svd vec size to 300

    def check_model_existance(self):
        model_exist = True
        for file_name in self.model_paths:
            if not os.path.exists(file_name):
                model_exist = False
                break
        if model_exist:
            self.load_all_models()

    def load_all_models(self):
        print('Loading Existing LSA models')
        self.vectorizer = pickle.load(open(self.model_paths[0], 'rb'))
        self.lsa = pickle.load(open(self.model_paths[1], 'rb'))
        self.lsa_dict = pickle.load(open(self.model_paths[2], 'rb'))
        print('Completed loading')

    def test_LSA(self, geo_dict):
        geo_vecs = {}
        geo_ids = list(geo_dict.keys())
        geo_sents = list(geo_dict.values())
        tf_idf_vecs = self.vectorizer.transform(geo_sents)
        lsa_vecs = self.lsa.transform(tf_idf_vecs)
        for geo_id, lsa_vec in zip(geo_ids, lsa_vecs):
            geo_vecs[geo_id] = lsa_vec 
        return geo_vecs

    def training_LSA(self, text_dict_article):
        pmids = list(text_dict_article.keys())
        final_text = list(text_dict_article.values())
        print('Total abstracts and titles = ', len(final_text))
        print('Forming vectorizer')
        self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, min_df=self.min_df)
        tfidf_vec = self.vectorizer.fit_transform(final_text)
        print('tf-idf shape = ', tfidf_vec.shape)
        print('tf-idf calculation complete')
        print('Starting SVD calculation')
        svd = TruncatedSVD(self.sdv_size, random_state=1234)
        print('SVD calculation complete')
        print('starting lsa calculation')
        self.lsa = make_pipeline(svd, Normalizer(copy=False))
        lsa_vec = self.lsa.fit_transform(tfidf_vec)
        print('Lsa calculation complete')
        print('LSA vector size = {}'.format(lsa_vec.shape))
        explained_variance = svd.explained_variance_ratio_.sum()
        print('Explained variance of svd = {}%'.format(int(explained_variance * 100)))
        self.lsa_dict = {}
        for pmid, vec in zip(pmids, lsa_vec):
            self.lsa_dict[pmid] = vec
        print('dumping all the pickle files')
        #self.rnw_pickle.write_pickle_file(self.vectorizer, self.model_paths[0])
        pickle.dump(self.vectorizer, open(self.model_paths[0], 'wb'))
        pickle.dump(self.lsa, open(self.model_paths[1], 'wb'))
        pickle.dump(self.lsa_dict, open(self.model_paths[2], 'wb'))
        #self.rnw_pickle.write_pickle_file(self.lsa, self.model_paths[1])
        #self.rnw_pickle.write_pickle_file(self.lsa_dict, self.model_paths[2])
        print('Saved all files')
        return self.lsa_dict


def main():
    x = LSA('/resources/lsa/')
    x.training_LSA('')
    x.test_LSA('')


if __name__ == '__main__':
    main()

