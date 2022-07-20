from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os


class TFIDF:

    def __init__(self, model_path):
        self.ngram_range = (1, 1)
        self.min_df = 2
        self.model_paths = [model_path + 'vectorizer.pickle', model_path + 'tfidf_vecs.pickle', \
                            model_path + 'pmids.pickle']
        self.vectorizer = None
        self.tfidf_vecs = None
        self.pmids = None
        self.check_model_existance()

    def check_model_existance(self):
        if os.path.exists(self.model_paths[0]) and \
           os.path.exists(self.model_paths[1]) and \
           os.path.exists(self.model_paths[2]):
            self.load_all_models()
      

    def load_all_models(self):
        self.vectorizer = pickle.load(open(self.model_paths[0], 'rb'))
        self.tfidf_vecs = pickle.load(open(self.model_paths[1], 'rb'))
        self.pmids = pickle.load(open(self.model_paths[2], 'rb'))

    def test_TFIDF(self, geo_dict):
        geo_vecs = {} 
        geo_ids = list(geo_dict.keys())
        sentences = list(geo_dict.values())
        tfidf_vecs = self.vectorizer.transform(sentences)
        #print('Total length of total test vectors ', tfidf_vecs.shape)
        return geo_ids, tfidf_vecs
        '''for geo_id, tfidf_vec in zip(geo_ids, tfidf_vecs):
            geo_vecs[geo_id] = tfidf_vec
        return geo_vecs'''


    def train_TFIDF(self, text_dict_article):
        self.pmids = list(text_dict_article.keys())
        final_text = list(text_dict_article.values())
        print('Total abstracts and titles = ', len(final_text))
        print('Forming vectorizer')
        self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, min_df=self.min_df)
        tfidf_vec = self.vectorizer.fit_transform(final_text)
        print('tf-idf calculation complete')

        print('dumbping all the pickle files')
        pickle.dump(self.vectorizer, open(self.model_paths[0], 'wb'))
        pickle.dump(tfidf_vec, open(self.model_paths[1], 'wb'))
        pickle.dump(self.pmids, open(self.model_paths[2], 'wb'))
        print('Saved all files')
        return self.pmids, tfidf_vec
        #return self.tfidf_dict'''


def main():
    x = TFIDF('../resources/TFIDFModels/')
    x.train_TFIDF('')
    x.test_TFIDF('')


if __name__ == '__main__':
    main()

