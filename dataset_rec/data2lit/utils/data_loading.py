import pickle, os
from preprocessing import PreProcessing
from read_geo_data import GetGeoData
from read_article_data import GetArticleData
from configuraiton import Rec_configuration

class DataLoading:

    def __init__(self):
        self.rec_conf = Rec_configuration()
        #self.pp = PreProcessing()
        #Geo data processing
        #self.geo_data = None
        self.geo_title, self.geo_summary = None, None
        self.citation_data = None
        #Article data processing 
        self.article_title, self.article_abstract = None, None
        self.initial_checking_loading_processing()

    def get_all_details(self):
        return self.article_title, self.article_abstract, \
               self.geo_title, self.geo_summary, self.citation_data
               
    def initial_checking_loading_processing(self, word_pr= False ):
        print('Loading all geo related preprocessed datasets')
        self.geo_title = pickle.load(open(self.rec_conf.file_address[2], 'rb'))
        self.geo_summary = pickle.load(open(self.rec_conf.file_address[3], 'rb'))
        self.citation_data = pickle.load(open(self.rec_conf.file_address[4], 'rb'))
        print('Loaded all geo related processed datasets')
        
        if word_pr:
            print('Preprocessing geo titles')
            self.geo_title = self.pp.preprocess_all(self.geo_title)
            pickle.dump(self.geo_title, open(self.rec_conf.file_address[2], 'wb'), protocol=4)
            print('Processed and save geo titles')
    
            print('Preprocessing geo summaries')
            self.geo_summary = self.pp.preprocess_all(self.geo_summary)
            pickle.dump(self.geo_summary, open(self.rec_conf.file_address[3], 'wb'), protocol=4)
            print('Processed and save geo summeries')

        
        print('Loading all title and abstract')
        self.article_title = pickle.load(open(self.rec_conf.file_address[0], 'rb'))
        self.article_abstract = pickle.load(open(self.rec_conf.file_address[1], 'rb'))
        print('Loaded all title and abstract')
        
        
        if word_pr:
            print('Preprocessing article titles')
            self.article_title = self.pp.preprocess_all(self.article_title)
            pickle.dump(self.article_title, open(self.rec_conf.file_address[0], 'wb'), protocol=4)
            print('Processed and save article titles')
    
            print('Preprocessing article abstracts')
            self.article_abstract = self.pp.preprocess_all(self.article_abstract)
            pickle.dump(self.article_abstract, open(self.rec_conf.file_address[1], 'wb'), protocol=4)
            print('Processed and save article abstracts')
            

def main():
    x = DataLoading() 


if __name__=='__main__':
    main()
