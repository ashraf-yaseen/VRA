import os.path
basedir = os.path.abspath(os.path.dirname(__file__))

class Rec_configuration:
    def __init__(self):
        """This defines the parameters used in master.py and all other files."""
        # Address listed for datasets. Change all addresses here.
        '''
        self.file_address = \
            [basedir + '/resources/article/title_ws.pickle',
             basedir + '/resources/article/abstract_ws.pickle',
             basedir + '/resources/dataset/title_ws.pickle',
             basedir + '/resources/dataset/summary_ws.pickle',
             basedir + '/resources/dataset/citations.pickle']
        
        '''#modified new
        self.file_address = \
            [basedir + '/resources/use4Maybe/new_article_title.pickle',
             basedir + '/resources/use4Maybe/new_article_abstract.pickle',
             basedir + '/resources/use4Maybe/new_geo_title.pickle',
             basedir + '/resources/use4Maybe/new_geo_summary.pickle',
             basedir + '/resources/use4Maybe/new_geo_haspmid.pickle']

        self.file_address2 = \
            [basedir + '/resources/bert_raw/new_article_title.pickle',
             basedir + '/resources/bert_raw/new_article_abstract.pickle',
             basedir + '/resources/bert_raw/new_geo_title.pickle',
             basedir + '/resources/bert_raw/new_geo_summary.pickle',
             basedir + '/resources/bert_raw/new_geo_haspmid.pickle']


      
        self.geo_file_address = basedir + '/resources/dataset/new_geo_dataset.pickle'
        # self.article_address = '/home/bgpatra/ftp.ncbi.nlm.nih.gov/processed_data/new_publication_data/'
        self.article_title_add = basedir +'/resources/final_pmids_title.pickle'
        self.article_abstract_add = basedir +'/resources/ArticleData/final_pmids_abstract.pickle'
        '''
        self.geo_file_address = basedir + '/resources/use4Maybe/new_article_title'
        # self.article_address = '/home/bgpatra/ftp.ncbi.nlm.nih.gov/processed_data/new_publication_data/'
        self.article_title_add = basedir + '/resources/use4Maybe/new_article_title'
        self.article_abstract_add = basedir + '/resources/use4Maybe/new_article_abtract'
        '''
        # Adddress for rfidf
        self.model_path_tfidf = basedir + '/resources/tf_idf_models/Plain/'
        self.result_address_tfidf = basedir + '/results/tf-idf_plain/'
        
        
        #model path for bm25
        self.model_path_bm25 = basedir + '/resources/BM25/'
        
        # Adddress for Bert embeddings
        self.model_path_bert = basedir + '/resources/berts/bert/'
        self.model_path_biobert = basedir + '/resources/berts/biobert/'
        self.model_path_scibert = basedir + '/resources/berts/scibert/'
        
        # Adddress for LDA
        self.result_address_lda = basedir + '/results/lda_plain/'
        
        # address for LSA
        self.model_path_lsa= basedir + '/resources/lsa/'
        self.result_address_lsa = basedir + '/results/lsa_plain/'
        
        # address for word2vec
        self.model_path_word2vec= basedir + '/resources/word2vec/'
        self.result_address_word2vec = basedir + '/results/word2vec_plain/'
        
        # address for doc2vec 
        self.model_path_doc2vec= basedir + '/resources/doc2vec/'
        self.result_address_doc2vec = basedir + '/results/doc2vec_plain/'
        
        # address for BM25
        self.result_address_bm25 = basedir + '/results/bm25_plain/'

        # address for USE
        self.model_path_use= "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.result_address_use = basedir + '/results/use_plain/'
        
        # address for infersent 
        self.model_path_inferSent1 = '../BertEmb/encoder/infersent1.pkl' 
        self.model_path_inferSent2 = '../BertEmb/encoder/infersent2.pkl' 
        # If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
        self.W2V_PATH1 = '../BertEmb/GloVe/glove.840B.300d.txt'
        self.W2V_PATH2 = '../BertEmb/fastText/crawl-300d-2M.vec'
        self.result_address_inferSent = basedir + '/results/inferSent_plain/'
        
    
        # address for elmo
        self.model_path_elmo = basedir + '/resources/elmo/'
        self.result_address_elmo = basedir + '/results/elmo_plain/'
        
        
        # address for roBerta
        self.result_address_roberta = basedir + '/results/roberta_plain/'
        
        # address for distilBert
        self.result_address_distilbert = basedir + '/results/distilbert_plain/'
        
        

        #result addresss 
        self.result_address_bert = basedir + '/results/bert_plain/'
        self.result_address_biobert = basedir + '/results/biobert_plain/'
        self.result_address_scibert = basedir + '/results/scibert_plain/'
        
        #result addresses for futher trained models
        self.result_address_bert_trained = basedir + '/results/bert_trained/'
        self.result_address_biobert_trained = basedir + '/results/biobert_trained/'
        self.result_address_scibert_trained = basedir + '/results/scibert_trained/'
        
        #ReRanking addresses
        self.w2v_path = basedir + '/resources/word2vec_models/Plain/'
        self.short_forms = basedir + '/resources/short_forms.txt'