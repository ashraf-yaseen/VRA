import sys
import string
from unidecode import unidecode
from nltk.corpus import stopwords
import nltk
from nltk import word_tokenize
from multiprocessing import Pool
import re
import codecs
from Rec_configuraiton import Rec_configuration


class PreProcessing:

    def __init__(self):
        self.conf = Rec_configuration()
        self.lemmatization = False
        print('Lemmatization is false, if required turn on again')
        self.wnl = nltk.WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.punct = set(string.punctuation)
        self.punct.update("''")
        self.punct.update('``')
        self.punct.update('-/-')
        self.punct.update('±')
        self.dict_normalize = self.read_any_dict(self.conf.short_forms)
        self.remove_items = ['background and objective', 'background and purpose', 'background and aims', 'background/objective', 'context and objective', 'outcome measures', 'discussion and conclusions', 'background', 'objective', 'aims', 'aim', 'methods', 'method', 'keywords', 'participants', 'introduction', 'discussion', 'setting', 'results', 'conclusions:', 'conclusion', 'conclusions']
        self.text_dict = {}

    def read_any_dict(self, file_name):
        temp_dict = {}
        with codecs.open(file_name, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                a = line.strip().split('\t')
                temp_dict[a[0]] = a[1]
        return temp_dict


    def preprocess_all(self, text_dict):
        self.text_dict = text_dict
        print(len(text_dict))
        i_ds = list(self.text_dict.keys())
        output = []
        pool = Pool()
        n_processes = pool._processes
        output = list(pool.map(self.preprocess, i_ds))
        processed_dict = {}
        for result in output:
            a = result.split('\t')
            processed_dict[a[0].strip()] = a[1].strip()
        return processed_dict

    def preprocess(self, i_d):
        #text = self.text_dict[i_d]
        text = 'Essential role of MALT1 protease activity in activated B cell-like diffuse large B-cell lymphoma'
        text = unidecode(text)
        #print(i_d, text)
        #TODO: turn on the normaliztion of text
        #text = self.normalize_text(text)
        all_links = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        for link in all_links:
            text = text.replace(link, '')
        text = text.lower()
        for item in self.remove_items:
            if item in text:
                text = text.replace(item, ' ')
        temp = []
        for token in word_tokenize(text):
            if token.startswith('modencode_submission'):
                continue
            if self.lemmatization == True:
                lemma = self.wnl.lemmatize(token)
            else:
                lemma = token
            if lemma in self.punct or lemma in self.stop_words:
                continue
            elif len(lemma) == 1:
                continue
            else:
                temp.append(lemma)
        text_preprocess = ' '.join(temp)
        #print(i_d, text_preprocess)
        return i_d + '\t' + text_preprocess

    def normalize_text(self, text):
        texts = []
        for word in word_tokenize(text):
            if word in self.dict_normalize:
                texts.append(self.dict_normalize[word])
            else:
                texts.append(word)
        return ' '.join(texts)

    def normalize_list(self, text_list):
        temp = []
        for item in text_list:
            if item in self.dict_normalize:
                temp.append(self.dict_normalize[item])
            else:
                temp.append(item)
        return temp

def main():
    pp = PreProcessing()
    print(pp.preprocess('111'))

if __name__=='__main__':
    main()
