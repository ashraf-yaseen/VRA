#!/usr/bin/env python
# -*- coding: utf-8 -*-

import string
from unidecode import unidecode
from nltk.corpus import stopwords
import nltk
from nltk import word_tokenize
import re


class PreProcessing:

    def __init__(self):
        self.stop_words = set(stopwords.words('english') + ['thus', 'other', 'level', 'using', 'i.e.',
                                'may', 'several', 'basic', 'other', 'ohters', "either", "trough", 'often', "'s", 
                                'also', 'into', 'e.g.'])
        self.lemmatization = False
        print('Lemmatization is false, if required turn on again')
        self.wnl = nltk.WordNetLemmatizer()
        self.punct = set(string.punctuation)
        self.punct.update("''")
        self.punct.update('``')
        self.punct.update('-/-')
        self.punct.update('±')

    def process_text(self, text):
        '''This will remove all junk characters, links, symbols after tokenizing them. Later it removes the stopwords.
        Lemmatization depends upon the self.lemmatization variable.
        Parameters:
                   text (str): text with stopwords, links, symbols.
        Returns:
                   clean text.
        '''
        text = unidecode(text)
        text = text.replace(u'\u00a0', ' ')
        # This is for links.
        all_links = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        for link in all_links:
            text = text.replace(link, '')
        text = text.lower()
        temp = []
        for token in word_tokenize(text):
            token = token.strip()
            if token == '' or token == '\n':
                continue
            if token.isdigit():
                continue
            if self.lemmatization:
                lemma = self.wnl.lemmatize(token)
            else:
                lemma = token
            if lemma in self.punct or lemma in self.stop_words:
                continue
            elif len(lemma) == 1:
                continue
            else:
                temp.append(lemma)
        return ' '.join(temp)
