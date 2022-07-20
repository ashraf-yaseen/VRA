#!/usr/bin/env python
# -*- coding: utf-8 -*-

import string
from unidecode import unidecode
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
import re


class PreProcessing:

    def __init__(self):
        self.stop_words = set(stopwords.words('english') + ['data', 'model', 'method', 'level', 'models', 'dataset',
                                'using', 'novel', 'significant', 'important', 'paper', 'experiment', 'experimental',
                                'evaluation', 'i.e.', 'may', 'application', 'example', 'real', 'several', 'free',
                                'year', 'earlier', 'process', 'basic', 'application', 'account', 'equation',
                                'numerical', 'estimation', 'parameter', 'approach', 'high', 'initial', 'selection',
                                'unknown', 'variable', 'value', 'time', 'effects', 'step', 'properties', 'study',
                                'network', 'state', 'mixed', 'procedure', 'simulation', 'order', 'function', 'many',
                                'cost', 'article', 'analysis', 'independence', 'problem', 'solution', 'stage', 'sure',
                                'non', 'performance', 'accuracy', 'new', 'higher', 'algorithm', 'proposed', 'output',
                                'aids', 'technique', 'constant', 'case', 'size', 'low', 'large', 'first', 'errors',
                                'local',  'available', 'global', 'system', 'product', 'maximum', 'exact', 'sample',
                                'rate', 'common', 'course', 'challenge', 'couple', 'class', 'formula', 'different',
                                'rule', 'best', 'use', 'clear', 'areas', 'complex', 'observations', 'much',
                                'supplemental', 'materials', 'natural', 'usually', 'impact', 'better', 'uncertainty',
                                'input', 'due to', 'addition', 'methodology', 'modeling', 'products', 'forms',
                                'positive', 'negative', 'able', 'form', 'estimate', 'usefulness', 'approaches',
                                'measurement', 'superior', 'relations', 'edge', 'general', 'newly', 'wide', 'well',
                                'node', 'lower', 'degree', 'role', 'likely', 'group', 'shape', 'need', 'other',
                                'ohters', 'type',  "unique", "subject", "either", "simple", "peak", "trough",
                                'propose', 'often'])
        self.lemmatize_exception = {'aids'}
        self.lemmatization = True
        print('Lemmatization is {}, if required turn on/off again'.format(self.lemmatization))
        self.wnl = WordNetLemmatizer()
        self.punct_list = set(string.punctuation)
        self.punct_list.update("''")
        self.punct_list.update('``')
        self.punct_list.update('-/-')
        self.punct_list.update('±')

    def process_text(self, text):
        """This will remove all junk characters, links, symbols after tokenizing them. Later it removes the stopwords.
        Lemmatization depends upon the self.lemmatization variable.
        Parameters:
            text (str): text with stopwords, links, symbols.
        Returns:
            clean text.
        """
        text = unidecode(text)
        text = text.replace(u'\u00a0', ' ')
        #there is no link for now still keep it.
        all_links = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        for link in all_links:
            text = text.replace(link, '')
        text = text.lower()
        temp = []
        for word, pos in pos_tag(word_tokenize(text)):
            word = word.strip()
            if word in self.punct_list:
                continue
            elif word == '' or word == '\n' or word == '\t' or word == '\r':
                continue
            elif word.isdigit():
                continue
            elif word in self.lemmatize_exception:
                temp.append(word)
                continue
            elif word.startswith('@') or word.startswith('https://') or word.startswith('http://'):
                continue
            tag = None
            if pos.startswith('N'):
                tag = 'n'
            elif pos.startswith('V'):
                tag = 'v'
            elif pos.startswith('R'):
                tag = 'r'
            elif pos.startswith('J') or pos.startswith('A'):
                tag = 'a'
            else:
                tag = pos[:1].lower()
            lemma = None
            try:
                if self.lemmatization:
                    lemma = self.wnl.lemmatize(word, pos=tag)
                else:
                    lemma = word
            except:
                lemma = self.wnl.lemmatize(word)
            if lemma in self.stop_words:
                continue
            elif len(lemma) == 1:
                continue
            else:
                temp.append(lemma)
        return ' '.join(temp)
