#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This class will process take CV and name. This will compare two publication lists and return the common pubs.


from Bio import Entrez
from lxml import etree as ET
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter # process_pdf
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from ..preprocessing.preprocessing import PreProcessing
#from preprocessing.processing_metamap import PMM
from .pubmed_web_parser import parse_xml_web
#from .pubmed_web_parser import parse_pubmed_web_tree
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


class PubCollection:
    def __init__(self, email, tool):
        Entrez.email = email
        Entrez.tool = tool
        self.pp = PreProcessing()
        # self.pmm = PMM()

    def get_pub_details(self, pmids): # this one is using pubmed_parser
        pubmed_data = {}
        for pmid in pmids:
            dict_out = parse_xml_web(pmid)
            pubmed_data[pmid] = dict_out
        return pubmed_data
        '''articles = self.fetch_pub_from_pubmed(pmids)
        for pmid, article in zip(pmids, articles):
            dict_out = parse_pubmed_web_tree(article)
            pubmed_data[pmid] = dict_out
        return pubmed_data'''

    '''def fetch_pub_from_pubmed(self, pmids):
        root = None
        try:
            handle = Entrez.efetch(db='pubmed', id=pmids, retmode='xml')
            tree = ET.parse(handle)
            root = tree.getroot()
            # print(ET.tostring(root, pretty_print=True))
        except:
            print('waiting')
            import time
            time.sleep(5)
            handle = Entrez.efetch(db='pubmed', id=pmids, retmode='xml')
            tree = ET.parse(handle)
            root = tree.getroot()
        with open('output_xml.txt', 'w') as f:
            x = ET.tostring(root, pretty_print=True)
            f.write(str(x))
        articles = root.findall('PubmedArticle')
        return articles'''

    '''def get_pub_details(self, pmids):
        publications = {}
        root = self.fetch_pub_from_pubmed(pmids)
        if root is None:
            return ''
        #TODO: Modify according to pubmed_parser
        #print(ET.tostring(root, pretty_print=True))
        articles = root.findall('PubmedArticle')
        for article, pmid in zip(articles, pmids):
            title = ''
            abstract = ''
            year = None
            for item in article.iter():
                #print(item.tag)
                if item.tag == 'ArticleTitle':
                    title = ET.tostring(item, method='text', encoding='UTF-8').strip().decode('UTF-8')
                if item.tag == 'Abstract':
                    abstract = ET.tostring(item, method='text', encoding='UTF-8').strip().decode('UTF-8')
                if item.tag == 'PubDate':
                    for child in item:
                        if child.tag == 'Year':
                            year = child.text
            if title.endswith('.'):
                title = title[:-1]
            publications[pmid] = [title, abstract, year]
        return publications'''

    def extract_author_pmids(self, f_name, l_name, m_name):
        query = None
        if m_name == '':
            query = l_name + ', ' + f_name + '[Author]'
        else:
            query = l_name + ', ' + f_name + ' ' + m_name + '[Full Author Name]'
            #query = l_name + ' ' + f_name[:1].upper() + m_name[:1].upper() + '[Author]'
        #print(query)
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        handle = Entrez.esearch(db='pubmed', sort='relevance', retmax='1000', retmode='xml', term=query)
        tree = ET.parse(handle)
        root = tree.getroot()
        # print(ET.tostring(root, pretty_print=True))
        count = 0
        pmids = []
        flag = 0
        for item in root.iter():
            if item.tag == 'Count' and flag == 0:
                flag = 1
                count = int(item.text)
            if count > 0 and item.tag == 'IdList':
                for child in item:
                    pmids.append(child.text)
        #print(count)
        #print(pmids)
        return pmids

    def parse_pdf(self, cv_path):
        rsrcmgr = PDFResourceManager()
        sio = StringIO()
        laparams = LAParams()
        device = TextConverter(rsrcmgr, sio, laparams=laparams) #newone
        #device = TextConverter(rsrcmgr, sio, codec='utf-8', laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        fp = open(cv_path, 'rb')
        for page in PDFPage.get_pages(fp):
            interpreter.process_page(page)
        fp.close()
        text = sio.getvalue()
        device.close()
        sio.close()
        return text

    def process_cv_text(self, cv_text):
        formated_text = ''
        for line in cv_text.split('\n'):
            line = line.strip()
            if line == '':
                continue
            if not line.endswith('.'):
                formated_text += line + ' '
            else:
                formated_text += line + '\n'
        formated_text = formated_text.replace('  ', ' ')
        return formated_text

    def process_pub_collection(self, cv_path, f_name, l_name, m_name):
        pmids = self.extract_author_pmids(f_name, l_name, m_name)
        #print(len(pmids))
        publications = self.get_pub_details(pmids)
        #print(len(publications))
       	#import pickle
        #pickle.dump(publications, open('jeffpub.pickle', 'wb'))
        cv_text = self.process_cv_text(self.parse_pdf(cv_path)).lower()
        #print(cv_text)
        final_publications = {}
        for pmid in publications:
            pub_details = publications[pmid]
            title = pub_details['title']
            if title.endswith('.'):
                title = title[:-1]
            if title == '' or title == ' ':
                continue
            #print(title.lower())
            if title.lower() in cv_text:
                title = pub_details['title']
                abstract = pub_details['abstract']
                pub_details['ptitle'] = self.pp.process_text(title)
                pub_details['pabstract'] = self.pp.process_text(abstract)
                # pub_details['mtitle'] = self.pmm.process_metamap(title)
                # pub_details['mabstract'] = self.pmm.process_metamap(abstract)
                pub_details['link'] = 'https://www.ncbi.nlm.nih.gov/pubmed/?term=' + pmid
                final_publications[pmid] = pub_details
        return final_publications


def main():
    #f_name = 'Kirk'
    #l_name = 'Roberts'
    f_name = 'Hulin'
    l_name = 'Wu'
    m_name = ''
    cv_path = '../resources/HulinCV.pdf'
    email = 'Braja.G.Patra@uth.tmc.edu'
    tool = 'RecommendationSystem'
    author_details = PubCollection(email, tool).process_pub_collection(cv_path, f_name, l_name, m_name)


if __name__ == '__main__':
    main()
