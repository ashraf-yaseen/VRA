from datetime import datetime
from dateutil import parser
import re
import sys
import pickle
import glob


class GetArticleData:

    def __init__(self, address):
        self.multi_processing = False
        self.article_dataset = self.read_files(address)

    def read_files(self, address):
        file_names = glob.glob(address + '*/*.pickle')
        print('total files = ', len(file_names))
        #file_names = file_names[50:61]
        new_files = []
        for file_name in file_names:
            if 'duplicate_pmids.pickle' in file_name or \
               'duplicate_items.pickle' in file_name:
                continue
            else:
                new_files.append(file_name)
        print('total article files to be loaded = ', len(new_files))
        dict_articles = {}
        for file_name in new_files:
            dict_articles.update(pickle.load(open(file_name, 'rb')))
        print('Total {} articles were loaded'.format(len(dict_articles)))
        return dict_articles


    def check_date(self, pmid):
        single_data = self.article_dataset[pmid]
        str_time = single_data['pubdate']
        pub_year = ''
        if not type(str_time) == str:
            if str_time == None:
                print('No pubdate ', pmid)
                return pmid
            str_time = str_time.decode('utf-8')
        if re.match('19[0-9][0-9]-19[0-9][0-9]', str_time) or \
           re.match('19[0-9][0-9]-20[0-9][0-9]', str_time) or \
           re.match('20[0-9][0-9]-20[0-9][0-9]', str_time):
            #print(str_time)
            pub_year = str_time.split('-')[0]
        elif re.match('19[0-9][0-9]-[0-9][0-9]', str_time) or \
             re.match('20[0-9][0-9]-[0-9][0-9]', str_time):
             pub_year = str_time.split('-')[0]
        else:
            try:
                dt = parser.parse(str_time)
                pub_year = dt.year
            except:
                print('Error pubdate format ', str_time, '\t', pmid)
                return pmid
        if int(pub_year) < 1998:
            return None
        else:
            return pmid

    def collect_aa(self): #aa -> author, affiliation
        aa_dict = {}
        for pmid in self.article_dataset.keys():
            single_data = self.article_dataset[pmid]
            authors = []
            affiliations = []
            if 'author' in single_data:
                ath = single_data['author']
                if ath is not None:
                    authors = ath.split(';')
            if 'affiliation' in single_data:
                aff = single_data['affiliation']
                if aff is not  None:
                    affiliations = aff.split('\n')
            temp = []
            temp.append(authors)
            temp.append(affiliations)
            aa_dict[pmid] = temp
        return aa_dict

    def collect_article_title_n_abstract(self):
        article_title = {}
        article_abstract = {}
        pmids = list(self.article_dataset.keys())
        output = []
        for pmid in pmids:
            output.append(self.check_date(pmid))
        output = list(set(output))
        for pmid in output:
            if pmid is None:
                continue
            single_data = self.article_dataset[pmid]
            title = single_data['title']
            if not type(title) == str:
                continue
            elif title == None:
                continue
            elif title == '':
                continue
            article_title[pmid] = single_data['title']
            article_abstract[pmid] = single_data['abstract']
        #sys.exit(1)
        print('Total articles = ', len(article_title))
        return article_title, article_abstract

    def get_pmid_mesh_dict(self):
        pmid_mesh_dict = {}
        for pmid in self.article_dataset:
            single_item = self.article_dataset[pmid]
            mesh_terms = []
            temp_mesh_terms = single_item['mesh_terms']
            if temp_mesh_terms == None or temp_mesh_terms == '':
                continue
            for mesh_term in temp_mesh_terms.split(';'):
                if mesh_term == None or mesh_term =='':
                    continue
                try:
                    mesh_term = mesh_term.split(':')[1]
                except:
                    print('Error mesh_term ', mesh_term)
                mesh_term = mesh_term.strip().lower()
                if '/' in mesh_term:
                    print(mesh_term)
                    sys.exit(1)
                    mesh_term = mesh_term.split('/')[0]
                mesh_terms.append(mesh_term)
            pmid_mesh_dict[pmid] = list(set(mesh_terms))
        return pmid_mesh_dict

    def get_mesh_pmid_dict(self, pmid_mesh_dict):
        mesh_pmid_dict = {}
        for pmid in pmid_mesh_dict:
            mesh_terms = pmid_mesh_dict[pmid]
            for mesh_term in mesh_terms:
                if mesh_term.strip() == '':
                    continue
                if mesh_term in mesh_pmid_dict:
                    temp = mesh_pmid_dict[mesh_term]
                    temp.append(pmid)
                    mesh_pmid_dict[mesh_term] = temp
                else:
                    mesh_pmid_dict[mesh_term] = [pmid]
        return mesh_pmid_dict


def main():
    st = datetime.now()
    article_address = '/home/bgpatra/ftp.ncbi.nlm.nih.gov/processed_data/new_publication_data/'
    x = GetArticleData(article_address)
    print('time spent = ', datetime.now()-st)
    st = datetime.now()
    # this is for storing the pmids with authors and affiliations
    aa_dict = x.collect_aa()
    print(len(aa_dict))
    print('time spent = ', datetime.now()-st)
    file_name = '/home/bgpatra/ftp.ncbi.nlm.nih.gov/processed_data/all_author_affiliation.pickle'
    with open(file_name, 'wb') as f:
        pickle.dump(aa_dict, f, protocol=4)
    #this was for storing the article title and abstract which are published after 1997.
    '''article_title, article_abstract = x.collect_article_title_n_abstract()
    print(len(article_title))
    print('time spent = ', datetime.now()-st)
    file_name = '/home/bgpatra/ftp.ncbi.nlm.nih.gov/processed_data/all_abstract_aft_1997.pickle'
    with open(file_name, 'wb') as f:
        pickle.dump(article_abstract, f, protocol=4)
    file_name = '/home/bgpatra/ftp.ncbi.nlm.nih.gov/processed_data/all_title_aft_1997.pickle'
    with open(file_name, 'wb') as f:
        pickle.dump(article_title, f, protocol=4)'''


if __name__ == '__main__':
    main()
