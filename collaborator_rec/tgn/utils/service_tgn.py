'''
This is the python file to prepare for service od collaborator recommendations
in order
1. get the user's CV 
2. parse them to get current years' publications only 2018-2020 default 
3. merge the file together with the training data
4. prepare all the testing pairs: exclude a list from the same deparment (currently we have affiliations lacking in our
pubmed crawled database, we're doing it the dirty way)

'''
import glob
import os
import fnmatch
import re
from pathlib import Path
import sys
from datetime import datetime
import pickle
from collections import defaultdict
import pandas as pd

from utils.prepare_servicedata_tgn import yearly_authors

# very last,import the functionality
sys.path.insert(0, os.path.abspath('../..'))
from RFOrecsys.pubmed_data_extraction.pubmed_query import PubCollection

# case insenstive find files
# if this works, modify the service data for GraphSage as well
def findfiles(which, where='.'):
    '''Returns list of filenames from `where` path matched by 'which'
       shell pattern. Matching is case-insensitive.'''
    
    # TODO: recursive param with walk() filtering
    rule = re.compile(fnmatch.translate(which), re.IGNORECASE)
    return [name for name in os.listdir(where) if rule.match(name)]



class service():
    def __init__(self, f_name = 'hulin', l_name = 'wu', m_name = '', path ='../service/', years = [2019, 2020], \
                 pubfile = 'data/20192020/[2019, 2020]_.pickle', exclude_users = '', options = 'mesh', val_ratio = 0.2):
        self.f_name = f_name
        self.l_name = l_name
        self.m_name = m_name
        if self.m_name.strip() == '':
            self.name = self.f_name + '_'  + self.l_name 
        else:
            self.name = self.f_name + '_' +  self.m_name +  '_' + self.l_name 
        self.path = path
        self.years = years
        self.pubfile = pubfile 
        self.exclude_users  = exclude_users
        self.options = options #'mesh' or 'pubs'
        self.val_ratio = val_ratio
        self.pub_collection = PubCollection(email = 'Jie.Zhu@uth.tmc.edu', tool = 'CollabRecommendationSystem')
        # self initiate the process needed 
        self.scrap_pubs()
        self.process()
        
    def scrap_pubs(self):
        """
        read in user's CV and grab the publication in self.years from pubmed 
        """
        cv = findfiles(self.l_name + '*.pdf', where= self.path)[0] #get the title form
        cv = self.path + cv
        pub_details = None
        pubs_output = self.path + self.l_name + self.f_name + '_pubs.pickle'
        if len(findfiles(self.l_name + self.f_name + '_pubs.pickle', where = self.path)) == 0:
            #not os.path.exists(pubs_output):
            print('Collecting publications from PubMed')
            time_pubmed = datetime.now()
            pub_details = self.pub_collection.process_pub_collection(cv, self.f_name, self.l_name, self.m_name)
            time_elapsed = datetime.now() - time_pubmed.pa
            print('Time taken to collect {} publications from PubMed {}'.format(len(pub_details), time_elapsed))
            pickle.dump(pub_details, open(pubs_output, 'wb'))
        else:
            name = findfiles(self.l_name + self.f_name + '_pubs.pickle', where = self.path)[0]
            pub_details = pickle.load(open(self.path + name, 'rb'))
        # filter to a particular year
        pub_details = {k:v for k, v in pub_details.items() if int(v['year']) >= self.years[0]}
        # make keys consistent
        new = {}
        for i, v in pub_details.items():
            v['mesh_terms'] = v.pop('keywords')
            v['pubdate'] = v.pop('year')
            v['pmid'] = str(i)
            new[str(i)] = v 
        self.pub_details = new
        print('filtered to total {} publications from PubMed'.format(len(self.pub_details)))
        return self.pub_details    

    
    def process(self):
        """
        To merge current users file together with the existing training file 2018-2020
        training, validation and test
        # let's get the data at [2019-2020]

        """
        save_path = 'data/service/' + self.name + '/' + self.options + '/'
        Path(save_path).mkdir(parents=True, exist_ok=True)
        # call the yearly_authors file and save the authors' file and collabs file
        yearly_authors(authfile = self.pubfile, service_dict = self.pub_details, savepath = save_path,\
                         l_name = self.l_name, f_name = self.f_name, m_name = self.m_name, options = self.options, 
                         val_ratio = self.val_ratio, exclude_users = self.exclude_users)   