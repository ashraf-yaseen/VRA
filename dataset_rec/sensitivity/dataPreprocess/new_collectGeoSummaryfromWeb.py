#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This program is to collect the geo details from the web

import urllib.request as req
from bs4 import BeautifulSoup as bs
import pickle
import os
from preprocessing import PreProcessing


class GeoDataCollection:
    def __init__(self, final_save_add, recollect=False):
        self.final_save_add = final_save_add
        self.geo_data = dict()
        self.geo_ids = list()
        if os.path.exists(self.final_save_add) and recollect is False:
            self.geo_data = pickle.load(open(self.final_save_add, 'rb'))
        print('Total geo datasets exist = ', len(self.geo_data))
        self.link = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc="
        self.pp = PreProcessing()

    def process_geo(self, all_geo_ids_path):
        """This will process all geo ids and collect data.
        :parameter
            all_geo_ids_path (str): path to all geo ids
        """
        self.geo_ids = list(set(pickle.load(open(all_geo_ids_path, 'rb'))) - set(self.geo_data.keys()))
        print('Total geo datasets need to be collected = ', len(self.geo_ids))
        all_gse = []

        from multiprocessing import Pool
        pool = Pool(processes=5)
        n_processes = pool._processes
        all_gse = list(pool.map(self.collect_geo_data, self.geo_ids))

        '''for geo_id in self.geo_ids:
            all_gse.append(self.collect_geo_data(geo_id))
            if len(all_gse) % 10 == 0:
                print('Data collected = ', len(all_gse))'''

        for gse in all_gse:
            if gse is None:
                print('GSE is none ')
                continue
            gse['p_title'] = self.pp.process_text(gse['title'])
            gse['p_summary'] = self.pp.process_text(gse['summary'])
            self.geo_data[gse['i_d']] = gse
        print('Total size of new collected GEO Dataset = ', len(self.geo_data))
        pickle.dump(self.geo_data, open(self.final_save_add, 'wb'))

    def collect_geo_data(self, geo_id):
        """This will collect data for single geo id.
        :parameter
                geo_id (str): single geo id
        :return
                dict_gse (dict): dict of all collected data
        """
        title, summary, author, affiliation = '', '', '', ''
        contact_name, e_mail = '', ''
        citations = []
        new_link = self.link + geo_id
        # print(new_link)
        my_file = None
        try:
            f = req.urlopen(new_link, timeout=1)
            my_file = f.read().decode('utf-8')
        except:
            return
        list_of_tr = str(my_file).split('</tr>')
        for item in list_of_tr:
            if '<td nowrap>Title</td>' in item or '<td>Title</td>' in item:
                soup = bs(item, 'html.parser')
                text = soup.get_text()
                text = text.replace('\\n', ' ').strip()
                title = text[6:]
                # print(title)

            if '<td nowrap>Summary</td>' in item or '<td>Summary</td>' in item:
                soup = bs(item, 'html.parser')
                text = soup.get_text()
                text = text.replace('\\n', ' ').strip()
                summary = text[8:]
                # print(summary)

            if '<td nowrap>Contributor(s)</td>' in item or '<td>Contributor(s)</td>' in item:
                soup = bs(item, 'html.parser')
                text = soup.get_text()
                text = text.replace('\\n', ' ').strip()
                author = text[15:].strip()
                # print(author)

            if '<td nowrap>Organization name</td>' in item or '<td>Organization name</td>' in item:
                soup = bs(item, 'html.parser')
                text = soup.get_text()
                text = text.replace('\\n', ' ').strip()
                affiliation = text[17:].strip()
                # print(affliation)

            if '<td nowrap>Citation(s)</td>' in item or '<td>Citation(s)</td>' in item:
                soup = bs(item, 'html.parser')
                text = soup.get_text()
                text = text.replace('\\n', ' ').strip()
                citations = [x.strip() for x in text[12:].strip().split(',')]
                # print(citations)
            if '<td nowrap>Contact name</td>' in item or '<td>Contact name</td>' in item:
                soup = bs(item, 'html.parser')
                text = soup.get_text()
                text = text.replace('\\n', ' ').strip()
                contact_name = text[12:].strip()
                # print(contact_name)
            if '<td nowrap>E-mail</td>' in item or '<td>E-mail</td>' in item:
                soup = bs(item, 'html.parser')
                text = soup.get_text()
                text = text.replace('\\n', ' ').strip()
                e_mail = text[6:].strip()
                # print(e_mail)
        dict_gse = {}
        dict_gse['i_d'] = geo_id
        dict_gse['title'] = title
        dict_gse['summary'] = summary
        dict_gse['author'] = author
        dict_gse['affiliation'] = affiliation
        dict_gse['citations'] = citations
        dict_gse['contact'] = contact_name
        dict_gse['email'] = e_mail
        return dict_gse


def main():
    print('To recollect data make the recollect variable True')
    x = GeoDataCollection('geo_dataset.pickle')
    x.process_geo('all_geo_ids.pickle')


if __name__ == '__main__':
    main()
