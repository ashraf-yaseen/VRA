import time
import requests
from lxml import html, etree
from itertools import chain
import pickle
try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen


def load_xml(pmid, sleep=10):
    """
    Load XML file from given pmid from eutils site
    return a dictionary for given pmid and xml string from the site
    sleep: how much time we want to wait until requesting new xml
    """
    link = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&id=%s" % str(pmid)
    page = requests.get(link)
    tree = html.fromstring(page.content)
    if sleep is not None:
        time.sleep(sleep)
    return tree


def parse_pubmed_web_tree(tree):
    """
    Giving tree, return simple parsed information from the tree
    """

    if len(tree.xpath('//articletitle')) != 0:
        print(tree.xpath('//articletitle'))
        #for title in tree.xpath('//articletitle'):
            #print(title.text)
        #modified by Ginny
        title = ' '.join([title.text for title in tree.xpath('//articletitle') if title.text != None ])
        print(title)
    elif len(tree.xpath('//booktitle')) != 0:
        title = ' '.join([title.text for title in tree.xpath('//booktitle') if title.text != None ])
    else:
        title = ''

    abstract_tree = tree.xpath('//abstract/abstracttext')
    abstract = ' '.join([stringify_children(a).strip() for a in abstract_tree])

    if len(tree.xpath('//article//title')) != 0:
        journal = ';'.join([t.text.strip() for t in tree.xpath('//article//title')])
    else:
        journal = ''

    pubdate = tree.xpath('//pubmeddata//history//pubmedpubdate[@pubstatus="medline"]')
    pubdatebook = tree.xpath('//pubmedbookdata//history//pubmedpubdate[@pubstatus="medline"]')
    if len(pubdate) >= 1 and pubdate[0].find('year') is not None:
        year = pubdate[0].find('year').text
    elif len(pubdatebook) >= 1 and pubdatebook[0].find('year') is not None:
        year = pubdatebook[0].find('year').text
    else:
        year = ''

    affiliations = list()
    if tree.xpath('//affiliationinfo/affiliation') is not None:
        for affil in tree.xpath('//affiliationinfo/affiliation'):
            affiliations.append(affil.text)
    affiliations_text = '; '.join(affiliations)

    authors_tree = tree.xpath('//authorlist/author')
    authors = list()
    if authors_tree is not None:
        for a in authors_tree:
            firstname = a.find('forename').text if a.find('forename') is not None else ''
            lastname = a.find('lastname').text if a.find('forename') is not None else ''
            fullname = (firstname + ' ' + lastname).strip()
            if fullname == '':
                fullname = a.find('collectivename').text if a.find('collectivename') is not None else ''
            authors.append(fullname)
        authors_text = '; '.join(authors)
    else:
        authors_text = ''

    keywords = ''
    keywords_mesh = tree.xpath('//meshheadinglist//meshheading')
    keywords_book = tree.xpath('//keywordlist//keyword')
    if len(keywords_mesh) > 0:
        mesh_terms_list = []
        for m in keywords_mesh:
            keyword = m.find('descriptorname').attrib.get('ui', '') + \
                ":" + \
                m.find('descriptorname').text
            mesh_terms_list.append(keyword)
        keywords = ';'.join(mesh_terms_list)
    elif len(keywords_book) > 0:
        keywords = ';'.join([m.text or '' for m in keywords_book])
    else:
        keywords = ''

    doi = ''
    article_ids = tree.xpath('//articleidlist//articleid')
    if len(article_ids) >= 1:
        for article_id in article_ids:
            if article_id.attrib.get('idtype') == 'doi':
                doi = article_id.text

    dict_out = {'title': title,
                'abstract': abstract,
                'journal': journal,
                'affiliations': affiliations_text,
                'authors': authors_text,
                'keywords': keywords,
                'doi': doi,
                'year': year}
    return dict_out


def stringify_children(node):
    """
    Filters and removes possible Nones in texts and tails
    ref: http://stackoverflow.com/questions/4624062/get-all-text-inside-a-tag-in-lxml
    """
    parts = ([node.text] +
             list(chain(*([c.text, c.tail] for c in node.getchildren()))) +
             [node.tail])
    return ''.join(filter(None, parts))

def parse_xml_web(pmid, sleep=None):
    """
    Give pmid, load and parse xml from Pubmed eutils
    if save_xml is True, save xml output in dictionary
    """
    sleep = 0.5
    tree = load_xml(pmid, sleep=sleep)
    dict_out = parse_pubmed_web_tree(tree)
    return dict_out


def main():
    #read in a list of files needed to be processed, can edit to be all pubs, rescrapped everything
    #rescapping all with 'pubsIncitation.ls'
    #print(parse_xml_web(str(21119629)))


    with (open("../../DataRec/IIdata/immport17Pubs.ls", "rb")) as openfile:
        pubs2scrap = pickle.load(openfile)
    
    
    ##next time we can start from where we have left
    # not to save in one lump, do it in batches 
    #add into the dictionary 
    #from original 0 to 60000
    count = 0
    #count = 60000
    every = 10000 
    new_pubs_dict ={}
    #from original everything to 60,000
    for pub in pubs2scrap:#[60000:]:
        pub_dict = parse_xml_web(str(pub))
        #print(pub_dict)
        #break #debug
        if pub_dict:
            new_pubs_dict[str(pub)] = pub_dict
            count +=1 
        else:
            print('not results returned:', str(pub))
        if count % every == 0:
            with open('../../DataRec/data/scraped_pubs_dict_' + str(count) + '.pickle', 'wb') as handle:
                    pickle.dump(new_pubs_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
   
    #print one, dump all 
    with open('../../DataRec/IIdata/scraped_immport17pubs_dict.pickle', 'wb') as handle:
        pickle.dump(new_pubs_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    #print an example 
    for k, v in new_pubs_dict.items():
        print(k, v)
        break 


if __name__ == '__main__':
    main()

