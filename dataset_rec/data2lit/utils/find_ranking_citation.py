class CitationRanking:

    def __init__(self, citations):
        self.geo_citation_dict = citations
        self.good_geo_rec = 0 #recommendations that score! (in citations)
        self.top1_geo_rec = 0 #recommendations that score on the first one!
        self.bad_geo_rec = 0 #recommendations that bad! (not in citations at all )
        self.geo_without_citations = 0 #number of geos that actually has no citation

    def get_values(self):
        return self.good_geo_rec, self.top1_geo_rec, self.bad_geo_rec, self.geo_without_citations
    
    '''
    #i dont think i need this 
    def find_associated_citation(self, geo_id, selected_pmids_dict):
        citations = self.geo_citation_dict[geo_id]
        if len(citations) == 0:
            citation_found = []
            for pmid in selected_pmids_dict:
                val = selected_pmids_dict[pmid]
                if val > 0.95: #this value is not correct
                #some argsort stuff
                    pair = (pmid, val)
                    citation_found.append(pair)
            if len(citation_found) > 0:
                print('{}\t{}'.format(geo_id, citation_found))
        else:
            pass
    '''

    def find_citations(self, geo_id, selected_pmids):
        citations = self.geo_citation_dict[geo_id]# actual citations
        if len(citations) == 0:
            self.geo_without_citations += 1
        else:
            present = list(set(citations).intersection(set(selected_pmids)))
            if len(present) == 0:
                self.bad_geo_rec += 1
            else:
                self.good_geo_rec += 1
                for paper in present:
                    if selected_pmids.index(paper) == 0:
                        self.top1_geo_rec += 1
                        break
