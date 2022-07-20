
class WriteResult:
    def __init__(self):
        self.link = 'https://www.ncbi.nlm.nih.gov/pubmed/?term='

    def write(self, address, selected_pmids_dict):
        output = ''
        for pmid in selected_pmids_dict:
            output += pmid + '\t' + str(selected_pmids_dict[pmid]) + '\t' + self.link + pmid + '\n'
        with open(address, 'w') as f:
            f.write(output)
