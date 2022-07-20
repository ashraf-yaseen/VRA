import pickle


class GetGeoData:

    def __init__(self, file_name):
        self.geo_dataset = pickle.load(open(file_name, 'rb'))

    def collect_geo_citations(self):
        geo_citation_dict = {}
        for geo_id in self.geo_dataset:
            single_data = self.geo_dataset[geo_id]
            geo_citation_dict[geo_id] = single_data['citations']
        return geo_citation_dict

    def collect_geo_title_n_summary(self):
        geo_title = {}
        geo_summary = {}
        for geo_id in self.geo_dataset:
            single_data = self.geo_dataset[geo_id]
            geo_title[geo_id] = single_data['title']
            geo_summary[geo_id] = single_data['summary']
        return geo_title, geo_summary
