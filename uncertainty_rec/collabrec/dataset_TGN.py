"""
references:https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
"""

import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset, TemporalData, download_url

class CollabDataset(InMemoryDataset):
    # url = 'http://snap.stanford.edu/jodie/{}.csv'
    names = ['reddit', 'wikipedia', 'mooc', 'lastfm', 'collab']

    def __init__(self, root, name, transform=None, pre_transform=None):
        # root should goes back to tgn folder
        self.name = name.lower()
        # assert self.name in self.names

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw') # change 

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        return f'{self.name}.csv'

    @property
    def processed_file_names(self):
        return 'data.pt'

    #def download(self):
        #download_url(self.url.format(self.name), self.raw_dir)

    def process(self):
        import pandas as pd

        df = pd.read_csv(self.raw_paths[0], skiprows=1, header=None)

        src = torch.from_numpy(df.iloc[:, 0].values).to(torch.long)
        dst = torch.from_numpy(df.iloc[:, 1].values).to(torch.long)
        dst += int(src.max()) + 1
        t = torch.from_numpy(df.iloc[:, 2].values).to(torch.long)
        y = torch.from_numpy(df.iloc[:, 3].values).to(torch.long)
        # we need to compute the message to 3 most recent articles between these two authors
        msg = torch.from_numpy(df.iloc[:, 4:].values).to(torch.float) # we dont have these msg, but we could compute

        data = TemporalData(src=src, dst=dst, t=t, msg=msg, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return f'{self.name.capitalize()}()'