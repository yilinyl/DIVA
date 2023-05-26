import numpy as np
import torch
import dgl
from torch.utils.data import Dataset


class GraphDataSetBase(Dataset):
    def __init__(self):
        super(GraphDataSetBase, self).__init__()
        self.data = []
        self.label = []
        self.n_nodes = []
        self.n_edges = []
        self.nfeat_key = 'feat'
        self.efeat_key = 'feat'

    def __getitem__(self, index):
        graph, label, alt_aa, target_idx, var_id = self.data[index]

        return graph, label, alt_aa, target_idx, var_id

    def __len__(self):
        return len(self.data)

    def count_positive(self):
        return sum(self.label)

    def get_seq_struct_map(self):
        return self.seq2struct_dict

    def get_ndata_dim(self, feat_name='feat'):
        g = self.data[0][0]

        return g.ndata[feat_name].shape[1]

    def get_edata_dim(self, feat_name='feat'):
        g = self.data[0][0]

        return g.edata[feat_name].shape[1]

    def process(self, df_in, norm_feat=False):
        pass
    # def get_patho_num(self):
    #     return np.mean(self.n_patho)

    def dataset_summary(self, stats_dict=None):
        if not stats_dict:
            return np.mean(self.n_nodes), np.mean(self.n_edges)
        return np.mean(stats_dict['nodes']), np.mean(stats_dict['edges'])
