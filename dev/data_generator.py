import os, pickle, hashlib, torch, dgl
from scipy import sparse as sp
import numpy as np
from torch.utils.data import Dataset

# Adapted from Dwivedi, Vijay Prakash, and Xavier Bresson.
# "A generalization of transformer networks to graphs." https://arxiv.org/abs/2012.09699

def laplacian_positional_encoding(g, pos_enc_dim):
    """
    Graph positional encoding v/ Laplacian eigenvectors
    Args:
        g: input graph
        pos_enc_dim: position encoding dimension

    Returns:

    """
    # Laplacian
    A = g.adjacency_matrix(scipy_fmt='csr').astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with scipy
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2) # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1].real).float()

    return g


def wl_positional_encoding(g):
    """
    WL-based absolute positional embedding
    """
    max_iter = 2
    node_color_dict = {}
    node_neighbor_dict = {}

    edge_list = torch.nonzero(g.adj().to_dense() != 0, as_tuple=False).numpy()
    node_list = g.nodes().numpy()

    # setting init
    for node in node_list:
        node_color_dict[node] = 1
        node_neighbor_dict[node] = {}

    for pair in edge_list:
        u1, u2 = pair
        if u1 not in node_neighbor_dict:
            node_neighbor_dict[u1] = {}
        if u2 not in node_neighbor_dict:
            node_neighbor_dict[u2] = {}
        node_neighbor_dict[u1][u2] = 1
        node_neighbor_dict[u2][u1] = 1

    # WL recursion
    iteration_count = 1
    exit_flag = False
    while not exit_flag:
        new_color_dict = {}
        for node in node_list:
            neighbors = node_neighbor_dict[node]
            neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
            color_string_list = [str(node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
            color_string = "_".join(color_string_list)
            hash_object = hashlib.md5(color_string.encode())
            hashing = hash_object.hexdigest()
            new_color_dict[node] = hashing
        color_index_dict = {k: v+1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
        for node in new_color_dict:
            new_color_dict[node] = color_index_dict[new_color_dict[node]]
        if node_color_dict == new_color_dict or iteration_count == max_iter:
            exit_flag = True
        else:
            node_color_dict = new_color_dict
        iteration_count += 1
        
    g.ndata['wl_pos_enc'] = torch.LongTensor(list(node_color_dict.values()))
    return g


def process_data(data_raw, graph_x_col_mean, graph_e_col_mean,
                 graph_x_min, graph_x_max, graph_e_min, graph_e_max, **kwargs):

    graph_raw = data_raw[0][0]
    if torch.isnan(graph_raw.ndata['feat']).any():
        inds = torch.where(torch.isnan(graph_raw.ndata['feat']))
        graph_raw.ndata['feat'][inds] = torch.take(graph_x_col_mean, inds[1])  # impute NA's with column mean
    graph_raw.ndata['feat'] = (graph_raw.ndata['feat'] - graph_x_min) / (graph_x_max - graph_x_min)
    graph_raw.ndata['feat'][graph_raw.ndata['feat'] > 1] = 1.0
    graph_raw.ndata['feat'][graph_raw.ndata['feat'] < 0] = 0.0

    if torch.isnan(graph_raw.edata['feat']).any():
        inds = torch.where(torch.isnan(graph_raw.edata['feat']))
        graph_raw.edata['feat'][inds] = torch.take(graph_e_col_mean, inds[1])  # impute NA's
    graph_raw.edata['feat'] = (graph_raw.edata['feat'] - graph_e_min) / (graph_e_max - graph_e_min)
    graph_raw.edata['feat'][graph_raw.edata['feat'] > 1] = 1.0
    graph_raw.edata['feat'][graph_raw.edata['feat'] < 0] = 0.0
    
    # if torch.isnan(data_raw[1]).any():  # partner
    #     inds = torch.where(torch.isnan(data_raw[1]))
    #     data_raw[1][inds] = torch.take(partner_col_mean, inds[1])
    # data_raw[1] = (data_raw[1] - partner_min) / (partner_max - partner_min)
    # data_raw[1][data_raw[1] > 1] = 1.0
    # data_raw[1][data_raw[1] < 0] = 0.0
        
    return data_raw

class LoadDataSet(Dataset):
    def __init__(self, data_files, net_params, graph_x_col_mean, graph_e_col_mean,
                 graph_x_min, graph_x_max, graph_e_min, graph_e_max, **kwargs):
        super(LoadDataSet, self).__init__()
        
        self.data = []
        for f in data_files:
            with open(f, 'rb') as infile:
                data_raw = pickle.load(infile)

            data_raw = process_data(data_raw, graph_x_col_mean, graph_e_col_mean,
                                    graph_x_min, graph_x_max, graph_e_min, graph_e_max)
            
            n_nodes = data_raw[0][0].number_of_nodes()
            assert np.unique(data_raw[0][0].edges()[1]).shape[0] == 1  # single source required
            target_node_id = np.unique(data_raw[0][0].edges()[1])[0]
            self.data.append([data_raw, n_nodes, target_node_id])

        # if net_params['lap_pos_enc']:
        #     print('Adding Laplacian positional encoding.')
            # for i in range(len(self.data)):
            #     self.data[i][0][0][0] = laplacian_positional_encoding(self.data[i][0][0][0], net_params['pos_enc_dim'])

        # if net_params['wl_pos_enc']:
        #     print('Adding WL positional encoding.')
            # for i in range(len(self.data)):
            #     self.data[i][0][0][0] = wl_positional_encoding(self.data[i][0][0][0])

        for i in range(len(self.data)):
            assert not torch.isnan(self.data[i][0][0][0].ndata['feat']).any()
            assert not torch.isnan(self.data[i][0][0][0].edata['feat']).any()
            assert not torch.isnan(self.data[i][0][1]).any()

            if net_params['lap_pos_enc']:
                self.data[i][0][0][0] = laplacian_positional_encoding(self.data[i][0][0][0], net_params['pos_enc_dim'])

            if net_params['wl_pos_enc']:
                self.data[i][0][0][0] = wl_positional_encoding(self.data[i][0][0][0])


    def __getitem__(self, index):
        graph_features = self.data[index]
        return graph_features
    
    def __len__(self):
        return len(self.data)
