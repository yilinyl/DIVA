"""
Modified from HAN example released by DGL team

This model shows an example of using dgl.metapath_reachable_graph on the original heterogeneous
graph.

Because the original HAN implementation only gives the preprocessed homogeneous graph, this model
could not reproduce the result in HAN as they did not provide the preprocessing code, and we
constructed another dataset from ACM with a different set of papers, connections, features and
labels.
"""

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv, EGATConv

from .VariantEncoder import VariantEncoder


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)  # (N, D * K)


class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_node_feats : input feature dimension
    out_node_feats : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability
    """

    def __init__(self,
                 meta_paths,
                 in_node_feats,
                 out_node_feats,
                 layer_num_heads,
                 dropout,
                 use_efeats=False,
                 in_edge_feats=None,
                 out_edge_feats=None):
        super(HANLayer, self).__init__()

        if use_efeats:
            self.gat_unit = EGATConv(
                in_node_feats,
                in_edge_feats,
                out_node_feats,
                out_edge_feats,
                num_heads=layer_num_heads
            )
        else:
            self.gat_unit = GATConv(
                in_node_feats,
                out_node_feats,
                layer_num_heads,
                dropout,
                dropout,
                activation=F.elu,
                allow_zero_in_degree=True,
            )
        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(self.gat_unit)

        # for i in range(len(meta_paths)):
        #     self.gat_layers.append(
        #         GATConv(
        #             in_node_feats,
        #             out_node_feats,
        #             layer_num_heads,
        #             dropout,
        #             dropout,
        #             activation=F.elu,
        #             allow_zero_in_degree=True,
        #         )
        #     )
        self.semantic_attention = SemanticAttention(
            in_size=out_node_feats * layer_num_heads
        )
        # self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self.meta_paths = list(meta_paths)
        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        """

        Args:
            g: [DGLGraph] the heterogeneous graph
            h: [tensor] input features

        Returns:

        """
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[
                    meta_path
                ] = dgl.metapath_reachable_graph(g, [meta_path])

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class HAN(nn.Module):
    def __init__(self,
                 meta_paths,  # Metapath in the form of a list of edge types
                 hidden_size,  # hidden size for each attention head
                 n_classes,
                 n_heads,
                 n_layers,
                 dropout,
                 readout,
                 ndata_dim_in,
                 ndata_dim_out,
                 device,
                 n_labels=21,
                 to_onehot=True,
                 embed_aa=False,
                 aa_embed_dim=None,
                 use_weight_in_loss=False,
                 **kwargs
                 ):
        super(HAN, self).__init__()
        self.device = device
        self.variant_processor = VariantEncoder(ndata_dim_in, ndata_dim_out, self.device, n_labels,
                                                to_onehot, embed_aa, aa_embed_dim)
        self.readout = readout
        self.use_weight_in_loss = use_weight_in_loss

        in_size = self.variant_processor.ndata_dim_out
        self.layers = nn.ModuleList()
        self.layers.append(
            HANLayer(meta_paths, in_size, hidden_size, n_heads, dropout)
        )
        for l in range(1, n_layers):
            self.layers.append(
                HANLayer(
                    meta_paths,
                    hidden_size * n_heads,
                    hidden_size,
                    n_heads,
                    dropout,
                )
            )
        self.out_layer = nn.Linear(hidden_size * n_heads, n_classes)
        self.predict = nn.Sigmoid()

    def forward(self, g, alt_aa, var_idx, ref_aa_key='ref_aa', feat_key='feat'):
        h = self.variant_processor(g, alt_aa, var_idx, ref_aa_key, feat_key)

        for gnn in self.layers:
            h = gnn(g, h)

        g.ndata['h'] = h
        hg = dgl.readout_nodes(g, 'h', op=self.readout)  # TODO: check dim

        return self.out_layer(hg)


    def loss(self, logits, label):
        if self.use_weight_in_loss:
            V = label.size(0)
            label_count = torch.bincount(label.long())
            cluster_sizes = torch.zeros(label_count.size(0)).long().to(self.device)
            cluster_sizes[torch.arange(label_count.size(0)).long()] = label_count
            weight = (V - cluster_sizes).float() / V
            weight *= (cluster_sizes>0).float()

            criterion = nn.BCEWithLogitsLoss(weight=weight[label.long()])
        else:
            criterion = nn.BCEWithLogitsLoss()

        loss = criterion(logits, label.float())

        return loss
