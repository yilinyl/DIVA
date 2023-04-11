import torch, dgl
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import GATConv, EGATConv, GINConv
from .graphtransformer_v0 import GraphTransformer, MLPReadout
from .gat import GAT
import numpy as np


class MultiModel(nn.Module):
    def __init__(self,
                 seq_params,
                 struct_params,
                 out_dim,
                 agg='add',
                 n_classes=1,
                 embed_graph=False, **kwargs):

        super(MultiModel, self).__init__()
        # self.aa_embedding = nn.Embedding(n_labels, aa_embed_dim)
        # seq_params['aa_embed_dim'] = aa_embed_dim
        # struct_params['aa_embed_dim'] = aa_embed_dim

        self.seq_gnn = GraphTransformer(**seq_params)
        self.struct_gnn = GAT(**struct_params)
        self.hidden_size = self.seq_gnn.out_dim + self.struct_gnn.out_dim
        self.out_dim = out_dim
        self.embed_graph = embed_graph

        if agg == 'add':
            self.fusion = nn.Linear(self.hidden_size, self.out_dim)

        self.MLP_layer = MLPReadout(self.out_dim, n_classes)
        self.predict = nn.Sigmoid()

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, g_seq, g_struct, alt_aa):

        # h_aa_seq = self.aa_embedding(g_seq.ndata['ref_aa'])  # n_nodes x hidden_dim
        # h_aa_str = self.aa_embedding(g_struct.ndata['ref_aa'])
        # g_seq.ndata['h_aa'] = h_aa_seq
        # h_alt = self.aa_embedding(alt_aa)  # n_batch x hidden_dim
        #
        # g_struct.ndata['h_aa'] = h_aa_str

        f_seq = self.seq_gnn(g_seq, alt_aa)  # graph level embedding
        f_struct = self.struct_gnn(g_struct, alt_aa)

        if self.embed_graph:
            g_seq.ndata['h'] = f_seq
            f_seq = dgl.readout_nodes(g_seq, 'h', op=self.seq_gnn.readout)

            g_struct.ndata['h'] = f_struct
            f_struct = dgl.readout_nodes(g_struct, 'h', op=self.struct_gnn.readout)

        # aggregation
        # if self.agg == 'add':
        h_all = torch.cat([f_seq, f_struct], dim=-1)
        h_all = self.fusion(h_all)
        h_out = self.MLP_layer(h_all)

        return h_out


    def loss(self, logits, label):
        return self.criterion(logits, label.float())
