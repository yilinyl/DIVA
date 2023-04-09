import torch
from torch import nn
import torch.functional as F
import dgl
from dgl.nn.pytorch import GATConv
from .graphtransformer_v0 import MLPReadout


class GAT(nn.Module):
    def __init__(self,
                 ndata_dim_in,
                 hidden_size,
                 out_dim,
                 n_classes,
                 n_heads,
                 n_layers,
                 readout,
                 activation=nn.ELU(),
                 n_labels=21,
                 aa_embed_dim=None,
                 residual=True,
                 dropout=0.0,
                 classify=True,
                 **kwargs):
        super(GAT, self).__init__()
        self.act = activation
        self.aa_embedding = nn.Embedding(n_labels, aa_embed_dim)
        self.readout = readout
        # self.ndata_encoder = nn.Linear(aa_embed_dim + ndata_dim_in, hidden_size)
        self.classify = classify
        self.gat_layers = nn.ModuleList()
        self.n_heads = n_heads

        if not isinstance(self.n_heads, list):
            self.n_heads = [n_heads] * n_layers

        h_dim_in = aa_embed_dim + ndata_dim_in
        h_dim_out = hidden_size  # 64
        for l in range(n_layers - 1):
            self.gat_layers.append(GATConv(h_dim_in,
                                           h_dim_out,
                                           self.n_heads[l],
                                           feat_drop=dropout,
                                           attn_drop=dropout,
                                           residual=residual,
                                           activation=self.act))
            # d_out = h_dim_out * n_heads # 128
            h_dim_in = h_dim_out * self.n_heads[l]  # 128
            h_dim_out = h_dim_in * 2

        self.gat_out_layer = GATConv(h_dim_in,
                                     h_dim_out,
                                     self.n_heads[-1],
                                     feat_drop=dropout,
                                     attn_drop=dropout,
                                     residual=residual,
                                     activation=None)

        self.MLP_layer = MLPReadout(h_dim_out, n_classes)
        self.predict = nn.Sigmoid()
        self.criterion = nn.BCEWithLogitsLoss()


    def add_aa_embedding(self, g, alt_aa):
        h_aa = self.aa_embedding(g.ndata['ref_aa'])  # n_nodes x hidden_dim
        g.ndata['h_aa'] = h_aa
        h_alt = self.aa_embedding(alt_aa)  # n_batch x hidden_dim

        seq_emb = []
        for b in range(g.batch_size):
            g_sub = dgl.slice_batch(g, b)
            # seq_emb.append(torch.matmul(g_sub.ndata['h_aa'], h_alt[b]))
            seq_emb.append(g_sub.ndata['h_aa'] * h_alt[b])
        h_aa = torch.cat(seq_emb)

        return h_aa


    def forward(self, g, alt_aa):
        assert not g.ndata['feat'].detach().cpu().isnan().any()

        if 'h_aa' not in g.node_attr_schemes().keys():
            g.ndata['h_aa'] = self.add_aa_embedding(g, alt_aa)

        # h = self.ndata_encoder(torch.cat([g.ndata['h_aa'], g.ndata['feat']], dim=-1))
        h = torch.cat([g.ndata['h_aa'], g.ndata['feat']], dim=-1)

        for i, att_conv in enumerate(self.gat_layers):
            h = att_conv(g, h)
            h = h.flatten(1)  # concat embeddings from different heads

        h = self.act(torch.mean(self.gat_out_layer(g, h), dim=1))

        g.ndata['h'] = h

        hg = dgl.readout_nodes(g, 'h', op=self.readout)

        if self.classify:
            return self.MLP_layer(hg)

        return hg


    def loss(self, logits, label):
        return self.criterion(logits, label.float())
