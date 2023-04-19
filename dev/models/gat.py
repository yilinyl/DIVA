import torch
from torch import nn
import torch.functional as F
import dgl
from dgl.nn.pytorch import GATConv, EGATConv
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
                 use_efeat=True,
                 edata_dim_in=None,
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
        self.use_efeat = use_efeat
        # self.feat_dropout = nn.Dropout(dropout)

        if not isinstance(self.n_heads, list):
            self.n_heads = [n_heads] * n_layers

        h_dim_in = aa_embed_dim + ndata_dim_in
        h_dim_out = hidden_size  # 64

        if self.use_efeat:
            # e_dim_in = edata_dim_in
            # f_dim_out = h_dim_out
            self.gat_layers.append(EGATConv(h_dim_in,
                                            edata_dim_in,
                                            hidden_size,
                                            hidden_size,
                                            self.n_heads[0]))
            for l in range(1, n_layers - 1):
                self.gat_layers.append(EGATConv(hidden_size * self.n_heads[l-1],
                                                hidden_size * self.n_heads[l-1],
                                                hidden_size,
                                                hidden_size,
                                                self.n_heads[l]))
                # h_dim_in = h_dim_out * self.n_heads[l]  # 128
                # e_dim_in = f_dim_out * self.n_heads[l]
                # h_dim_out = h_dim_in * 2
                # f_dim_out = h_dim_out

            self.gat_out_layer = EGATConv(hidden_size * self.n_heads[-2],
                                          hidden_size * self.n_heads[-2],
                                          out_dim,
                                          out_dim,
                                          self.n_heads[-1])
        else:
            for l in range(n_layers - 1):
                self.gat_layers.append(GATConv(h_dim_in,
                                               h_dim_out,
                                               self.n_heads[l],
                                               attn_drop=dropout,
                                               residual=residual,
                                               activation=self.act))
                # d_out = h_dim_out * n_heads # 128
                h_dim_in = h_dim_out * self.n_heads[l]  # 128
                h_dim_out = h_dim_in * 2

            self.gat_out_layer = GATConv(h_dim_in,
                                         h_dim_out,
                                         self.n_heads[-1],
                                         attn_drop=dropout,
                                         residual=residual,
                                         activation=None)
        self.out_dim = out_dim
        self.MLP_layer = MLPReadout(self.out_dim, n_classes)
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
        f = g.edata['feat']
        if self.use_efeat:
            for i, att_conv in enumerate(self.gat_layers):
                h, f = att_conv(g, h, f)
                h = self.act(h).flatten(1)  # concat embeddings from different heads
                f = self.act(f).flatten(1)

            h, f = self.gat_out_layer(g, h, f)

        else:
            for i, att_conv in enumerate(self.gat_layers):
                h = att_conv(g, h)  # activation applied in GAT layer
                h = h.flatten(1)  # concat embeddings from different heads
            h = self.gat_out_layer(g, h)

        h = self.act(h.mean(dim=1))

        g.ndata['h'] = h

        hg = dgl.readout_nodes(g, 'h', op=self.readout)

        if self.classify:
            return self.MLP_layer(hg)

        return hg


    def loss(self, logits, label):
        return self.criterion(logits, label.float())
