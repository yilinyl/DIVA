import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


class VariantEncoder(nn.Module):
    def __init__(self,
                 ndata_dim_in,
                 ndata_dim_out,
                 device,
                 n_labels=20,
                 to_onehot=True,
                 embed_aa=False,
                 aa_embed_dim=None):

        super(VariantEncoder, self).__init__()
        self.device = device
        self.ndata_dim_in = ndata_dim_in
        self.ndata_dim_out = ndata_dim_out
        self.n_labels = n_labels
        self.to_onehot = to_onehot
        self.embed_aa = embed_aa
        self.aa_embed_dim = aa_embed_dim
        if self.to_onehot:
            self.embed_aa = False
            self.aa_embed_dim = self.n_labels
        self.variant_encoder = nn.Linear(self.ndata_dim_in + 2 * self.aa_embed_dim, self.ndata_dim_out)
        self.neighbor_encoder = nn.Linear(self.ndata_dim_in + self.aa_embed_dim, self.ndata_dim_out)
        self.aa_encoder = None
        if self.embed_aa:
            self.aa_encoder = nn.Embedding(self.n_labels, self.aa_embed_dim)

    def forward(self, g, alt_aa, var_idx, ref_aa_key='ref_aa', feat_key='feat'):
        if self.embed_aa:
            h_aa = self.aa_encoder(g.ndata[ref_aa_key])
            h_alt = self.aa_encoder(alt_aa)
        else:
            h_aa = F.one_hot(g.ndata[ref_aa_key], num_classes=self.n_labels)
            h_alt = F.one_hot(alt_aa, num_classes=self.n_labels)

        h = g.ndata[feat_key]
        var_mask = torch.ones(g.num_nodes(), dtype=torch.bool)
        var_mask[var_idx] = False

        h_var = self.variant_encoder(torch.cat([h_alt, h_aa[var_idx], h[var_idx, :]], dim=1))  # check dim
        h_nbr = self.neighbor_encoder(torch.cat([h_aa, h], dim=1)[var_mask, :])

        h_out = torch.zeros((g.num_nodes(), self.ndata_dim_out), device=self.device)
        h_out[var_idx] = h_var
        h_out[var_mask] = h_nbr

        return h_out


