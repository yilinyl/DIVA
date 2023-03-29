import torch, dgl
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import numpy as np


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}
    return func

def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}
    return func

# Improving implicit attention scores with explicit edge features, if available
def imp_exp_attn(implicit_attn, explicit_edge):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
    """
    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}
    return func

# To copy edge features to be passed to FFN_e
def out_edge_features(edge_feat):
    def func(edges):
        return {'e_out': edges.data[edge_feat]}
    return func


def exp(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}
    return func


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=False)
    
    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score')) #, edges)
        
        # scaling
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))
        
        # Use available edge features to modify the scores
        g.apply_edges(imp_exp_attn('score', 'proj_e'))
        
        # Copy edge features as e_out to be passed to FFN_e
        g.apply_edges(out_edge_features('score'))
        
        # softmax
        g.apply_edges(exp('score'))

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))
    
    def forward(self, g, h, e):
        
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        proj_e = self.proj_e(e)
        
        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        g.edata['proj_e'] = proj_e.view(-1, self.num_heads, self.out_dim)
        
        self.propagate_attention(g)
        
        h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6)) # adding eps to all values here
        e_out = g.edata['e_out']
        
        return h_out, e_out
    

class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False,
                 batch_norm=True, residual=True, use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm     
        self.batch_norm = batch_norm
        
        self.attention = MultiHeadAttentionLayer(in_dim, out_dim//num_heads, num_heads, use_bias)
        
        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim)
        
        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_h_layer2 = nn.Linear(out_dim*2, out_dim)
        
        # FFN for e
        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_e_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            self.layer_norm2_e = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            self.batch_norm2_e = nn.BatchNorm1d(out_dim)
        
    def forward(self, g, h, e):
        h_in1 = h # for first residual connection
        e_in1 = e # for first residual connection
        
        # multi-head attention out
        h_attn_out, e_attn_out = self.attention(g, h, e)
        
        h = h_attn_out.view(-1, self.out_channels)
        e = e_attn_out.view(-1, self.out_channels)
        
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        h = self.O_h(h)
        e = self.O_e(e)

        if self.residual:
            h = h_in1 + h # residual connection
            e = e_in1 + e # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            e = self.batch_norm1_e(e)

        h_in2 = h # for second residual connection
        e_in2 = e # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        # FFN for e
        e = self.FFN_e_layer1(e)
        e = F.relu(e)
        e = F.dropout(e, self.dropout, training=self.training)
        e = self.FFN_e_layer2(e)

        if self.residual:
            h = h_in2 + h # residual connection       
            e = e_in2 + e # residual connection  

        if self.layer_norm:
            h = self.layer_norm2_h(h)
            e = self.layer_norm2_e(e)

        if self.batch_norm:
            h = self.batch_norm2_h(h)
            e = self.batch_norm2_e(e)             

        return h, e
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)

class MLPReadout(nn.Module):
    def __init__(self, input_dim, output_dim, L=2):
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        # y = torch.sigmoid(y)
        return y


class GraphTransformer(nn.Module):

    def __init__(self,
                 hidden_size,
                 out_dim,
                 n_classes,
                 n_heads,
                 n_layers,
                 readout,
                 ndata_dim_in,
                 edata_dim_in,
                 device,
                 lap_pos_enc,
                 pos_enc_dim=None,
                 wl_pos_enc=False,
                 n_labels=21,
                 aa_embed_dim=None,
                 residual=True,
                 dropout=0.0,
                 in_feat_dropout=0.0,
                 layer_norm=True,
                 batch_norm=False,
                 use_weight_in_loss=False,
                 **kwargs):
        super().__init__()

        self.dropout = dropout
        self.n_layers = n_layers
        self.readout = readout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self.device = device
        self.lap_pos_enc = lap_pos_enc
        self.wl_pos_enc = wl_pos_enc
        # self.embed_graph = net_params['embed_graph']
        self.use_weight_in_loss = use_weight_in_loss
        max_wl_role_index = 100
        
        
        if self.lap_pos_enc:
            pos_enc_dim = pos_enc_dim
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_size)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_size)

        # self.onehot =
        self.aa_embedding = nn.Embedding(n_labels, aa_embed_dim)
        # self.center_encoder = nn.Linear(ndata_dim_in, hidden_size)
        # self.ndata_encoder = nn.Sequential(nn.Linear(aa_embed_dim + ndata_dim_in, hidden_size), nn.ReLU())
        # self.edata_encoder = nn.Sequential(nn.Linear(edata_dim_in, hidden_size), nn.ReLU())
        self.ndata_encoder = nn.Linear(aa_embed_dim + ndata_dim_in, hidden_size)
        self.edata_encoder = nn.Linear(edata_dim_in, hidden_size)
        # self.embedding_partner = nn.Linear(in_dim2, out_dim2)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_size, hidden_size, n_heads,
                                                           self.dropout, self.layer_norm, self.batch_norm, self.residual)
                                     for _ in range(self.n_layers-1)])
        self.layers.append(GraphTransformerLayer(hidden_size, out_dim, n_heads, self.dropout,
                                                 self.layer_norm, self.batch_norm, self.residual))
        
        self.SAP = nn.Sequential(nn.Linear(out_dim, 1), nn.Softmax(0))
        
        self.MLP_layer = MLPReadout(out_dim, n_classes)  # modified for no-partner
        self.predict = nn.Sigmoid()

    def forward(self, g, h_lap_pos_enc, alt_aa):
        # batch_n_nodes = g.batch_num_nodes()
        # target_indices = torch.cat([torch.tensor([0], device=self.device), batch_n_nodes])[:-1]

        # ref_aa_enc = F.one_hot(g.ndata['ref_aa'], num_classes=21)
        # alt_aa_enc = F.one_hot(alt_aa, num_classes = 21)

        # target_ref_aa = ref_aa_enc[target_indices]
        assert not g.ndata['feat'].detach().cpu().isnan().any()

        h_aa = self.aa_embedding(g.ndata['ref_aa'])  # n_nodes x hidden_dim
        g.ndata['h_aa'] = h_aa
        h_alt = self.aa_embedding(alt_aa)  # n_batch x hidden_dim

        seq_emb = []
        for b in range(g.batch_size):
            g_sub = dgl.slice_batch(g, b)
            # seq_emb.append(torch.matmul(g_sub.ndata['h_aa'], h_alt[b]))
            seq_emb.append(g_sub.ndata['h_aa'] * h_alt[b])
        h_aa = torch.cat(seq_emb)
        h = self.ndata_encoder(torch.cat([h_aa, g.ndata['feat']], dim=-1))
        # h = self.embedding_h(g.ndata['feat']) + h_aa
        # h = self.embedding_h(g.ndata['feat'])
        # h = torch.cat([h_seq.unsqueeze(1), h], dim=1)
        # g.edata['feat'] = torch.cat([g.edata['dist'], g.edata['angle']], dim=1)
        assert not g.edata['feat'].detach().cpu().isnan().any()
        e = self.edata_encoder(g.edata['feat'])
        assert not e.detach().cpu().isnan().any()
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc)
            h = h + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(g.ndata['wl_pos_enc'])
            h = h + h_wl_pos_enc
        h = self.in_feat_dropout(h)
        
        for conv in self.layers:
            h, e = conv(g, h, e)

        g.ndata['h'] = h
        hg = dgl.readout_nodes(g, 'h', op=self.readout)  # graph embedding: batch_size x embedding_dim

        hg_out = self.MLP_layer(hg)  # check hg dimension,
        return hg_out

    
    def loss(self, logits, label):  # TODO: modify for top-k recommendation
        # scores.topk(self.topk)
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

