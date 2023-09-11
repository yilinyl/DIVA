"""
This is an implementation of ProNet model (from official release of DIG package)

"""

from torch_geometric.nn import inits, MessagePassing
from torch_geometric.nn import radius_graph

from .features import d_angle_emb, d_theta_phi_emb

from torch_scatter import scatter
from torch_sparse import matmul

import torch
from torch import nn
from torch.nn import Embedding
import torch.nn.functional as F

import numpy as np


num_aa_type = 21
num_side_chain_embs = 8
num_bb_embs = 6

def swish(x):
    return x * torch.sigmoid(x)


class Linear(torch.nn.Module):
    """
        A linear method encapsulation similar to PyG's

        Parameters
        ----------
        in_channels (int)
        out_channels (int)
        bias (int)
        weight_initializer (string): (glorot or zeros)
    """

    def __init__(self, in_channels, out_channels, bias=True, weight_initializer='glorot'):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight_initializer == 'glorot':
            inits.glorot(self.weight)
        elif self.weight_initializer == 'zeros':
            inits.zeros(self.weight)
        if self.bias is not None:
            inits.zeros(self.bias)

    def forward(self, x):
        """"""
        return F.linear(x, self.weight, self.bias)


class TwoLinear(torch.nn.Module):
    """
        A layer with two linear modules

        Parameters
        ----------
        in_channels (int)
        middle_channels (int)
        out_channels (int)
        bias (bool)
        act (bool)
    """

    def __init__(
            self,
            in_channels,
            middle_channels,
            out_channels,
            bias=False,
            act=False
    ):
        super(TwoLinear, self).__init__()
        self.lin1 = Linear(in_channels, middle_channels, bias=bias)
        self.lin2 = Linear(middle_channels, out_channels, bias=bias)
        self.act = act

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        x = self.lin1(x)
        if self.act:
            x = swish(x)
        x = self.lin2(x)
        if self.act:
            x = swish(x)
        return x


class EdgeGraphConv(MessagePassing):
    """
        Graph convolution similar to PyG's GraphConv(https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GraphConv)

        The difference is that this module performs Hadamard product between node feature and edge feature

        Parameters
        ----------
        in_channels (int)
        out_channels (int)
    """
    def __init__(self, in_channels, out_channels):
        super(EdgeGraphConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_l = Linear(in_channels, out_channels)
        self.lin_r = Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, edge_weight, size=None):
        x = (x, x)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        out = self.lin_l(out)
        return out + self.lin_r(x[1])

    def message(self, x_j, edge_weight):
        return edge_weight * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x[0], reduce=self.aggr)


class InteractionBlock(torch.nn.Module):
    def __init__(
            self,
            hidden_channels,
            output_channels,
            num_radial,
            num_spherical,
            num_layers,
            mid_emb,
            act=swish,
            num_pos_emb=16,
            dropout=0,
            level='allatom'
    ):
        super(InteractionBlock, self).__init__()
        self.act = act
        self.dropout = nn.Dropout(dropout)
        
        # self.conv0 = EdgeGraphConv(hidden_channels, hidden_channels)
        self.conv1 = EdgeGraphConv(hidden_channels, hidden_channels)
        self.conv2 = EdgeGraphConv(hidden_channels, hidden_channels)

        # self.lin_feature0 = TwoLinear(num_radial * num_spherical ** 2, mid_emb, hidden_channels)
        if level == 'aminoacid':
            self.lin_feature1 = TwoLinear(num_radial * num_spherical, mid_emb, hidden_channels)
        elif level == 'backbone' or level == 'allatom':
            self.lin_feature1 = TwoLinear(3 * num_radial * num_spherical, mid_emb, hidden_channels)
        self.lin_feature2 = TwoLinear(num_pos_emb, mid_emb, hidden_channels)

        self.lin_1 = Linear(hidden_channels, hidden_channels)
        self.lin_2 = Linear(hidden_channels, hidden_channels)

        # self.lin0 = Linear(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.lins_cat = torch.nn.ModuleList()
        self.lins_cat.append(Linear(2*hidden_channels, hidden_channels))  # YL: 3 --> 2 * hidden channels (temporarily remove feature0)
        for _ in range(num_layers-1):
            self.lins_cat.append(Linear(hidden_channels, hidden_channels))

        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers-1):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.final = Linear(hidden_channels, output_channels)

        self.reset_parameters()

    def reset_parameters(self):
        # self.conv0.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

        # self.lin_feature0.reset_parameters()
        self.lin_feature1.reset_parameters()
        self.lin_feature2.reset_parameters()

        self.lin_1.reset_parameters()
        self.lin_2.reset_parameters()

        # self.lin0.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        for lin in self.lins:
            lin.reset_parameters()
        for lin in self.lins_cat:
            lin.reset_parameters()

        self.final.reset_parameters()


    def forward(self, x, feature1, pos_emb, edge_index, batch):  # YL: temoporariliy remove feature0
        x_lin_1 = self.act(self.lin_1(x))
        x_lin_2 = self.act(self.lin_2(x))
        
        # feature0 = self.lin_feature0(feature0)  # TODO: concatenate original edge features
        # h0 = self.conv0(x_lin_1, edge_index, feature0)
        # h0 = self.lin0(h0)
        # h0 = self.act(h0)
        # h0 = self.dropout(h0)

        feature1 = self.lin_feature1(feature1)
        h1 = self.conv1(x_lin_1, edge_index, feature1)
        h1 = self.lin1(h1)
        h1 = self.act(h1)
        h1 = self.dropout(h1)

        feature2 = self.lin_feature2(pos_emb)
        h2 = self.conv2(x_lin_1, edge_index, feature2)
        h2 = self.lin2(h2)
        h2 = self.act(h2)
        h2 = self.dropout(h2)

        h = torch.cat((h1, h2),1)
        for lin in self.lins_cat:
            h = self.act(lin(h)) 

        h = h + x_lin_2

        for lin in self.lins:
            h = self.act(lin(h)) 
        h = self.final(h)
        return h


class ProNet(nn.Module):
    r"""
         The ProNet from the "Learning Protein Representations via Complete 3D Graph Networks" paper.
        
        Args:
            level: (str, optional): The level of protein representations. It could be :obj:`aminoacid`, obj:`backbone`, and :obj:`allatom`. (default: :obj:`aminoacid`)
            num_blocks (int, optional): Number of building blocks. (default: :obj:`4`)
            hidden_channels (int, optional): Hidden embedding size. (default: :obj:`128`)
            out_channels (int, optional): Size of each output sample. (default: :obj:`1`)
            mid_emb (int, optional): Embedding size used for geometric features. (default: :obj:`64`)
            num_radial (int, optional): Number of radial basis functions. (default: :obj:`6`)
            num_spherical (int, optional): Number of spherical harmonics. (default: :obj:`2`)
            cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`10.0`)
            max_num_neighbors (int, optional): Max number of neighbors during graph construction. (default: :obj:`32`)
            int_emb_layers (int, optional): Number of embedding layers in the interaction block. (default: :obj:`3`)
            out_layers (int, optional): Number of layers for features after interaction blocks. (default: :obj:`2`)
            num_pos_emb (int, optional): Number of positional embeddings. (default: :obj:`16`)
            dropout (float, optional): Dropout. (default: :obj:`0`)
            data_augment_eachlayer (bool, optional): Data augmentation tricks. If set to :obj:`True`, will add noise to the node features before each interaction block. (default: :obj:`False`)
            euler_noise (bool, optional): Data augmentation tricks. If set to :obj:`True`, will add noise to Euler angles. (default: :obj:`False`)
            
    """
    def __init__(
            self,
            ndata_dim_in,  # dimension of raw node features
            level='aminoacid',
            num_blocks=4,
            aa_embed_dim=64,
            hidden_channels=128,
            out_channels=1,
            mid_emb=64,
            num_radial=6,
            num_spherical=2,
            cutoff=10.0,
            max_num_neighbors=32,
            int_emb_layers=3,
            out_layers=2,
            num_pos_emb=16,
            dropout=0,
            data_augment_eachlayer=False,
            euler_noise = False, 
            aa_only=False,
            **kwargs
    ):
        super(ProNet, self).__init__()
        self.cutoff = cutoff
        # self.max_num_neighbors = max_num_neighbors
        self.num_pos_emb = num_pos_emb
        self.data_augment_eachlayer = data_augment_eachlayer
        self.euler_noise = euler_noise
        self.level = level
        self.act = swish
        self.aa_only = aa_only

        # self.feature0 = d_theta_phi_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)
        self.feature1 = d_angle_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)
        
        self.aa_encoder = nn.Embedding(num_aa_type, aa_embed_dim)

        in_channels = ndata_dim_in + aa_embed_dim
        if level == 'backbone':
            in_channels += num_bb_embs
        elif level == 'allatom':
            in_channels += num_side_chain_embs
        
        self.ndata_encoder = nn.Linear(in_channels, hidden_channels)
        self.layernorm = nn.LayerNorm(hidden_channels)
        # if level == 'aminoacid':
        #     self.embedding = Embedding(num_aa_type, hidden_channels)
        # elif level == 'backbone':
        #     self.embedding = torch.nn.Linear(num_aa_type + num_bb_embs, hidden_channels)
        # elif level == 'allatom':
        #     self.embedding = torch.nn.Linear(num_aa_type + num_bb_embs + num_side_chain_embs, hidden_channels)
        # else:
        #     print('No supported model!')

        self.interaction_blocks = torch.nn.ModuleList(
            [
                InteractionBlock(
                    hidden_channels=hidden_channels,
                    output_channels=hidden_channels,
                    num_radial=num_radial,
                    num_spherical=num_spherical,
                    num_layers=int_emb_layers,
                    mid_emb=mid_emb,
                    act=self.act,
                    num_pos_emb=num_pos_emb,
                    dropout=dropout,
                    level=level
                )
                for _ in range(num_blocks)
            ]
        )

        self.variant_layer = nn.Linear(hidden_channels + aa_embed_dim, hidden_channels)

        # self.lins_out = torch.nn.ModuleList()
        # for _ in range(out_layers-1):
        #     self.lins_out.append(Linear(hidden_channels, hidden_channels))

        self.lin_out = Linear(hidden_channels, out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        self.aa_encoder.reset_parameters()
        self.ndata_encoder.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()
        # for lin in self.lins_out:
        #     lin.reset_parameters()
        self.variant_layer.reset_parameters()
        self.lin_out.reset_parameters()

    def pos_emb(self, edge_index, num_pos_emb=16):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        d = edge_index[0] - edge_index[1]
     
        frequency = torch.exp(
            torch.arange(0, num_pos_emb, 2, dtype=torch.float64, device=edge_index.device)  # YL: modified dtype to float64
            * -(np.log(10000.0) / num_pos_emb)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def forward(self, batch_data):

        ref_seq, alt_aa, pos, batch = batch_data.ref_seq, batch_data.alt_aa, batch_data.coords, batch_data.batch

        device = ref_seq.device
        # Initialize node features
        # TODO: add alt_aa information
        # TODO: add ss embedding
        
        # aa_enc = torch.squeeze(F.one_hot(ref_seq, num_classes=num_aa_type).to(torch.float64))
        ref_aa_emb = self.aa_encoder(ref_seq)
        alt_aa_emb = self.aa_encoder(alt_aa)

        if self.level == 'aminoacid':
            # x_raw = torch.cat([aa_enc, batch_data.x], dim=1)
            x_raw = batch_data.x
        elif self.level == 'backbone':
            bb_embs = batch_data.bb_embs
            x_raw = torch.cat([bb_embs, batch_data.x], dim = 1)
        elif self.level == 'allatom':
            bb_embs = batch_data.bb_embs
            side_chain_embs = batch_data.side_chain_embs
            x_raw = torch.cat([bb_embs, side_chain_embs, batch_data.x], dim = 1)
        else:
            print('No supported model!')
        
        assert not x_raw.detach().cpu().isnan().any()
        x = self.ndata_encoder(torch.cat([ref_aa_emb, x_raw], -1)) # YL: concatenate initial embeddings with raw features
        x = self.layernorm(x)
        # edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors)
        edge_index = batch_data.edge_index
        pos_emb = self.pos_emb(edge_index, self.num_pos_emb)
        j, i = edge_index

        # Calculate distances.
        dist = (pos[i] - pos[j]).norm(dim=1)
        
        num_nodes = len(ref_seq)

        # Calculate angles theta and phi.
        refi0 = (i-1) % num_nodes
        refi1 = (i+1) % num_nodes

        a = ((pos[j] - pos[i]) * (pos[refi0] - pos[i])).sum(dim=-1)
        b = torch.cross(pos[j] - pos[i], pos[refi0] - pos[i]).norm(dim=-1)
        theta = torch.atan2(b, a)

        plane1 = torch.cross(pos[refi0] - pos[i], pos[refi1] - pos[i])
        plane2 = torch.cross(pos[refi0] - pos[i], pos[j] - pos[i])
        a = (plane1 * plane2).sum(dim=-1)
        b = (torch.cross(plane1, plane2) * (pos[refi0] - pos[i])).sum(dim=-1) / ((pos[refi0] - pos[i]).norm(dim=-1))
        phi = torch.atan2(b, a)

        # feature0 = self.feature0(dist, theta, phi)

        if self.level == 'backbone' or self.level == 'allatom':
            pos_n = batch_data.coords_n
            pos_c = batch_data.coords_c
        
            # Calculate Euler angles.
            Or1_x = pos_n[i] - pos[i]
            Or1_z = torch.cross(Or1_x, torch.cross(Or1_x, pos_c[i] - pos[i]))
            Or1_z_length = Or1_z.norm(dim=1) + 1e-7
            
            Or2_x = pos_n[j] - pos[j]
            Or2_z = torch.cross(Or2_x, torch.cross(Or2_x, pos_c[j] - pos[j]))
            Or2_z_length = Or2_z.norm(dim=1) + 1e-7

            Or1_Or2_N = torch.cross(Or1_z, Or2_z)
            
            angle1 = torch.atan2((torch.cross(Or1_x, Or1_Or2_N) * Or1_z).sum(dim=-1)/Or1_z_length, (Or1_x * Or1_Or2_N).sum(dim=-1))
            angle2 = torch.atan2(torch.cross(Or1_z, Or2_z).norm(dim=-1), (Or1_z * Or2_z).sum(dim=-1))
            angle3 = torch.atan2((torch.cross(Or1_Or2_N, Or2_x) * Or2_z).sum(dim=-1)/Or2_z_length, (Or1_Or2_N * Or2_x).sum(dim=-1))

            if self.euler_noise:
                euler_noise = torch.clip(torch.empty(3,len(angle1)).to(device).normal_(mean=0.0, std=0.025), min=-0.1, max=0.1)
                angle1 += euler_noise[0]
                angle2 += euler_noise[1]
                angle3 += euler_noise[2]

            feature1 = torch.cat((self.feature1(dist, angle1), self.feature1(dist, angle2), self.feature1(dist, angle3)),1)

        elif self.level == 'aminoacid':
            refi = (i-1)%num_nodes

            refj0 = (j-1)%num_nodes
            refj = (j-1)%num_nodes
            refj1 = (j+1)%num_nodes

            mask = refi0 == j
            refi[mask] = refi1[mask]
            mask = refj0 == i
            refj[mask] = refj1[mask]

            plane1 = torch.cross(pos[j] - pos[i], pos[refi] - pos[i])
            plane2 = torch.cross(pos[j] - pos[i], pos[refj] - pos[j])
            a = (plane1 * plane2).sum(dim=-1) 
            b = (torch.cross(plane1, plane2) * (pos[j] - pos[i])).sum(dim=-1) / dist
            tau = torch.atan2(b, a)

            feature1 = self.feature1(dist, tau)

        # Interaction blocks.
        for interaction_block in self.interaction_blocks:
            if self.data_augment_eachlayer:
                # add gaussian noise to features
                gaussian_noise = torch.clip(torch.empty(x.shape).to(device).normal_(mean=0.0, std=0.025), min=-0.1, max=0.1)
                x += gaussian_noise
            x = interaction_block(x, feature1, pos_emb, edge_index, batch)  # YL: temporarily remove feature0

        y = scatter(x, batch, dim=0)

        y = self.relu(self.variant_layer(torch.cat([y, alt_aa_emb], -1)))  # graph-level feature processing (ref & alt)

        # for lin in self.lins_out:
        #     y = self.relu(lin(y))
        #     y = self.dropout(y)        
        y = self.lin_out(y)
        return y

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
    

