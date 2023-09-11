import os, sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm

import torch 
import torch.nn.functional as F

import dgl
from torch_geometric.data import Data
from torch_geometric.data import Dataset

from .dataset_base import GraphDataSetBase
from dev.preprocess.utils import *
from .data_utils import *


class ProNetDataset(GraphDataSetBase):
    def __init__(self, 
                 df_in, 
                 feat_dir, 
                 cov_thres=0.5, 
                 use_cosmis=False, 
                 cosmis_dir=None, 
                 seq2struct_dict=None,
                 norm_feat=False, 
                 struct_graph_cache=None, 
                 cosmis_cols=['cosmis'], 
                 cosmis_suffix='.pkl', 
                 use_oe=False, 
                 oe_dir=None, 
                 use_dssp=False, 
                 dssp_dir=None, 
                 sift_map=None, 
                 use_ires=False, 
                 ires_pred_dir=None, 
                 ires_gt_file=None, 
                 **kwargs):
        # PyG dataset that loads pre-computed DGL graphs and convert into PyG Data objects

        super(ProNetDataset, self).__init__()
        
        self.feat_root = Path(feat_dir)
        self.use_cosmis = use_cosmis
        self.cosmis_dir = cosmis_dir
        self.use_oe = use_oe
        self.oe_dir = oe_dir
        self.use_dssp = use_dssp
        self.dssp_dir = dssp_dir
        self.use_ires = use_ires
        if isinstance(ires_pred_dir, str):
            ires_pred_dir = Path(ires_pred_dir)
        self.ires_pred_dir = ires_pred_dir
        if self.use_ires:
            try:
                self.ires_gt_dict = pd.read_pickle(ires_gt_file)
            except FileNotFoundError:
                self.ires_gt_dict = dict()

        if not seq2struct_dict:
            seq2struct_dict = dict()
        self.seq2struct_dict = seq2struct_dict
        self.sift_map = sift_map
        self.seq_graph_stats = {'nodes': [], 'edges': []}
        self.struct_graph_stats = {'nodes': [], 'edges': []}
        
        self.struct_graph_root = Path(struct_graph_cache)

        self.cov_thres = cov_thres

        self.load_cache_data(df_in, cosmis_cols, cosmis_suffix)        
    
    def load_cache_data(self, df_in, cosmis_cols=['cosmis'], cosmis_suffix='.pkl'):

        for i, record in tqdm(df_in.iterrows(), total=df_in.shape[0]):
            var_data = Data()
            nfeat_list = []
            uprot = record['UniProt']
            uprot_pos = record['Protein_position']
            var_id = record['prot_var_id']

            # load structural graph
            if record['PDB_coverage'] >= self.cov_thres:
                model = 'PDB'
                struct_id = record['PDB']
                chain = record['Chain']
                key = ':'.join([uprot, struct_id, chain])
                if key not in self.seq2struct_dict:
                    struct_info = self.sift_map.query('UniProt == @uprot & PDB == @struct_id & Chain == @chain').iloc[0]
                    seq_pos = list(map(int, unzip_res_range(struct_info['MappableResInPDBChainOnUniprotBasis'])))
                    struct_pos = unzip_res_range(struct_info['MappableResInPDBChainOnPDBBasis'])
                    self.seq2struct_dict[key] = dict(zip(seq_pos, struct_pos))
                f_struct_graph = self.struct_graph_root / f'{model}-{struct_id}_{uprot}_{uprot_pos}.pkl'

            else:
                model = 'AF'
                struct_id = uprot
                chain = 'A'
                key = ':'.join([uprot, model, chain])
                seq_pos = list(range(1, record['prot_length'] + 1))
                struct_pos = list(map(str, seq_pos))
                self.seq2struct_dict[key] = dict(zip(seq_pos, struct_pos))
                f_struct_graph = self.struct_graph_root / f'{model}-{struct_id}_{uprot_pos}.pkl'

            seq2struct_pos = self.seq2struct_dict[key]

            if not f_struct_graph.exists():
                continue

            with open(f_struct_graph, 'rb') as f_pkl:
                # pos_remain_str: sequential position of remained residues in structural graph
                struct_graph, pos_remain_str, var_idx = pickle.load(f_pkl)
                nfeat_list.append(struct_graph.ndata['feat'])

            if self.use_cosmis:
                try:
                    cosmis_feat_raw = load_cosmis_feats(uprot, self.cosmis_dir, cols=cosmis_cols, suffix=cosmis_suffix)
                    if np.isnan(cosmis_feat_raw).all():
                        logging.warning("NA in COSMIS for {}".format(uprot))
                        continue

                    cosmis_stats = {'mean': torch.tensor(np.nanmean(cosmis_feat_raw, axis=0)),
                                    'min': torch.tensor(np.nanmin(cosmis_feat_raw, axis=0)),
                                    'max': torch.tensor(np.nanmax(cosmis_feat_raw, axis=0))}

                    cosmis_feat, cosmis_stats = impute_nan(torch.tensor(cosmis_feat_raw[list(map(lambda x: x - 1, pos_remain_str)), :]), cosmis_stats)
                    nfeat_list.append(cosmis_feat)

                except Exception as e:
                    logging.warning(f'Exception {e} in loading COSMIS feature for {uprot}:{uprot_pos}')
                    continue
            
            if self.use_oe:
                try:
                    oe_feat_raw = load_oe_feats(uprot, self.oe_dir, cols=['obs_exp_mean', 'obs_exp_max'])
                    if np.isnan(oe_feat_raw).all():
                        logging.warning("NA in OE feature for {}".format(uprot))
                        continue

                    oe_stats = {'mean': torch.tensor(np.nanmean(oe_feat_raw, axis=0)),
                                'min': torch.tensor(np.nanmin(oe_feat_raw, axis=0)),
                                'max': torch.tensor(np.nanmax(oe_feat_raw, axis=0))}

                    oe_feat, oe_stats = impute_nan(torch.tensor(oe_feat_raw[list(map(lambda x: x - 1, pos_remain_str)), :]), oe_stats)
                    nfeat_list.append(oe_feat)

                except (FileNotFoundError, ValueError, IndexError) as e:
                    logging.warning(f'{e} in loading OE feature for {uprot}')
                    continue
            
            if self.use_dssp:
                try:
                    ss8, dssp_feat_raw = load_dssp_feats(struct_id, pos_remain_str, chain, model, seq2struct_pos, self.dssp_dir)
                    # struct_graph.ndata['dssp'] = torch.tensor(dssp_feat_raw)
                    nfeat_list.append(torch.tensor(dssp_feat_raw))
                    var_data.ss = torch.tensor(list(map(ss8_to_index, ss8))).unsqueeze(-1)
                    # TODO: encode ss8
                except Exception as e:
                    logging.warning(f'{e} in loading DSSP feature for {var_id} struct-id={struct_id}')
                    continue
            
            if self.use_ires:
                try:
                    ires_feat_raw = load_ires_feats(uprot, record['prot_length'], self.ires_pred_dir, self.ires_gt_dict)
                    ires_feat = torch.tensor(ires_feat_raw[list(map(lambda x: x - 1, pos_remain_str))]).unsqueeze(-1)
                except Exception as e:
                    logging.warning(f'{e} in loading IRES feature for {uprot}')
                    ires_feat = torch.zeros((struct_graph.num_nodes(), 1))
                nfeat_list.append(ires_feat)
            
            var_data.id = var_id
            var_data.x = torch.cat(nfeat_list, dim=-1)  # concatenated features
            var_data.y = record['label']  # label
            var_data.coords = struct_graph.ndata['coords']
            
            var_data.alt_aa = aa_to_index(protein_letters_1to3_extended[record['ALT_AA']].upper())
            var_data.ref_seq = struct_graph.ndata['ref_aa']
            var_data.pos_remain = pos_remain_str
            var_data.var_idx = var_idx

            var_data.edge_index = torch.stack(struct_graph.edges(), dim=0) # convert dgl edges to pyg edge_index
            var_data.edge_attr = struct_graph.edata['coev'] # edge features

            self.data.append(var_data)
            self.label.append(var_data.y)
            self.n_nodes.append(var_data.num_nodes)
            self.n_edges.append(var_data.num_edges)
    
    def __getitem__(self, index):
        return self.data[index]

    def get_ndata_dim(self, feat_name='feat'):
        return self.data[0].x.shape[1]

    def get_edata_dim(self, feat_name='feat'):
        return self.data[0].edge_attr.shape[1]