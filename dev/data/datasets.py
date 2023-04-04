import dgl
import numpy as np
import torch
import logging
from tqdm import *
from torch.utils.data import Dataset
from dev.preprocess.utils import *
from .data_utils import *


class GraphData(object):
    def __init__(self, graph, label, target, var_id=None):
        self.graph = graph
        self.label = label
        self.target = target
        self.var_id = var_id


class GraphDataSetBase(Dataset):
    def __init__(self):
        super(GraphDataSetBase, self).__init__()
        self.data = []
        self.label = []
        self.n_nodes = []
        self.n_edges = []

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

    def dataset_summary(self):
        return np.mean(self.n_nodes), np.mean(self.n_edges)


class VariantGraphDataSet(GraphDataSetBase):
    def __init__(self, df_in, graph_cache, pdb_root_dir, af_root_dir, feat_dir, sift_map, lap_pos_enc=True, wl_pos_enc=False, pos_enc_dim=None,
                 cov_thres=0.5, num_neighbors=10, distance_type='centroid', method='radius', radius=10, df_ires=None, save=False,
                 anno_ires=False, coord_option=None, feat_stats=None, var_db=None, seq2struct_all=None,
                 var_graph_radius=None, seq_dict=None, window_size=None, graph_type='hetero', norm_feat=True,
                 save_var_graph=False, var_graph_cache='./var_graph_cache', nsp_dir='', **kwargs):
        super(VariantGraphDataSet, self).__init__()

        # self.n_patho = []
        # self.aa_idx = []
        # self.aa_mask = []
        self.lap_pos_enc = lap_pos_enc
        self.wl_pos_enc = wl_pos_enc
        self.pos_enc_dim = pos_enc_dim
        if not seq2struct_all:
            seq2struct_all = dict()
        self.seq2struct_dict = seq2struct_all
        self.graph_type = graph_type
        self.window_size = window_size
        if not seq_dict:
            seq_dict = dict()
        self.seq_dict = seq_dict

        feat_root = Path(feat_dir)
        graph_cache_path = Path(graph_cache)
        if not graph_cache_path.exists():
            graph_cache_path.mkdir(parents=True)

        var_graph_path = Path(var_graph_cache) / self.graph_type
        if save_var_graph:
            if not var_graph_path.exists():
                var_graph_path.mkdir(parents=True)

        self.var_db = var_db
        if isinstance(var_db, type(None)):
            var_db = df_in
            self.var_db = var_db.groupby(['UniProt', 'Protein_position'])['label'].any().reset_index()
            self.var_db = self.var_db.rename(columns={'label': 'any_patho'}).query('any_patho == True').reset_index(drop=True)
        uprot_prev = ''
        model_prev = ''
        struct_prev = ''

        for i, record in tqdm(df_in.iterrows(), total=df_in.shape[0]):
            uprot = record['UniProt']
            uprot_pos = record['Protein_position']
            if uprot not in self.seq_dict:
                self.seq_dict[uprot] = fetch_prot_seq(uprot)
            seq = self.seq_dict[uprot]
            if record['PDB_coverage'] >= cov_thres:
                model = 'PDB'
                struct_id = record['PDB']
                chain = record['Chain']
                key = ':'.join([uprot, struct_id, chain])

                if key not in self.seq2struct_dict:
                    struct_info = sift_map.query('UniProt == @uprot & PDB == @struct_id & Chain == @chain').iloc[0]
                    seq_pos = list(map(int, unzip_res_range(struct_info['MappableResInPDBChainOnUniprotBasis'])))
                    struct_pos = unzip_res_range(struct_info['MappableResInPDBChainOnPDBBasis'])
                    self.seq2struct_dict[key] = dict(zip(seq_pos, struct_pos))

                seq2struct_pos = self.seq2struct_dict[key]
                if max(seq2struct_pos.keys()) > len(seq):
                    logging.warning('Inconsistent sequence length for {}'.format(uprot))
                    continue

            else:
                model = 'AF'
                struct_id = uprot
                chain = 'A'
                key = ':'.join([uprot, model, chain])
                seq_pos = list(range(1, record['prot_length']+1))
                struct_pos = list(map(str, seq_pos))
                self.seq2struct_dict[key] = dict(zip(seq_pos, struct_pos))
                seq2struct_pos = self.seq2struct_dict[key]

            f_graph = graph_cache_path / '{}_{}_graph.pkl'.format(model, struct_id)
            f_var_graph = var_graph_path / f'{model}-{struct_id}_{uprot_pos}.pkl'
            # f_graph = os.path.join(g_data_dir, '{}_{}_graph.pkl'.format(model, struct_id))
            if model != model_prev or struct_id != struct_prev:
                if f_graph.exists():
                    with open(f_graph, 'rb') as f_pkl:
                        prot_graph, chain_res_list = pickle.load(f_pkl)
                else:
                    new_graph = build_struct_graph(record, model, pdb_root_dir, af_root_dir, graph_cache,
                                                   num_neighbors, distance_type, method, radius, df_ires,
                                                   save, anno_ires, coord_option)

                    if not new_graph:  # fail to build protein graph
                        continue
                    prot_graph, chain_res_list = new_graph

            angle_stats = {'mean': torch.nanmean(prot_graph.edata['angle']),
                           'min': torch.tensor(np.nanmin(prot_graph.edata['angle'].numpy())),
                           'max': torch.tensor(np.nanmax(prot_graph.edata['angle'].numpy()))}

            dist_stats = {'mean': torch.nanmean(prot_graph.edata['dist']),
                          'min': torch.tensor(np.nanmin(prot_graph.edata['dist'].numpy())),
                          'max': torch.tensor(np.nanmax(prot_graph.edata['dist'].numpy()))}

            # feat_version = record['feat_version']
            # feat_path = feat_root / feat_version / 'sequence_features'
            # feat_data = load_pio_features(uprot, feat_path)
            # feat_data = normalize_data(feat_data, feat_stats)
            if uprot != uprot_prev:
                try:
                    bio_feats = load_expasy_feats(uprot, feat_root, '3dvip_expasy')
                    pssm_feats = load_pssm(uprot, feat_root, '3dvip_pssm')
                    coev_feat_df = load_coev_feats(uprot, feat_root, '3dvip_sca', '3dvip_dca')
                    nsp_feat = load_nsp_feats(uprot, nsp_dir, exclude=['asa', 'phi', 'psi', 'disorder'])

                    feat_data = np.hstack([bio_feats, pssm_feats, nsp_feat])
                except (FileNotFoundError, ValueError) as e:
                    logging.warning(f'{e} in loading feature for {uprot}')
                    continue

            feat_stats = {'mean': torch.tensor(np.nanmean(feat_data, axis=0)),
                          'min': torch.tensor(np.nanmin(feat_data, axis=0)),
                          'max': torch.tensor(np.nanmax(feat_data, axis=0))}

            if f_var_graph.exists():
                with open(f_var_graph, 'rb') as f_pkl:
                    var_graph, seq_pos_remain, var_idx = pickle.load(f_pkl)
                struct_etype = '_E' if self.graph_type == 'struct' else 'struct'

            else:
                if self.graph_type == 'struct':
                    # Extract structure-based variant graph
                    var_graph, seq_pos_remain = extract_variant_graph(uprot_pos, chain, chain_res_list, seq2struct_pos,
                                                                      prot_graph, feat_data, feat_stats, coev_feat_df,
                                                                      patch_radius=var_graph_radius, normalize=norm_feat)
                    struct_etype = '_E'
                    var_idx = 0
                else:

                    seq_array = np.array(
                        list(map(lambda x: aa_to_index(protein_letters_1to3_extended[x].upper()), list(seq))))

                    seq_src, seq_dst, start_idx, end_idx = self.add_seq_edges(uprot, uprot_pos, len(seq))
                    seq_nodes = list(range(start_idx, end_idx+1))

                    g_sca = add_coev_edges(coev_feat_df, 'sca', nlargest=10)
                    g_dca = add_coev_edges(coev_feat_df, ['di', 'mi'], nlargest=20)
                    g_sca_sub = dgl.node_subgraph(g_sca, seq_nodes)
                    g_dca_sub = dgl.node_subgraph(g_dca, seq_nodes)

                    try:
                        struct_g = fetch_struct_edges(start_idx, end_idx, chain, prot_graph, chain_res_list, seq2struct_pos)
                    except ValueError:
                        logging.warning('Exception in building structural graph for {}-{}'.format(model, record['prot_var_id']))
                        continue
                    var_idx = uprot_pos - start_idx - 1
                    edge_dict = {
                        ('residue', 'seq', 'residue'): (seq_src, seq_dst),
                        ('residue', 'sca', 'residue'): g_sca_sub.edges(),
                        ('residue', 'dca', 'residue'): g_dca_sub.edges(),
                        ('residue', 'struct', 'residue'): struct_g.edges()
                    }

                    var_graph = dgl.heterograph(edge_dict)
                    angle_feat, angle_stats = impute_nan(struct_g.edata['angle'], angle_stats)
                    dist_feat, dist_stats = impute_nan(struct_g.edata['dist'], dist_stats)
                    aa_feat = torch.tensor(feat_data[start_idx: end_idx + 1])
                    if norm_feat:
                        var_graph.edges['struct'].data['angle'] = normalize_feat(angle_feat, angle_stats)
                        var_graph.edges['struct'].data['dist'] = normalize_feat(dist_feat, dist_stats)
                        aa_feat = normalize_feat(aa_feat, feat_stats)

                    var_graph.edges['sca'].data['sca'] = g_sca_sub.edata['sca'].unsqueeze(1)
                    var_graph.edges['dca'].data['dca'] = torch.cat([g_dca_sub.edata['di'].unsqueeze(1),
                                                                      g_dca_sub.edata['mi'].unsqueeze(1)], dim=-1)

                    var_graph.ndata['ref_aa'] = torch.tensor(seq_array[start_idx:(end_idx+1)], dtype=torch.int64)
                    var_graph.ndata['aa_feat'] = aa_feat

                    seq_pos_remain = list(range(start_idx+1, end_idx+2))  # convert index to 1-based sequential positions
                    struct_etype = 'struct'

            if var_graph.num_nodes() == 0:
                logging.warning('Empty graph for {}:{}'.format(uprot, uprot_pos))
                continue
            if self.graph_type == 'struct':
                struct_g = var_graph
            else:
                struct_g = dgl.edge_type_subgraph(var_graph, etypes=['struct'])

            # prot_patho_pos = self.var_db.query('UniProt == @uprot')
            # patho_tag = list(map(lambda x: x in prot_patho_pos['Protein_position'], seq_pos_remain))
            # self.n_patho.append(sum(patho_tag))
            # var_graph.ndata['patho_tag'] = torch.tensor(patho_tag, dtype=torch.float64).unsqueeze(1)

            # var_graph.edges[struct_etype].data['angle'] = impute_and_normalize(struct_g.edata['angle'], angle_stats,
            #                                                                    normalize=norm_feat)
            # var_graph.edges[struct_etype].data['dist'] = impute_and_normalize(struct_g.edata['dist'], dist_stats,
            #                                                                   normalize=norm_feat)
            # Operations on structual subgraph
            if struct_g.edata['angle'].isnan().any():
                logging.warning('NA in angle data for {}-{}'.format(model, record['prot_var_id']))
                continue
            if struct_g.edata['dist'].isnan().any():
                logging.warning('NA in dist data for {}-{}'.format(model, record['prot_var_id']))
                continue

            if not f_var_graph.exists() and save_var_graph:
                with open(f_var_graph, 'wb') as f_pkl:  # Update: also save var_idx
                    pickle.dump((var_graph, seq_pos_remain, var_idx), f_pkl)

            if self.lap_pos_enc:
                lap = laplacian_positional_encoding(struct_g, pos_enc_dim)
                if isinstance(lap, type(None)):
                    logging.warning('Laplacian position encoding not applicable for variant {}'.format(record['prot_var_id']))
                    # print('Variant graph: {} nodes'.format(var_graph.num_nodes()))
                    continue
                else:
                    var_graph.ndata['lap_pos_enc'] = lap
            if self.wl_pos_enc:
                var_graph.ndata['wl_pos_enc'] = wl_positional_encoding(struct_g)

            nfeat_all = list(var_graph.node_attr_schemes().keys())
            if '_ID' in nfeat_all:
                nfeat_all.pop(nfeat_all.index('_ID'))

            if 'ref_aa' in nfeat_all:
                nfeat_all.pop(nfeat_all.index('ref_aa'))  # process primary sequence separately
            if self.graph_type == 'hetero':
                nfeat_comb = list(map(lambda x: var_graph.ndata[x], nfeat_all))
                var_graph.ndata['feat'] = torch.cat(nfeat_comb, dim=-1)
            # else:
            #     var_graph.ndata['feat'] = torch.cat([var_graph.ndata['feat'], var_graph.ndata['patho_tag']], dim=-1)
            # ref_aa = aa_to_index(record['REF_AA'])
            alt_aa = aa_to_index(protein_letters_1to3_extended[record['ALT_AA']].upper())
            self.data.append((var_graph, record['label'], alt_aa, var_idx, record['prot_var_id']))
            self.label.append(record['label'])
            self.n_nodes.append(var_graph.num_nodes())
            self.n_edges.append(var_graph.num_edges())
            uprot_prev = uprot
            struct_prev = struct_id
            model_prev = model

    def get_var_db(self):
        return self.var_db

    def add_seq_edges(self, uprot, uprot_pos, prot_len, max_dist=1, inverse=True):
        w = self.window_size // 2
        if uprot not in self.seq_dict:
            self.seq_dict[uprot] = fetch_prot_seq(uprot)
        # seq = self.seq_dict[uprot]
        # seq_array = np.array(list(map(lambda x: aa_to_index(protein_letters_1to3_extended[x].upper()), list(seq))))
        start = max(uprot_pos - w - 1, 0)
        end = min(uprot_pos + w - 1, prot_len - 1)
        g_size = end - start + 1
        nodes = list(range(g_size))

        node_in = []
        node_out = []
        for d in range(1, max_dist+1):
            node_in.extend(nodes[:-d])
            node_out.extend(nodes[d:])
            # node_in = torch.arange(start, end+1-d)
            # node_out = torch.arange(start+d, end+1)
        if inverse:
            nodes_r = nodes[::-1]
            for d in range(1, max_dist+1):
                node_in.extend(nodes_r[:-d])
                node_out.extend(nodes_r[d:])
        return node_in, node_out, start, end

# def collate(samples):
#     # The input `samples` is a list of pairs
#     #  (graph, label).
#     graphs, labels = map(list, zip(*samples))
#     batched_graph = dgl.batch(graphs)
#     return batched_graph, torch.tensor(labels)


class VariantGraphCacheDataSet(GraphDataSetBase):
    def __init__(self, df_in, graph_cache, pdb_root_dir, af_root_dir, feat_dir, sift_map, lap_pos_enc=True, wl_pos_enc=False, pos_enc_dim=None,
                 cov_thres=0.5, var_db=None, seq2struct_all=None, seq_dict=None, window_size=None, graph_type='hetero', norm_feat=True,
                 save_var_graph=False, var_graph_cache='./var_graph_cache', use_nsp=False, nsp_dir='', use_cosmis=False, cosmis_dir='', **kwargs):
        super(VariantGraphCacheDataSet, self).__init__()

        self.data = []
        self.label = []
        self.n_nodes = []
        self.n_edges = []
        # self.n_patho = []
        # self.aa_idx = []
        # self.aa_mask = []
        self.lap_pos_enc = lap_pos_enc
        self.wl_pos_enc = wl_pos_enc
        self.pos_enc_dim = pos_enc_dim
        if not seq2struct_all:
            seq2struct_all = dict()
        self.seq2struct_dict = seq2struct_all
        self.graph_type = graph_type
        self.window_size = window_size
        if not seq_dict:
            seq_dict = dict()
        self.seq_dict = seq_dict

        feat_root = Path(feat_dir)
        graph_cache_path = Path(graph_cache)
        if not graph_cache_path.exists():
            graph_cache_path.mkdir(parents=True)

        var_graph_path = Path(var_graph_cache) / self.graph_type
        if save_var_graph:
            if not var_graph_path.exists():
                var_graph_path.mkdir(parents=True)

        self.var_db = var_db
        if isinstance(var_db, type(None)):
            var_db = df_in
            self.var_db = var_db.groupby(['UniProt', 'Protein_position'])['label'].any().reset_index()
            self.var_db = self.var_db.rename(columns={'label': 'any_patho'}).query('any_patho == True').reset_index(drop=True)

        for i, record in tqdm(df_in.iterrows(), total=df_in.shape[0]):
            uprot = record['UniProt']
            uprot_pos = record['Protein_position']
            # if uprot not in self.seq_dict:
            #     self.seq_dict[uprot] = fetch_prot_seq(uprot)
            # seq = self.seq_dict[uprot]
            if record['PDB_coverage'] >= cov_thres:
                model = 'PDB'
                struct_id = record['PDB']
                chain = record['Chain']
                key = ':'.join([uprot, struct_id, chain])

            else:
                model = 'AF'
                struct_id = uprot
                chain = 'A'
                key = ':'.join([uprot, model, chain])

            f_graph = graph_cache_path / '{}_{}_graph.pkl'.format(model, struct_id)
            if graph_type == 'seq':
                f_var_graph = var_graph_path / f'{uprot}_{uprot_pos}.pkl'
            else:
                f_var_graph = var_graph_path / f'{model}-{struct_id}_{uprot_pos}.pkl'
            # f_graph = os.path.join(g_data_dir, '{}_{}_graph.pkl'.format(model, struct_id))

            # try:
            #     with open(f_var_graph, 'rb') as f_pkl:
            #         cache_load = pickle.load(f_pkl)
            # except FileNotFoundError:
            #     continue
            if not f_var_graph.exists():
                continue

            with open(f_var_graph, 'rb') as f_pkl:
                var_graph, seq_pos_remain, var_idx = pickle.load(f_pkl)

            if self.graph_type == 'struct':
                struct_g = var_graph
                struct_etype = '_E'
            else:
                struct_g = dgl.edge_type_subgraph(var_graph, etypes=['struct'])
                struct_etype = 'struct'

            if var_graph.num_nodes() == 0:
                logging.warning('Empty graph for {}:{}'.format(uprot, uprot_pos))
                continue

            # Operations on structual subgraph
            if struct_g.edata['angle'].isnan().any():
                logging.warning('NA in angle data for {}-{}'.format(model, record['prot_var_id']))
                continue
            if struct_g.edata['dist'].isnan().any():
                logging.warning('NA in dist data for {}-{}'.format(model, record['prot_var_id']))
                continue
            if use_nsp:
                try:
                    nsp_feat_raw = load_nsp_feats(uprot, nsp_dir, exclude=['asa', 'phi', 'psi', 'disorder'])
                    nsp_stats = {'mean': torch.tensor(np.nanmean(nsp_feat_raw, axis=0)),
                                  'min': torch.tensor(np.nanmin(nsp_feat_raw, axis=0)),
                                  'max': torch.tensor(np.nanmax(nsp_feat_raw, axis=0))}
                    nsp_feat, nsp_stats = impute_nan(
                        torch.tensor(nsp_feat_raw[list(map(lambda x: x - 1, seq_pos_remain)), :]), nsp_stats)
                    var_graph.ndata['nps_feat'] = nsp_feat
                except (FileNotFoundError, ValueError) as e:
                    logging.warning(f'{e} in loading feature for {uprot}')
                    continue
            if use_cosmis:
                try:
                    cosmis_feat = load_cosmis_feats(uprot, cosmis_dir, cols=['cosmis', 'cosmis_pvalue'])
                    var_graph.ndata['cosmis'] = torch.tensor(cosmis_feat[list(map(lambda x: x - 1, seq_pos_remain)), :])
                except FileNotFoundError:
                    continue

            if self.lap_pos_enc:
                lap = laplacian_positional_encoding(struct_g, pos_enc_dim)
                if isinstance(lap, type(None)):
                    logging.warning('Laplacian position encoding not applicable for variant {}'.format(record['prot_var_id']))
                    # print('Variant graph: {} nodes'.format(var_graph.num_nodes()))
                    continue
                else:
                    var_graph.ndata['lap_pos_enc'] = lap
            if self.wl_pos_enc:
                var_graph.ndata['wl_pos_enc'] = wl_positional_encoding(struct_g)
            # if 'patho_tag' not in var_graph.node_attr_schemes().keys():
            #     prot_patho_pos = self.var_db.query('UniProt == @uprot')
            #     patho_tag = list(map(lambda x: x in prot_patho_pos['Protein_position'], seq_pos_remain))
            #     var_graph.ndata['patho_tag'] = torch.tensor(patho_tag, dtype=torch.float64).unsqueeze(1)

            # self.n_patho.append(var_graph.ndata['patho_tag'].mean().item())
            feat_exclude = ['_ID', 'ref_aa', 'lap_pos_enc', 'wl_pos_enc', 'coords']  # skip for current step
            # if self.graph_type == 'hetero':
            nfeat_all = list(var_graph.node_attr_schemes().keys())
            for key in feat_exclude:
                if key in nfeat_all:
                    nfeat_all.pop(nfeat_all.index(key))

            nfeat_comb = list(map(lambda x: var_graph.ndata[x], nfeat_all))
            var_graph.ndata['feat'] = torch.cat(nfeat_comb, dim=-1)
            # else:
            #     if use_nsp:
            #         var_graph.ndata['feat'] = torch.cat([var_graph.ndata['feat'], var_graph.ndata['nsp_feat']], dim=-1)
            # ref_aa = aa_to_index(record['REF_AA'])
            alt_aa = aa_to_index(protein_letters_1to3_extended[record['ALT_AA']].upper())
            self.data.append((var_graph, record['label'], alt_aa, var_idx, record['prot_var_id']))
            self.label.append(record['label'])
            self.n_nodes.append(var_graph.num_nodes())
            self.n_edges.append(var_graph.num_edges())

    def get_var_db(self):
        return self.var_db


class VariantSeqGraphDataSet(GraphDataSetBase):
    def __init__(self, df_in, window_size, feat_dir, lap_pos_enc, wl_pos_enc, pos_enc_dim, graph_type='seq', seq_dict=None,
                 seq_graph_option='star', use_nsp=False, nsp_dir=None, use_cosmis=False, cosmis_dir=None,
                 norm_feat=False, cache_only=False, var_graph_cache='./var_graph_cache', **kwargs):
        super(VariantSeqGraphDataSet, self).__init__()
        # self.n_patho = []
        # self.aa_idx = []
        self.option = seq_graph_option
        self.graph_type = graph_type
        self.window_size = window_size
        self.seq_dict = seq_dict
        self.lap_pos_enc = lap_pos_enc
        self.wl_pos_enc = wl_pos_enc
        self.pos_enc_dim = pos_enc_dim
        self.feat_root = Path(feat_dir)
        self.use_nsp = use_nsp
        self.nsp_dir = nsp_dir
        self.use_cosmis = use_cosmis
        self.cosmis_dir = cosmis_dir
        self.var_graph_path = Path(var_graph_cache) / self.graph_type

        if cache_only and self.var_graph_path.exists():
            self.load_cache_data(df_in)
        else:
            self.process(df_in, norm_feat)


    def load_cache_data(self, df_in):
        for i, record in tqdm(df_in.iterrows(), total=df_in.shape[0]):
            uprot = record['UniProt']
            uprot_pos = record['Protein_position']
            f_var_graph = self.var_graph_path / f'{uprot}_{uprot_pos}.pkl'
            if not self.var_graph_path.exists():
                continue

            with open(f_var_graph, 'rb') as f_pkl:
                var_graph, seq_pos_remain, var_idx = pickle.load(f_pkl)

            if self.use_nsp:
                try:
                    nsp_feat_raw = load_nsp_feats(uprot, self.nsp_dir, exclude=['asa', 'phi', 'psi', 'disorder'])
                    nsp_stats = {'mean': torch.tensor(np.nanmean(nsp_feat_raw, axis=0)),
                                 'min': torch.tensor(np.nanmin(nsp_feat_raw, axis=0)),
                                 'max': torch.tensor(np.nanmax(nsp_feat_raw, axis=0))}
                    nsp_feat, nsp_stats = impute_nan(torch.tensor(nsp_feat_raw[list(map(lambda x: x - 1, seq_pos_remain)), :]), nsp_stats)
                    var_graph.ndata['nps_feat'] = nsp_feat
                except (FileNotFoundError, ValueError) as e:
                    logging.warning(f'{e} in loading feature for {uprot}')
                    continue
            if self.use_cosmis:
                try:
                    cosmis_feat = load_cosmis_feats(uprot, self.cosmis_dir, cols=['cosmis', 'cosmis_pvalue'])
                    var_graph.ndata['cosmis'] = torch.tensor(cosmis_feat[list(map(lambda x: x - 1, seq_pos_remain)), :])
                except FileNotFoundError:
                    continue

            if self.lap_pos_enc:
                lap = laplacian_positional_encoding(var_graph, self.pos_enc_dim)
                if isinstance(lap, type(None)):
                    logging.warning(
                        'Laplacian position encoding not applicable for variant {}'.format(record['prot_var_id']))
                    # print('Variant graph: {} nodes'.format(var_graph.num_nodes()))
                    continue
                else:
                    var_graph.ndata['lap_pos_enc'] = lap
            if self.wl_pos_enc:
                var_graph.ndata['wl_pos_enc'] = wl_positional_encoding(var_graph)

            feat_exclude = ['_ID', 'ref_aa', 'lap_pos_enc', 'wl_pos_enc', 'coords']  # skip for current step
            nfeat_all = list(var_graph.node_attr_schemes().keys())
            for key in feat_exclude:
                if key in nfeat_all:
                    nfeat_all.pop(nfeat_all.index(key))

            nfeat_comb = list(map(lambda x: var_graph.ndata[x], nfeat_all))
            var_graph.ndata['feat'] = torch.cat(nfeat_comb, dim=-1)
            var_graph.edata['feat'] = var_graph.edata['coev']
            alt_aa = aa_to_index(protein_letters_1to3_extended[record['ALT_AA']].upper())

            self.data.append((var_graph, record['label'], alt_aa, var_idx, record['prot_var_id']))
            self.label.append(record['label'])
            self.n_nodes.append(var_graph.num_nodes())
            self.n_edges.append(var_graph.num_edges())


    def process(self, df_in, norm_feat=False):
        for i, record in tqdm(df_in.iterrows(), total=df_in.shape[0]):
            uprot = record['UniProt']
            uprot_pos = record['Protein_position']
            if uprot not in self.seq_dict:
                self.seq_dict[uprot] = fetch_prot_seq(uprot)
            seq = self.seq_dict[uprot]
            seq_array = np.array(
                list(map(lambda x: aa_to_index(protein_letters_1to3_extended[x].upper()), list(seq))))

            bio_feats = load_expasy_feats(uprot, self.feat_root, '3dvip_expasy')
            pssm_feats = load_pssm(uprot, self.feat_root, '3dvip_pssm')
            coev_feat_df = load_coev_feats(uprot, self.feat_root, '3dvip_sca', '3dvip_dca')
            feat_data = np.hstack([bio_feats, pssm_feats])

            seq_src, seq_dst, start_idx, end_idx = self.add_seq_edges(uprot_pos, len(seq), option=self.option)
            var_graph = dgl.graph((seq_src, seq_dst))

            if var_graph.num_nodes() == 0:
                logging.warning('Empty graph for {}:{}'.format(uprot, uprot_pos))
                continue

            var_idx = uprot_pos - start_idx - 1
            aa_feat = torch.tensor(feat_data[start_idx: end_idx + 1, :])
            feat_stats = {'mean': torch.tensor(np.nanmean(feat_data, axis=0)),
                          'min': torch.tensor(np.nanmin(feat_data, axis=0)),
                          'max': torch.tensor(np.nanmax(feat_data, axis=0))}
            if norm_feat:
                aa_feat = normalize_feat(aa_feat, feat_stats)

            var_graph.ndata['aa_feat'] = aa_feat
            var_graph.ndata['ref_aa'] = torch.tensor(seq_array[start_idx:(end_idx + 1)], dtype=torch.int64)

            if self.use_nsp:
                try:
                    nsp_feat_raw = load_nsp_feats(uprot, self.nsp_dir, exclude=['asa', 'phi', 'psi', 'disorder'])
                    nsp_stats = {'mean': torch.tensor(np.nanmean(nsp_feat_raw, axis=0)),
                                 'min': torch.tensor(np.nanmin(nsp_feat_raw, axis=0)),
                                 'max': torch.tensor(np.nanmax(nsp_feat_raw, axis=0))}
                    nsp_feat, nsp_stats = impute_nan(torch.tensor(nsp_feat_raw[start_idx: end_idx + 1, :]), nsp_stats)
                    var_graph.ndata['nps_feat'] = nsp_feat
                except (FileNotFoundError, ValueError) as e:
                    logging.warning(f'{e} in loading feature for {uprot}')
                    continue
            if self.use_cosmis:
                try:
                    cosmis_feat = load_cosmis_feats(uprot, self.cosmis_dir, cols=['cosmis', 'cosmis_pvalue'])
                    var_graph.ndata['cosmis'] = torch.tensor(cosmis_feat[start_idx: end_idx + 1, :])
                except FileNotFoundError:
                    continue

            coev_feat_dict = dict(zip(coev_feat_df['key'], coev_feat_df[['sca', 'mi', 'di']].values.tolist()))

            coev_feat = []
            for j in range(len(seq_src)):
                src_pos = seq_src[j] + 1 + start_idx
                dst_pos = seq_dst[j] + 1 + start_idx
                try:
                    coev_feat.append(coev_feat_dict[(src_pos, dst_pos)])
                except KeyError:
                    coev_feat.append(coev_feat_dict[(dst_pos, src_pos)])

            var_graph.edata['coev'] = torch.tensor(coev_feat)

            if self.lap_pos_enc:
                lap = laplacian_positional_encoding(var_graph, self.pos_enc_dim)
                if isinstance(lap, type(None)):
                    logging.warning(
                        'Laplacian position encoding not applicable for variant {}'.format(record['prot_var_id']))
                    # print('Variant graph: {} nodes'.format(var_graph.num_nodes()))
                    continue
                else:
                    var_graph.ndata['lap_pos_enc'] = lap
            if self.wl_pos_enc:
                var_graph.ndata['wl_pos_enc'] = wl_positional_encoding(var_graph)

            feat_exclude = ['_ID', 'ref_aa', 'lap_pos_enc', 'wl_pos_enc', 'coords']  # skip for current step
            nfeat_all = list(var_graph.node_attr_schemes().keys())
            for key in feat_exclude:
                if key in nfeat_all:
                    nfeat_all.pop(nfeat_all.index(key))

            nfeat_comb = list(map(lambda x: var_graph.ndata[x], nfeat_all))
            var_graph.ndata['feat'] = torch.cat(nfeat_comb, dim=-1)

            alt_aa = aa_to_index(protein_letters_1to3_extended[record['ALT_AA']].upper())

            self.data.append((var_graph, record['label'], alt_aa, var_idx, record['prot_var_id']))
            self.label.append(record['label'])
            self.n_nodes.append(var_graph.num_nodes())
            self.n_edges.append(var_graph.num_edges())


    def add_seq_edges(self, uprot_pos, prot_len, option='star', max_dist=1, inverse=True):
        w = self.window_size // 2
        
        # seq = self.seq_dict[uprot]
        # seq_array = np.array(list(map(lambda x: aa_to_index(protein_letters_1to3_extended[x].upper()), list(seq))))
        start = max(uprot_pos - w - 1, 0)
        end = min(uprot_pos + w - 1, prot_len - 1)
        target_idx = uprot_pos - 1 - start
        g_size = end - start + 1
        nodes = list(range(g_size))

        node_in = []
        node_out = []
        if option == 'seq':
            for d in range(1, max_dist+1):
                node_in.extend(nodes[:-d])
                node_out.extend(nodes[d:])
                # node_in = torch.arange(start, end+1-d)
                # node_out = torch.arange(start+d, end+1)
            if inverse:
                nodes_r = nodes[::-1]
                for d in range(1, max_dist+1):
                    node_in.extend(nodes_r[:-d])
                    node_out.extend(nodes_r[d:])
        else:  # Star-shape graph
            for n_cur in nodes:
                if n_cur != target_idx:
                    node_in.append(n_cur)
                    node_out.append(target_idx)
            if inverse:
                node_out.extend(node_in)
                node_in.extend([target_idx] * (g_size - 1))
                    
        return node_in, node_out, start, end
