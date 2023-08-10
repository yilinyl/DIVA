import torch
import dgl
from tqdm import *
from .dataset_base import GraphDataSetBase
from dev.preprocess.utils import *
from .data_utils import *

from .lm_utils import *

class MultiModalData(object):
    def __init__(self, seq_graph, seq_var_idx, struct_graph, str_var_idx):
        self.seq_graph = seq_graph
        self.seq_var_idx = seq_var_idx
        self.struct_graph = struct_graph
        self.str_var_idx = str_var_idx

    def get_nfeat_dim_all(self, feat_key='feat'):
        return {'seq': self.seq_graph.ndata[feat_key].shape[0],
                'struct': self.struct_graph.ndata[feat_key].shape[0]}

    def get_efeat_dim_all(self, feat_key='feat'):
        return {'seq': self.seq_graph.edata[feat_key].shape[0],
                'struct': self.struct_graph.edata[feat_key].shape[0]}


class MultiModalDataSet(GraphDataSetBase):
    def __init__(self, df_in, window_size, feat_dir, lap_pos_enc, wl_pos_enc, pos_enc_dim, cov_thres=0.5, seq_dict=None,
                 seq_graph_option='star', use_nsp=False, nsp_dir=None, use_cosmis=False, cosmis_dir=None, var_db=None, seq2struct_dict=None,
                 norm_feat=False, var_graph_cache=None, struct_graph_cache=None, seq_graph_cache=None, use_patho_tag=False, 
                 cosmis_cols=['cosmis'], cosmis_suffix='.pkl', use_oe=False, oe_dir=None, use_dssp=False, dssp_dir=None, sift_map=None, 
                 use_ires=False, ires_pred_dir=None, ires_gt_file=None, **kwargs):
        super(MultiModalDataSet, self).__init__()
        # self.n_patho = []
        # self.aa_idx = []
        self.option = seq_graph_option
        # self.graph_type = graph_type
        self.var_db = var_db
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
        self.use_patho_tag = use_patho_tag
        self.seq_graph_stats = {'nodes': [], 'edges': []}
        self.struct_graph_stats = {'nodes': [], 'edges': []}
        if var_graph_cache:
            self.seq_graph_root = Path(var_graph_cache) / 'seq'
            self.struct_graph_root = Path(var_graph_cache) / 'struct'
        else:
            self.seq_graph_root = Path(seq_graph_cache)
            self.struct_graph_root = Path(struct_graph_cache)

        self.cov_thres = cov_thres

        self.load_cache_data(df_in, cosmis_cols, cosmis_suffix)


    def load_cache_data(self, df_in, cosmis_cols=['cosmis'], cosmis_suffix='.pkl'):

        if isinstance(self.var_db, type(None)):
            self.var_db = df_in
            self.var_db = self.var_db.groupby(['UniProt', 'Protein_position'])['label'].any().reset_index()
            self.var_db = self.var_db.rename(columns={'label': 'any_patho'}).query('any_patho == True').reset_index(
                drop=True)

        for i, record in tqdm(df_in.iterrows(), total=df_in.shape[0]):
            uprot = record['UniProt']
            uprot_pos = record['Protein_position']

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
                struct_graph, pos_remain_str, var_idx_struct = pickle.load(f_pkl)

            if struct_graph.edata['angle'].isnan().any():
                logging.warning('NA in angle data for {}-{}'.format(model, record['prot_var_id']))
                continue
            if struct_graph.edata['dist'].isnan().any():
                logging.warning('NA in dist data for {}-{}'.format(model, record['prot_var_id']))
                continue

            # load sequential graph
            f_seq_graph = self.seq_graph_root / f'{uprot}_{uprot_pos}.pkl'
            if not f_seq_graph.exists():
                continue

            with open(f_seq_graph, 'rb') as f_pkl:
                seq_graph, pos_remain_seq, var_idx_seq = pickle.load(f_pkl)
            
            feat_exclude = ['_ID', 'ref_aa', 'lap_pos_enc', 'wl_pos_enc', 'coords']  # skip for feature concatenation

            if self.use_nsp:
                try:
                    nsp_feat_raw = load_nsp_feats(uprot, self.nsp_dir, exclude=['asa', 'phi', 'psi', 'disorder'])
                    nsp_stats = {'mean': torch.tensor(np.nanmean(nsp_feat_raw, axis=0)),
                                 'min': torch.tensor(np.nanmin(nsp_feat_raw, axis=0)),
                                 'max': torch.tensor(np.nanmax(nsp_feat_raw, axis=0))}

                    seq_nsp_feat, nsp_stats = impute_nan(torch.tensor(nsp_feat_raw[list(map(lambda x: x - 1, pos_remain_seq)), :]), nsp_stats)
                    seq_graph.ndata['nsp_feat'] = seq_nsp_feat

                    str_nsp_feat, nsp_stats = impute_nan(
                        torch.tensor(nsp_feat_raw[list(map(lambda x: x - 1, pos_remain_str)), :]), nsp_stats)
                    struct_graph.ndata['nsp_feat'] = str_nsp_feat

                except (FileNotFoundError, ValueError, IndexError) as e:
                    logging.warning(f'{e} in loading feature for {uprot}')
                    continue

            if self.use_cosmis:
                try:
                    cosmis_feat_raw = load_cosmis_feats(uprot, self.cosmis_dir, cols=cosmis_cols, suffix=cosmis_suffix)
                    if np.isnan(cosmis_feat_raw).all():
                        logging.warning("NA in COSMIS for {}".format(uprot))
                        continue

                    cosmis_stats = {'mean': torch.tensor(np.nanmean(cosmis_feat_raw, axis=0)),
                                    'min': torch.tensor(np.nanmin(cosmis_feat_raw, axis=0)),
                                    'max': torch.tensor(np.nanmax(cosmis_feat_raw, axis=0))}

                    seq_cosmis_feat, cosmis_stats = impute_nan(torch.tensor(cosmis_feat_raw[list(map(lambda x: x - 1, pos_remain_seq)), :]), cosmis_stats)
                    seq_graph.ndata['cosmis'] = seq_cosmis_feat

                    str_cosmis_feat, cosmis_stats = impute_nan(torch.tensor(cosmis_feat_raw[list(map(lambda x: x - 1, pos_remain_str)), :]), cosmis_stats)
                    struct_graph.ndata['cosmis'] = str_cosmis_feat

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

                    seq_oe_feat, oe_stats = impute_nan(torch.tensor(oe_feat_raw[list(map(lambda x: x - 1, pos_remain_seq)), :]), oe_stats)
                    seq_graph.ndata['oe'] = seq_oe_feat

                    str_oe_feat, oe_stats = impute_nan(torch.tensor(oe_feat_raw[list(map(lambda x: x - 1, pos_remain_str)), :]), oe_stats)
                    struct_graph.ndata['oe'] = str_oe_feat

                except (FileNotFoundError, ValueError, IndexError) as e:
                    logging.warning(f'{e} in loading OE feature for {uprot}')
                    continue
            
            if self.use_dssp:
                try:
                    ss8, dssp_feat_raw = load_dssp_feats(struct_id, pos_remain_str, chain, model, seq2struct_pos, self.dssp_dir)
                    struct_graph.ndata['ss'] = torch.tensor(list(map(ss8_to_index, ss8))).unsqueeze(-1)
                    struct_graph.ndata['dssp'] = torch.tensor(dssp_feat_raw)
                    feat_exclude.append('ss')
                    # TODO: encode ss8
                except Exception as e:
                    logging.warning(f'{e} in loading DSSP feature for {uprot} at {uprot_pos}')
                    continue
            
            if self.use_ires:
                try:
                    ires_feat = load_ires_feats(uprot, record['prot_length'], self.ires_pred_dir, self.ires_gt_dict)
                    struct_graph.ndata['ires'] = torch.tensor(ires_feat[list(map(lambda x: x - 1, pos_remain_str))]).unsqueeze(-1)
                    seq_graph.ndata['ires'] = torch.tensor(ires_feat[list(map(lambda x: x - 1, pos_remain_seq))]).unsqueeze(-1)
                except Exception as e:
                    logging.warning(f'{e} in loading IRES feature for {uprot}')
                    struct_graph.ndata['ires'] = torch.zeros((struct_graph.num_nodes(), 1))
                    seq_graph.ndata['ires'] = torch.zeros((seq_graph.num_nodes(), 1))

            isolated_nodes = ((struct_graph.in_degrees() == 0) & (struct_graph.out_degrees() == 0)).nonzero().squeeze(1)
            struct_graph = dgl.remove_nodes(struct_graph, isolated_nodes)

            if self.use_patho_tag:
                prot_patho_pos = self.var_db.query('UniProt == @uprot')
                patho_tag = list(map(lambda x: x in prot_patho_pos['Protein_position'], pos_remain_seq))
                # self.n_patho.append(sum(patho_tag))
                seq_graph.ndata['patho_tag'] = torch.tensor(patho_tag, dtype=torch.float64).unsqueeze(1)

            if self.lap_pos_enc:
                lap_seq = laplacian_positional_encoding(seq_graph, self.pos_enc_dim)
                lap_str = laplacian_positional_encoding(struct_graph, self.pos_enc_dim)
                if isinstance(lap_seq, type(None)) or isinstance(lap_str, type(None)):
                    logging.warning(
                        'Laplacian position encoding not applicable for variant {}'.format(record['prot_var_id']))
                    # print('Variant graph: {} nodes'.format(var_graph.num_nodes()))
                    continue
                else:
                    seq_graph.ndata['lap_pos_enc'] = lap_seq
                    struct_graph.ndata['lap_pos_enc'] = lap_str

            if self.wl_pos_enc:
                seq_graph.ndata['wl_pos_enc'] = wl_positional_encoding(seq_graph)
                struct_graph.ndata['wl_pos_enc'] = wl_positional_encoding(struct_graph)

            struct_graph.ndata['feat'] = self._compile_feats(struct_graph, feat_exclude)
            # struct_graph.edata['feat'] =
            seq_graph.ndata['feat'] = self._compile_feats(seq_graph, feat_exclude)
            seq_graph.edata['feat'] = seq_graph.edata['coev']
            alt_aa = aa_to_index(protein_letters_1to3_extended[record['ALT_AA']].upper())

            # graph_data = MultiModalData(seq_graph, var_idx_seq, struct_graph, var_idx_struct)
            graph_data_dict = {'seq_graph': seq_graph, 'seq_idx': var_idx_seq, 'struct_graph': struct_graph,
                          'struct_idx': var_idx_struct}
            self.data.append((graph_data_dict, record['label'], alt_aa, record['prot_var_id']))
            self.label.append(record['label'])
            self.seq_graph_stats['nodes'].append(seq_graph.num_nodes())
            self.seq_graph_stats['edges'].append(seq_graph.num_edges())
            self.struct_graph_stats['nodes'].append(struct_graph.num_nodes())
            self.struct_graph_stats['edges'].append(struct_graph.num_edges())

    def __getitem__(self, index):
        graph_data_dict, label, alt_aa, var_id = self.data[index]

        return graph_data_dict, label, alt_aa, var_id

    def get_ndata_dim(self, feat_name='feat'):
        graph_data_dict = self.data[0][0]

        ndata_dim_all = {'seq': graph_data_dict['seq_graph'].ndata[feat_name].shape[1],
                         'struct': graph_data_dict['struct_graph'].ndata[feat_name].shape[1]}
        return ndata_dim_all

    def get_edata_dim(self, feat_name='feat'):
        graph_data_dict = self.data[0][0]

        edata_dim_all = {'seq': graph_data_dict['seq_graph'].edata[feat_name].shape[1],
                         'struct': graph_data_dict['struct_graph'].edata[feat_name].shape[1]}
        return edata_dim_all

    def get_var_db(self):
        return self.var_db

    def _compile_feats(self, g, feat_exclude, feat_keep=None):
        nfeat_all = list(g.node_attr_schemes().keys())
        if feat_keep:
            nfeat_comb = list(map(lambda x: g.ndata[x], feat_keep))
        else:
            for key in feat_exclude:
                if key in nfeat_all:
                    nfeat_all.pop(nfeat_all.index(key))

            nfeat_comb = list(map(lambda x: g.ndata[x], nfeat_all))

        return torch.cat(nfeat_comb, dim=-1)


# def collate(samples):
#     # The input `samples` is a list of pairs
#     #  (graph, label).
#     graphs, labels = map(list, zip(*samples))
#     batched_graph = dgl.batch(graphs)
#     return batched_graph, torch.tensor(labels)


class MultiModalLMDataset(GraphDataSetBase):
    def __init__(self, df_in, tokenizer, lm_model, device, lap_pos_enc, wl_pos_enc, pos_enc_dim, cov_thres=0.5, seq_dict=None,
                 use_lm_cache=False, lm_cache=None, var_graph_cache=None, struct_graph_cache=None, seq_graph_cache=None, 
                 use_nsp=False, nsp_dir=None, use_cosmis=False, cosmis_dir=None, cosmis_cols=['cosmis'], cosmis_suffix='.pkl',
                 use_oe=False, oe_dir=None, **kwargs):
        super(MultiModalLMDataset, self).__init__()
        self.seq_dict = seq_dict
        self.tokenizer = tokenizer
        self.lm_model = lm_model
        # self.lm_model = self.lm_model.to(device)
        self.lm_cache = lm_cache
        self.lm_dict = dict()
        self.device = device
        if use_lm_cache and self.lm_cache:
            with open(self.lm_cache, 'rb') as f_pkl:
                self.lm_dict = pickle.load(f_pkl)

        self.lap_pos_enc = lap_pos_enc
        self.wl_pos_enc = wl_pos_enc
        self.pos_enc_dim = pos_enc_dim
        self.cov_thres = cov_thres
        self.use_nsp = use_nsp
        self.nsp_dir = nsp_dir
        self.use_cosmis = use_cosmis
        self.cosmis_dir = cosmis_dir
        self.use_oe = use_oe
        self.oe_dir = oe_dir
        self.seq_graph_stats = {'nodes': [], 'edges': []}
        self.struct_graph_stats = {'nodes': [], 'edges': []}
        # self.nfeat_key = 'feat_ref'

        if var_graph_cache:
            self.seq_graph_root = Path(var_graph_cache) / 'seq'
            self.struct_graph_root = Path(var_graph_cache) / 'struct'
        else:
            self.seq_graph_root = Path(seq_graph_cache)
            self.struct_graph_root = Path(struct_graph_cache)

        self.process(df_in, cosmis_cols=cosmis_cols, cosmis_suffix=cosmis_suffix)


    def process(self, df_in, norm_feat=False, max_len=1200, cosmis_cols=['cosmis'], cosmis_suffix='.pkl'):
        for i, record in tqdm(df_in.iterrows(), total=df_in.shape[0]):
            if record['prot_length'] > max_len:
                continue

            uprot = record['UniProt']
            uprot_pos = record['Protein_position']

            if record['PDB_coverage'] >= self.cov_thres:
                model = 'PDB'
                struct_id = record['PDB']
                chain = record['Chain']
                f_struct_graph = self.struct_graph_root / f'{model}-{struct_id}_{uprot}_{uprot_pos}.pkl'

            else:
                model = 'AF'
                struct_id = uprot
                chain = 'A'
                f_struct_graph = self.struct_graph_root / f'{model}-{struct_id}_{uprot_pos}.pkl'
            
            if not f_struct_graph.exists():  # only load pre-constructed protein structure graph
                continue
            # if not f_struct_graph.exists():
            #     new_graph = build_struct_graph(record, model, pdb_root_dir, af_root_dir, graph_cache,
            #                                    num_neighbors, distance_type, method, radius, df_ires,
            #                                    save, anno_ires, coord_option)
            #
            #     if not new_graph:  # fail to build protein graph
            #         continue
            #     prot_graph, chain_res_list = new_graph

            with open(f_struct_graph, 'rb') as f_pkl:
                # pos_remain_str: sequential position of remained residues in structural graph
                struct_graph, pos_remain_str, var_idx_struct = pickle.load(f_pkl)

            if struct_graph.edata['angle'].isnan().any():
                logging.warning('NA in angle data for {}-{}'.format(model, record['prot_var_id']))
                continue
            if struct_graph.edata['dist'].isnan().any():
                logging.warning('NA in dist data for {}-{}'.format(model, record['prot_var_id']))
                continue

            # load sequential graph
            f_seq_graph = self.seq_graph_root / f'{uprot}_{uprot_pos}.pkl'
            if not f_seq_graph.exists():
                continue

            with open(f_seq_graph, 'rb') as f_pkl:
                seq_graph, pos_remain_seq, var_idx_seq = pickle.load(f_pkl)
            if self.use_nsp:
                try:
                    nsp_feat_raw = load_nsp_feats(uprot, self.nsp_dir, exclude=['asa', 'phi', 'psi', 'disorder'])
                    nsp_stats = {'mean': torch.tensor(np.nanmean(nsp_feat_raw, axis=0)),
                                 'min': torch.tensor(np.nanmin(nsp_feat_raw, axis=0)),
                                 'max': torch.tensor(np.nanmax(nsp_feat_raw, axis=0))}

                    seq_nsp_feat, nsp_stats = impute_nan(torch.tensor(nsp_feat_raw[list(map(lambda x: x - 1, pos_remain_seq)), :]), nsp_stats)
                    seq_graph.ndata['nsp_feat'] = seq_nsp_feat

                    str_nsp_feat, nsp_stats = impute_nan(
                        torch.tensor(nsp_feat_raw[list(map(lambda x: x - 1, pos_remain_str)), :]), nsp_stats)
                    struct_graph.ndata['nsp_feat'] = str_nsp_feat

                except (FileNotFoundError, ValueError, IndexError) as e:
                    logging.warning(f'{e} in loading feature for {uprot}')
                    continue

            if self.use_cosmis:
                try:
                    cosmis_feat_raw = load_cosmis_feats(uprot, self.cosmis_dir, cols=cosmis_cols, suffix=cosmis_suffix)
                    if np.isnan(cosmis_feat_raw).all():
                        logging.warning("NA in COSMIS for {}".format(uprot))
                        continue

                    cosmis_stats = {'mean': torch.tensor(np.nanmean(cosmis_feat_raw, axis=0)),
                                    'min': torch.tensor(np.nanmin(cosmis_feat_raw, axis=0)),
                                    'max': torch.tensor(np.nanmax(cosmis_feat_raw, axis=0))}
                    
                    seq_cosmis_feat, cosmis_stats = impute_nan(torch.tensor(cosmis_feat_raw[list(map(lambda x: x - 1, pos_remain_seq)), :]), cosmis_stats)
                    seq_graph.ndata['cosmis'] = seq_cosmis_feat

                    str_cosmis_feat, cosmis_stats = impute_nan(torch.tensor(cosmis_feat_raw[list(map(lambda x: x - 1, pos_remain_str)), :]), cosmis_stats)
                    struct_graph.ndata['cosmis'] = str_cosmis_feat

                except FileNotFoundError:
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

                    seq_oe_feat, oe_stats = impute_nan(torch.tensor(oe_feat_raw[list(map(lambda x: x - 1, pos_remain_seq)), :]), oe_stats)
                    seq_graph.ndata['oe'] = seq_oe_feat

                    str_oe_feat, oe_stats = impute_nan(torch.tensor(oe_feat_raw[list(map(lambda x: x - 1, pos_remain_str)), :]), oe_stats)
                    struct_graph.ndata['oe'] = str_oe_feat

                except (FileNotFoundError, ValueError, IndexError) as e:
                    logging.warning(f'{e} in loading OE feature for {uprot}')
                    continue

            if uprot not in self.seq_dict:
                self.seq_dict[uprot] = fetch_prot_seq(uprot)
            seq = self.seq_dict[uprot]
            emb_ref = calc_esm_emb(seq, self.tokenizer, self.lm_model)
            # emb_ref = emb_ref.cpu()
            seq_alt = seq[:uprot_pos - 1] + record['ALT_AA'] + seq[uprot_pos:]
            emb_alt = calc_esm_emb(seq_alt, self.tokenizer, self.lm_model)
            # emb_alt = emb_alt.cpu()
            try:
                struct_graph.ndata['esm_ref'] = emb_ref[list(map(lambda x: x - 1, pos_remain_str)), :]
                struct_graph.ndata['feat_alt'] = emb_alt[list(map(lambda x: x - 1, pos_remain_str)), :]
            except IndexError as e:
                logging.warning(f'{e} for {uprot}-{uprot_pos}')
                continue

            seq_graph.ndata['esm_ref'] = emb_ref[list(map(lambda x: x - 1, pos_remain_seq)), :]
            seq_graph.ndata['feat_alt'] = emb_alt[list(map(lambda x: x - 1, pos_remain_seq)), :]
            nfeat_all = ['esm_ref']
            if self.use_nsp:
                nfeat_all.append('nsp_feat')
            if self.use_cosmis:
                nfeat_all.append('cosmis')
            if self.use_oe:
                nfeat_all.append('oe')
            
            struct_graph.ndata['feat_ref'] = torch.cat(list(map(lambda x: struct_graph.ndata[x], nfeat_all)), dim=-1)
            seq_graph.ndata['feat_ref'] = torch.cat(list(map(lambda x: seq_graph.ndata[x], nfeat_all)), dim=-1)
            # with torch.cuda.device(self.device):
            #     del emb_ref
            #     del emb_alt
            #     torch.cuda.empty_cache()
            alt_aa = aa_to_index(protein_letters_1to3_extended[record['ALT_AA']].upper())

            isolated_nodes = ((struct_graph.in_degrees() == 0) & (struct_graph.out_degrees() == 0)).nonzero().squeeze(1)
            struct_graph = dgl.remove_nodes(struct_graph, isolated_nodes)

            if self.lap_pos_enc:
                lap_seq = laplacian_positional_encoding(seq_graph, self.pos_enc_dim)
                lap_str = laplacian_positional_encoding(struct_graph, self.pos_enc_dim)
                if isinstance(lap_seq, type(None)) or isinstance(lap_str, type(None)):
                    logging.warning(
                        'Laplacian position encoding not applicable for variant {}'.format(record['prot_var_id']))
                    # print('Variant graph: {} nodes'.format(var_graph.num_nodes()))
                    continue
                else:
                    seq_graph.ndata['lap_pos_enc'] = lap_seq
                    struct_graph.ndata['lap_pos_enc'] = lap_str

            if self.wl_pos_enc:
                seq_graph.ndata['wl_pos_enc'] = wl_positional_encoding(seq_graph)
                struct_graph.ndata['wl_pos_enc'] = wl_positional_encoding(struct_graph)

            seq_graph.edata['feat'] = seq_graph.edata['coev']

            graph_data_dict = {'seq_graph': seq_graph, 'seq_idx': var_idx_seq, 'struct_graph': struct_graph,
                               'struct_idx': var_idx_struct}
            self.data.append((graph_data_dict, record['label'], alt_aa, record['prot_var_id']))
            self.label.append(record['label'])
            self.seq_graph_stats['nodes'].append(seq_graph.num_nodes())
            self.seq_graph_stats['edges'].append(seq_graph.num_edges())
            self.struct_graph_stats['nodes'].append(struct_graph.num_nodes())
            self.struct_graph_stats['edges'].append(struct_graph.num_edges())

    def __getitem__(self, index):
        graph_data_dict, label, alt_aa, var_id = self.data[index]

        return graph_data_dict, label, alt_aa, var_id


    def get_ndata_dim(self, feat_name='feat'):
        graph_data_dict = self.data[0][0]

        ndata_dim_all = {'seq': (graph_data_dict['seq_graph'].ndata[f'{feat_name}_ref'].shape[1], 
                                 graph_data_dict['seq_graph'].ndata[f'{feat_name}_alt'].shape[1]),
                         'struct': (graph_data_dict['struct_graph'].ndata[f'{feat_name}_ref'].shape[1],
                                    graph_data_dict['struct_graph'].ndata[f'{feat_name}_alt'].shape[1])}
        return ndata_dim_all

    def get_edata_dim(self, feat_name='feat'):
        graph_data_dict = self.data[0][0]

        edata_dim_all = {'seq': graph_data_dict['seq_graph'].edata[feat_name].shape[1],
                         'struct': graph_data_dict['struct_graph'].edata[feat_name].shape[1]}
        return edata_dim_all
