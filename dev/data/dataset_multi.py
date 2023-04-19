import torch
import dgl
from tqdm import *
from .dataset_base import GraphDataSetBase
from dev.preprocess.utils import *
from .data_utils import *


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
                 seq_graph_option='star', use_nsp=False, nsp_dir=None, use_cosmis=False, cosmis_dir=None, var_db=None,
                 norm_feat=False, var_graph_cache=None, struct_graph_cache=None, seq_graph_cache=None, **kwargs):
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
        if var_graph_cache:
            self.seq_graph_root = Path(var_graph_cache) / 'seq'
            self.struct_graph_root = Path(var_graph_cache) / 'struct'
        else:
            self.seq_graph_root = Path(seq_graph_cache)
            self.struct_graph_root = Path(struct_graph_cache)

        self.cov_thres = cov_thres

        self.load_cache_data(df_in)


    def load_cache_data(self, df_in):

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

            else:
                model = 'AF'
                struct_id = uprot
                chain = 'A'
                key = ':'.join([uprot, model, chain])

            f_struct_graph = self.struct_graph_root / f'{model}-{struct_id}_{uprot_pos}.pkl'
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

            if self.use_nsp:
                try:
                    nsp_feat_raw = load_nsp_feats(uprot, self.nsp_dir, exclude=['asa', 'phi', 'psi', 'disorder'])
                    nsp_stats = {'mean': torch.tensor(np.nanmean(nsp_feat_raw, axis=0)),
                                 'min': torch.tensor(np.nanmin(nsp_feat_raw, axis=0)),
                                 'max': torch.tensor(np.nanmax(nsp_feat_raw, axis=0))}

                    seq_nsp_feat, nsp_stats = impute_nan(torch.tensor(nsp_feat_raw[list(map(lambda x: x - 1, pos_remain_seq)), :]), nsp_stats)
                    seq_graph.ndata['nps_feat'] = seq_nsp_feat

                    str_nsp_feat, nsp_stats = impute_nan(
                        torch.tensor(nsp_feat_raw[list(map(lambda x: x - 1, pos_remain_str)), :]), nsp_stats)
                    struct_graph.ndata['nps_feat'] = str_nsp_feat

                except (FileNotFoundError, ValueError) as e:
                    logging.warning(f'{e} in loading feature for {uprot}')
                    continue

            if self.use_cosmis:
                try:
                    cosmis_feat = load_cosmis_feats(uprot, self.cosmis_dir, cols=['cosmis', 'cosmis_pvalue'])
                    seq_graph.ndata['cosmis'] = torch.tensor(cosmis_feat[list(map(lambda x: x - 1, pos_remain_seq)), :])
                    struct_graph.ndata['cosmis'] = torch.tensor(cosmis_feat[list(map(lambda x: x - 1, pos_remain_str)), :])
                except FileNotFoundError:
                    continue

            isolated_nodes = ((struct_graph.in_degrees() == 0) & (struct_graph.out_degrees() == 0)).nonzero().squeeze(1)
            struct_graph = dgl.remove_nodes(struct_graph, isolated_nodes)

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

            feat_exclude = ['_ID', 'ref_aa', 'lap_pos_enc', 'wl_pos_enc', 'coords']  # skip for current step

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
            # self.n_nodes.append(seq_graph.num_nodes())
            # self.n_edges.append(seq_graph.num_edges())

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

    def _compile_feats(self, g, feat_exclude):
        nfeat_all = list(g.node_attr_schemes().keys())
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
