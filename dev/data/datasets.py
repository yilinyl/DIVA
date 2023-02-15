import torch
import logging
from torch.utils.data import Dataset
from dev.preprocess.utils import *
from .data_utils import *


class GraphData(object):
    def __init__(self, graph, label, target, var_id=None):
        self.graph = graph
        self.label = label
        self.target = target
        self.var_id = var_id


class VariantGraphDataSet(Dataset):
    def __init__(self, df_in, graph_cache, pdb_root_dir, af_root_dir, feat_dir, sift_map, lap_pos_enc=True, wl_pos_enc=False, pos_enc_dim=None,
                 cov_thres=0.5, num_neighbors=10, distance_type='centroid', method='radius', radius=10, df_ires=None, save=False,
                 anno_ires=False, coord_option=None, feat_stats=None, var_db=None, seq2struct_all=None, **kwargs):
        super(VariantGraphDataSet, self).__init__()

        self.data = []
        # self.aa_idx = []
        # self.aa_mask = []
        self.lap_pos_enc = lap_pos_enc
        self.wl_pos_enc = wl_pos_enc
        self.pos_enc_dim = pos_enc_dim
        if not seq2struct_all:
            seq2struct_all = dict()
        self.seq2struct_dict = seq2struct_all

        feat_root = Path(feat_dir)
        graph_cache_path = Path(graph_cache)
        if not graph_cache_path.exists():
            graph_cache_path.mkdir(parents=True)

        self.var_db = var_db
        if isinstance(var_db, type(None)):
            var_db = df_in
            self.var_db = var_db.groupby(['UniProt', 'Protein_position'])['label'].any().reset_index()
            self.var_db = self.var_db.rename(columns={'label': 'any_patho'}).query('any_patho == True').reset_index(drop=True)

        for i, record in df_in.iterrows():
            uprot = record['UniProt']
            uprot_pos = record['Protein_position']

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
            # f_graph = os.path.join(g_data_dir, '{}_{}_graph.pkl'.format(model, struct_id))
            if f_graph.exists():
                with open(f_graph, 'rb') as f_pkl:
                    prot_graph, chain_res_list = pickle.load(f_pkl)
            else:
                new_graph = build_single_graph(record, model, pdb_root_dir, af_root_dir, graph_cache,
                                               num_neighbors, distance_type, method, radius, df_ires,
                                               save, anno_ires, coord_option)

                if not new_graph:  # fail to build protein graph
                    continue
                prot_graph, chain_res_list = new_graph

            feat_version = record['feat_version']
            feat_path = feat_root / feat_version / 'sequence_features'
            feat_data = load_features(uprot, feat_path)
            # feat_data = normalize_data(feat_data, feat_stats)
            var_graph, seq_pos_remain = extract_variant_graph(uprot_pos, chain, chain_res_list, seq2struct_pos,
                                                              prot_graph, feat_data, feat_stats)

            if var_graph.num_nodes() == 0:
                logging.warning('Empty graph for {}:{}'.format(uprot, uprot_pos))
                continue

            if var_graph.edata['angle'].isnan().any():
                logging.warning('NA in angle data for {}-{}'.format(model, record['prot_var_id']))
                continue
            if var_graph.edata['dist'].isnan().any():
                logging.warning('NA in dist data for {}-{}'.format(model, record['prot_var_id']))
                continue

            if self.lap_pos_enc:
                lap = laplacian_positional_encoding(var_graph, pos_enc_dim)
                if isinstance(lap, type(None)):
                    logging.warning('Laplacian position encoding not applicable for variant {}'.format(record['prot_var_id']))
                    # print('Variant graph: {} nodes'.format(var_graph.num_nodes()))
                    continue
                else:
                    var_graph.ndata['lap_pos_enc'] = lap
            if self.wl_pos_enc:
                var_graph.ndata['wl_pos_enc'] = wl_positional_encoding(var_graph)

            prot_patho_pos = self.var_db.query('UniProt == @uprot')
            patho_tag = list(map(lambda x: x in prot_patho_pos['Protein_position'], seq_pos_remain))
            # for prot_pos in seq_pos_remain:  # indicator for pathogenic / not
                # query = self.var_db.query('UniProt == @uprot & Protein_position == @prot_pos')
                # if len(query):
                #     patho_tag.append(1)
                # else:
                #     patho_tag.append(0)
            var_graph.ndata['patho_tag'] = torch.tensor(patho_tag, dtype=torch.float64)

            # ref_aa = aa_to_index(record['REF_AA'])
            alt_aa = aa_to_index(protein_letters_1to3_extended[record['ALT_AA']].upper())
            self.data.append((var_graph, record['label'], alt_aa, record['prot_var_id']))
            # self.data.append(GraphData(var_graph, record['label', alt_aa, record['prot_var_id']]))
            # aa_idx_cur = []
            # aa_mask_cur = torch.zeros(20, dtype=torch.bool)
            # ref_aa = var_graph.ndata['ref_aa'].detach()
            # for aa in range(20):
            #     matches = (ref_aa == aa).nonzero()
            #     if matches.size(0) == 0:
            #         aa_idx_cur.append(torch.tensor(-1))
            #     else:
            #         aa_mask_cur[aa] = True
            #         aa_idx_cur.append(matches[0].detach().squeeze(0))
            # self.aa_idx.append(torch.stack(aa_idx_cur))
            # self.aa_mask.append(aa_mask_cur)

    def __getitem__(self, index):
        graph, label, alt_aa, var_id = self.data[index]

        return graph, label, alt_aa, var_id

    def __len__(self):
        return len(self.data)

    def get_seq_struct_map(self):
        return self.seq2struct_dict

    def get_var_db(self):
        return self.var_db


# def collate(samples):
#     # The input `samples` is a list of pairs
#     #  (graph, label).
#     graphs, labels = map(list, zip(*samples))
#     batched_graph = dgl.batch(graphs)
#     return batched_graph, torch.tensor(labels)




