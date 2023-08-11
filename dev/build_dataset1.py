import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import dgl
import numpy as np
import torch
import logging
from tqdm import *
from torch.utils.data import Dataset
from preprocess.utils import *
from data.data_utils import *

torch.set_default_dtype(torch.float64)


# def add_seq_edges(uprot_pos, prot_len, window_size=128, max_dist=1, inverse=True):
#     w = window_size // 2
#     # if uprot not in seq_dict:
#     #     self.seq_dict[uprot] = fetch_prot_seq(uprot)
#     # seq = self.seq_dict[uprot]
#     # seq_array = np.array(list(map(lambda x: aa_to_index(protein_letters_1to3_extended[x].upper()), list(seq))))
#     start = max(uprot_pos - w - 1, 0)
#     end = min(uprot_pos + w - 1, prot_len - 1)
#     g_size = end - start + 1
#     nodes = list(range(g_size))
#
#     node_in = []
#     node_out = []
#     for d in range(1, max_dist+1):
#         node_in.extend(nodes[:-d])
#         node_out.extend(nodes[d:])
#         # node_in = torch.arange(start, end+1-d)
#         # node_out = torch.arange(start+d, end+1)
#     if inverse:
#         nodes_r = nodes[::-1]
#         for d in range(1, max_dist+1):
#             node_in.extend(nodes_r[:-d])
#             node_out.extend(nodes_r[d:])
#
#     return node_in, node_out, start, end


def build_variant_graph(df_in, graph_cache, pdb_root_dir, af_root_dir, feat_dir, sift_map,
                        cov_thres=0.5, num_neighbors=10, distance_type='centroid', method='radius', radius=10, df_ires=None, save=False,
                        anno_ires=False, coord_option=None, feat_stats=None, var_db=None, seq2struct_dict=None,
                        var_graph_radius=None, seq_dict=None, window_size=None, graph_type='hetero', norm_feat=True,
                        save_var_graph=False, var_graph_cache='./var_graph_cache', nsp_dir='', overwrite=False, **kwargs):
    if not seq2struct_dict:
        seq2struct_dict = dict()

    if not seq_dict:
        seq_dict = dict()

    feat_root = Path(feat_dir)
    graph_cache_path = Path(graph_cache)
    if not graph_cache_path.exists():
        graph_cache_path.mkdir(parents=True)

    var_graph_path = Path(var_graph_cache) / graph_type
    if save_var_graph:
        if not var_graph_path.exists():
            var_graph_path.mkdir(parents=True)

    # var_db = var_db
    # if isinstance(var_db, type(None)):
    #     var_db = df_in
    #     var_db = var_db.groupby(['UniProt', 'Protein_position'])['label'].any().reset_index()
    #     var_db = var_db.rename(columns={'label': 'any_patho'}).query('any_patho == True').reset_index(
    #         drop=True)
    uprot_prev = ''

    for i, record in tqdm(df_in.iterrows(), total=df_in.shape[0]):
        uprot = record['UniProt']
        uprot_pos = record['Protein_position']
        if uprot_pos > record['prot_length']:
            logging.warning('Invalid position {} for protein {} (length {})'.format(uprot_pos, uprot, record['prot_length']))
            continue

        if uprot not in seq_dict:
            seq_dict[uprot] = fetch_prot_seq(uprot)
        seq = seq_dict[uprot]
        if record['PDB_coverage'] >= cov_thres:
            model = 'PDB'
            struct_id = record['PDB']
            chain = record['Chain']
            key = ':'.join([uprot, struct_id, chain])
            f_var_graph = var_graph_path / f'{model}-{struct_id}_{uprot}_{uprot_pos}.pkl'
            if key not in seq2struct_dict:
                struct_info = sift_map.query('UniProt == @uprot & PDB == @struct_id & Chain == @chain').iloc[0]
                seq_pos = list(map(int, unzip_res_range(struct_info['MappableResInPDBChainOnUniprotBasis'])))
                struct_pos = unzip_res_range(struct_info['MappableResInPDBChainOnPDBBasis'])
                seq2struct_dict[key] = dict(zip(seq_pos, struct_pos))

            seq2struct_pos = seq2struct_dict[key]
            if max(seq2struct_pos.keys()) > len(seq):
                logging.warning('Inconsistent sequence length for {}'.format(uprot))
                continue

        else:
            model = 'AF'
            struct_id = uprot
            chain = 'A'
            key = ':'.join([uprot, model, chain])
            seq_pos = list(range(1, record['prot_length'] + 1))
            struct_pos = list(map(str, seq_pos))
            seq2struct_dict[key] = dict(zip(seq_pos, struct_pos))
            seq2struct_pos = seq2struct_dict[key]
            f_var_graph = var_graph_path / f'{model}-{struct_id}_{uprot_pos}.pkl'

        f_graph = graph_cache_path / '{}_{}_graph.pkl'.format(model, struct_id)
        # f_graph = os.path.join(g_data_dir, '{}_{}_graph.pkl'.format(model, struct_id))
        
        if overwrite and f_var_graph.exists():
            with open(f_var_graph, 'rb') as f_pkl:
                var_graph_prev, seq_pos_prev, var_idx_prev = pickle.load(f_pkl)
        else:
            if f_var_graph.exists():
                continue
        # if model != model_prev or struct_id != struct_prev:
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
                # nsp_feat = load_nsp_feats(uprot, nsp_dir, exclude=['asa', 'phi', 'psi', 'disorder'])
                # feat_data = np.hstack([bio_feats, pssm_feats])
                feat_data = np.hstack([bio_feats, pssm_feats])
            except FileNotFoundError:
                logging.warning('Feature file not available for {}'.format(uprot))
                continue

        if graph_type == 'struct':
            # Extract structure-based variant graph
            var_graph_new, seq_pos_remain = extract_variant_graph(uprot_pos, chain, chain_res_list, seq2struct_pos,
                                                              prot_graph, feat_data, feat_stats, coev_feat_df,
                                                              patch_radius=var_graph_radius, normalize=norm_feat)
            struct_etype = '_E'
            struct_g = var_graph_new
            var_idx = 0
        else:
            seq_array = np.array(
                list(map(lambda x: aa_to_index(protein_letters_1to3_extended[x].upper()), list(seq))))

            seq_src, seq_dst, start_idx, end_idx = add_seq_edges(uprot_pos, len(seq), window_size, option='star')
            seq_nodes = list(range(start_idx, end_idx + 1))

            g_sca = add_coev_edges(coev_feat_df, 'sca', nlargest=10)
            g_dca = add_coev_edges(coev_feat_df, ['di', 'mi'], nlargest=20)
            g_sca_sub = dgl.node_subgraph(g_sca, seq_nodes)
            g_dca_sub = dgl.node_subgraph(g_dca, seq_nodes)

            try:
                struct_g = fetch_struct_edges(start_idx, end_idx, chain, prot_graph,
                                                                         chain_res_list, seq2struct_pos)
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

            var_graph_new = dgl.heterograph(edge_dict)
            var_graph_new.edges['struct'].data['angle'], angle_stats = impute_nan(struct_g.edata['angle'], angle_stats)
            var_graph_new.edges['struct'].data['dist'], dist_stats = impute_nan(struct_g.edata['dist'], dist_stats)

            if norm_feat:
                var_graph_new.edges['struct'].data['angle'] = normalize_feat(var_graph_new.edges['struct'].data['angle'], angle_stats)
                var_graph_new.edges['struct'].data['dist'] = normalize_feat(var_graph_new.edges['struct'].data['dist'], dist_stats)

            var_graph_new.ndata['aa_feat'] = torch.tensor(feat_data[start_idx: end_idx + 1])
            var_graph_new.edges['sca'].data['sca'] = g_sca_sub.edata['sca'].unsqueeze(1)
            var_graph_new.edges['dca'].data['dca'] = torch.cat([g_dca_sub.edata['di'].unsqueeze(1),
                                                            g_dca_sub.edata['mi'].unsqueeze(1)], dim=-1)

            var_graph_new.ndata['ref_aa'] = torch.tensor(seq_array[start_idx:(end_idx + 1)], dtype=torch.int64)
            seq_pos_remain = list(range(start_idx + 1, end_idx + 2))  # convert index to 1-based sequential positions
            struct_etype = 'struct'
            struct_g = dgl.edge_type_subgraph(var_graph_new, etypes=['struct'])

        if var_graph_new.num_nodes() == 0:
            logging.warning('Empty graph for {}:{}'.format(uprot, uprot_pos))
            continue

        # Operations on structual subgraph
        if struct_g.edata['angle'].isnan().any():
            logging.warning('NA in angle data for {}-{}'.format(model, record['prot_var_id']))
            continue
        if struct_g.edata['dist'].isnan().any():
            logging.warning('NA in dist data for {}-{}'.format(model, record['prot_var_id']))
            continue
        
        if f_var_graph.exists():
            if overwrite and var_graph_new.num_nodes() != var_graph_prev.num_nodes():
                logging.warning('Rewrite variant graph for {}-{}; #nodes: {} -> {}'.format(struct_id, uprot_pos, 
                                                                                           var_graph_prev.num_nodes(), var_graph_new.num_nodes()))
                
                with open(f_var_graph, 'wb') as f_pkl:
                    pickle.dump((var_graph_new, seq_pos_remain, var_idx), f_pkl)
        else:
            with open(f_var_graph, 'wb') as f_pkl:
                pickle.dump((var_graph_new, seq_pos_remain, var_idx), f_pkl)

        uprot_prev = uprot


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/data_config.json', help="Config file path (.json)")
    parser.add_argument('--fname', default='train.csv', help="Input file name")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite variant graph or not")

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    data_params = config['data_params']
    data_path = Path(config['data_dir'])
    # Graph cache config
    graph_cache_root = Path(data_params['graph_cache_root'])
    if data_params['method'] == 'radius':
        graph_cache = graph_cache_root / f'radius{data_params["radius"]}'
    else:
        graph_cache = graph_cache_root / f'knn{data_params["num_neighbors"]}'

    data_params['graph_cache'] = os.fspath(graph_cache)

    df_var = pd.read_csv(data_path / args.fname)
    df_var = df_var.drop_duplicates(['UniProt', 'Protein_position', 'PDB'])
    sift_map = pd.read_csv(data_params['sift_file'], sep='\t').dropna().reset_index(drop=True)
    sift_map = sift_map.merge(df_var, how='inner').drop_duplicates().reset_index(drop=True)
    if Path(data_params['seq2struct_cache']).exists():
        with open(data_params['seq2struct_cache'], 'rb') as f_pkl:
            seq_struct_dict = pickle.load(f_pkl)
    else:
        seq_struct_dict = dict()

    seq_dict = dict()
    for fname in data_params['seq_fasta']:
        try:
            seq_dict.update(parse_fasta(fname))
        except FileNotFoundError:
            pass
    data_params['seq_dict'] = seq_dict

    build_variant_graph(df_var, sift_map=sift_map, seq2struct_dict=seq_struct_dict, overwrite=args.overwrite, **data_params)
