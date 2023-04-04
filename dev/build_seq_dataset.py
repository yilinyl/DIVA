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


def build_variant_graph(df_in, feat_dir, window_size=128, graph_type='seq', save_var_graph=False, var_graph_cache='./var_graph_cache',
                        option='star', norm_feat=False, **kwargs):

    feat_root = Path(feat_dir)
    var_graph_path = Path(var_graph_cache) / graph_type
    if save_var_graph:
        if not var_graph_path.exists():
            var_graph_path.mkdir(parents=True)
    uprot_prev = ''
    for i, record in tqdm(df_in.iterrows(), total=df_in.shape[0]):
        uprot = record['UniProt']
        uprot_pos = record['Protein_position']
        if uprot not in seq_dict:
            seq_dict[uprot] = fetch_prot_seq(uprot)
        seq = seq_dict[uprot]
        seq_array = np.array(
            list(map(lambda x: aa_to_index(protein_letters_1to3_extended[x].upper()), list(seq))))

        f_var_graph = var_graph_path / f'{uprot}_{uprot_pos}.pkl'
        if f_var_graph.exists():
            continue
        if uprot != uprot_prev:
            bio_feats = load_expasy_feats(uprot, feat_root, '3dvip_expasy')
            pssm_feats = load_pssm(uprot, feat_root, '3dvip_pssm')
            coev_feat_df = load_coev_feats(uprot, feat_root, '3dvip_sca', '3dvip_dca')
            feat_data = np.hstack([bio_feats, pssm_feats])

        seq_src, seq_dst, start_idx, end_idx = add_seq_edges(uprot_pos, len(seq), window_size, option=option)
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

        seq_pos_remain = list(range(start_idx + 1, end_idx + 2))  # convert index to 1-based sequential positions
        if save_var_graph and not f_var_graph.exists():
            with open(f_var_graph, 'wb') as f_pkl:
                pickle.dump((var_graph, seq_pos_remain, var_idx), f_pkl)

        uprot_prev = uprot


def add_seq_edges(uprot_pos, prot_len, window_size, option='star', max_dist=1, inverse=True):
    w = window_size // 2

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
        for d in range(1, max_dist + 1):
            node_in.extend(nodes[:-d])
            node_out.extend(nodes[d:])
            # node_in = torch.arange(start, end+1-d)
            # node_out = torch.arange(start+d, end+1)
        if inverse:
            nodes_r = nodes[::-1]
            for d in range(1, max_dist + 1):
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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/seq_data_config.json', help="Config file path (.json)")
    parser.add_argument('--fname', default='train.csv', help="Input file name")

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    data_params = config['data_params']
    data_path = Path(config['data_dir'])

    df_var = pd.read_csv(data_path / args.fname)

    seq_dict = dict()
    for fname in data_params['seq_fasta']:
        try:
            seq_dict.update(parse_fasta(fname))
        except FileNotFoundError:
            pass
    data_params['seq_dict'] = seq_dict

    build_variant_graph(df_var, **data_params)
