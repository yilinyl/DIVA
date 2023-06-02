import copy
import pickle, hashlib, torch, dgl
from pathlib import Path

import networkx as nx
import pandas as pd
from scipy import sparse as sp
import numpy as np

from dev.preprocess.utils import aa_to_index, fetch_prot_seq


def extract_variant_graph(uprot_pos, chain, chain_res_list, seq2struct_pos, prot_graph, feat_data, feat_stats,
                          coev_feat_df, patch_radius=None, normalize=False):
    """
    Extract variant sub-graph from protein structure graph (target site at position 0)
    Args:
        uprot_pos:
        chain:
        chain_res_list: struct-residue position sorted by node index
        seq2struct_pos:  sequence position to struct-residue position
        prot_graph:
        feat_data:
        feat_stats:
        coev_feat_df:
        patch_radius:
        normalize:

    Returns:

    """
    # struct_id = record['PDB'] if model == 'PDB' else record['UniProt']
    # uprot = record['UniProt']
    # chain = record['Chain'] if model == 'PDB' else 'A'
    # uprot_pos = record['Protein_position']
    struct_pos = seq2struct_pos[uprot_pos]
    struct2seq_pos = {':'.join([chain, v]): k for k, v in seq2struct_pos.items()}
    chain_pos_idx = ':'.join([chain, str(struct_pos)])
    try:
        node_idx = chain_res_list.index(chain_pos_idx)  # map raw position to position index in structure
    except ValueError:
        return dgl.graph([]), None

    coords = prot_graph.ndata['coords']
    center = coords[node_idx]
    pdist = torch.nn.PairwiseDistance(p=2)
    dists = pdist(coords, center)
    nodes = [node_idx]
    if patch_radius:
        selected_idx = torch.where(dists <= patch_radius)[0]
        selected_idx = selected_idx[selected_idx != node_idx].tolist()
        nodes.extend(selected_idx)
        # nodes = torch.cat([torch.tensor([node_idx]), selected_idx[selected_idx != node_idx]])
    else:
        nodes.extend(prot_graph.predecessors(node_idx).tolist())
        # nodes = torch.cat([torch.tensor([node_idx]), prot_graph.predecessors(node_idx)])

    res_in_vargraph = [chain_res_list[i] for i in nodes]
    res_mappable = set(':'.join([chain, i]) for i in seq2struct_pos.values())  # E.g. 'A:10'
    node_remain = []
    seq_pos_remain = []
    res_remain = []
    for res in res_in_vargraph:
        if res in res_mappable:
            pos = struct2seq_pos[res]
            seq_pos_remain.append(pos)
            res_remain.append(res)
            node_remain.append(chain_res_list.index(':'.join([chain, seq2struct_pos[pos]])))

    var_graph = dgl.node_subgraph(prot_graph, node_remain)
    if isinstance(feat_data, pd.DataFrame):
        feat_data = feat_data.values

    src_raw, dst_raw = tuple(map(lambda x: x.tolist(), prot_graph.find_edges(var_graph.edata['_ID'])))

    n_edges = var_graph.num_edges()
    coev_feats = []
    coev_feat_df = coev_feat_df.loc[coev_feat_df['pos1'].isin(seq_pos_remain) & coev_feat_df['pos2'].isin(seq_pos_remain)]
    coev_feat_dict = dict(zip(coev_feat_df['key'], coev_feat_df[['sca', 'mi', 'di']].values.tolist()))

    for i in range(n_edges):
        src_seq, dst_seq = tuple(map(lambda x: struct2seq_pos[chain_res_list[x]], [src_raw[i], dst_raw[i]]))
        try:
            coev_feats.append(coev_feat_dict[(src_seq, dst_seq)])
        except KeyError:
            coev_feats.append(coev_feat_dict[(dst_seq, src_seq)])
    angle_stats = {'mean': torch.nanmean(prot_graph.edata['angle']),
                   'min': torch.tensor(np.nanmin(prot_graph.edata['angle'].numpy())),
                   'max': torch.tensor(np.nanmax(prot_graph.edata['angle'].numpy()))}

    dist_stats = {'mean': torch.nanmean(prot_graph.edata['dist']),
                  'min': torch.tensor(np.nanmin(prot_graph.edata['dist'].numpy())),
                  'max': torch.tensor(np.nanmax(prot_graph.edata['dist'].numpy()))}

    feat_stats = {'mean': torch.tensor(np.nanmean(feat_data, axis=0)),
                  'min': torch.tensor(np.nanmin(feat_data, axis=0)),
                  'max': torch.tensor(np.nanmax(feat_data, axis=0))}

    var_feats = torch.tensor(feat_data[list(map(lambda x: x - 1, seq_pos_remain)), :])
    var_feats, feat_stats = impute_nan(var_feats, feat_stats)
    var_graph.edata['angle'], angle_stats = impute_nan(var_graph.edata['angle'], feat_stats=angle_stats)
    var_graph.edata['dist'], dist_stats = impute_nan(var_graph.edata['dist'], feat_stats=dist_stats)
    if normalize:
        var_graph.edata['angle'] = normalize_feat(var_graph.edata['angle'], feat_stats=angle_stats)
        var_graph.edata['dist'] = normalize_feat(var_graph.edata['dist'], feat_stats=dist_stats)
        var_feats = normalize_feat(var_feats, feat_stats=feat_stats)

    var_graph.ndata['feat'] = var_feats
    var_graph.edata['coev'] = torch.tensor(coev_feats)
    var_graph.edata['feat'] = torch.cat([var_graph.edata['angle'], var_graph.edata['dist'], var_graph.edata['coev']], dim=1)

    return var_graph, seq_pos_remain


def impute_nan(feat_raw, feat_stats=None):
    """
    Impute missing values with mean (1-dim feature only)
    Args:
        feat_raw:
        feat_stats:

    Returns:

    """
    # target_idx = []
    # for col in feat_cols:
    #     if col in target_cols:
    #         target_idx.append(feat_cols.index(col))
    # mean = feat_stats['mean'][target_idx]
    feat_raw = copy.deepcopy(feat_raw)
    if not feat_stats:
        feat_stats = dict()
        feat_stats['mean'] = torch.tensor(np.nanmean(feat_raw.numpy(), axis=0))
        feat_stats['min'] = torch.tensor(np.nanmin(feat_raw.numpy(), axis=0))
        feat_stats['max'] = torch.tensor(np.nanmax(feat_raw.numpy(), axis=0))
    # for key in feat_stats:
    #     feat_stats[key] = torch.tensor(feat_stats[key]).clone().detach()

    if torch.isnan(feat_raw).any():
        na_idx = torch.where(torch.isnan(feat_raw))
        feat_raw[na_idx] = torch.take(feat_stats['mean'], na_idx[1])

    return feat_raw, feat_stats


def normalize_feat(feat_raw, feat_stats):
    feat_raw = copy.deepcopy(feat_raw)
    return (feat_raw - feat_stats['min']) / (feat_stats['max'] - feat_stats['min'])


def load_pio_features(uprot, feat_path):
    # load pre-computed PIONEER features (.pkl file)

    feat_cols = ['expasy_ACCE', 'expasy_AREA', 'expasy_BULK', 'expasy_COMP',
                 'expasy_HPHO', 'expasy_POLA', 'expasy_TRAN', 'expasy_ACCE_norm',
                 'expasy_AREA_norm', 'expasy_BULK_norm', 'expasy_COMP_norm',
                 'expasy_HPHO_norm', 'expasy_POLA_norm', 'expasy_TRAN_norm', 'SS_H',
                 'SS_E', 'SS_C', 'SS_H_norm', 'SS_E_norm', 'SS_C_norm', 'ACC_B', 'ACC_M',
                 'ACC_E', 'ACC_B_norm', 'ACC_M_norm', 'ACC_E_norm', 'JS', 'JS_norm']

    with open(list(feat_path.glob('*{}*.pkl'.format(uprot)))[0], 'rb') as f:
        pair, pair_feat = pickle.load(f)
    idx = 0 if pair[0] == uprot else 1
    feat_data = pair_feat[idx][feat_cols].reset_index(drop=True)

    return feat_data


def load_expasy_feats(uprot, feat_root, feat_dir):
    if isinstance(feat_root, str):
        feat_root = Path(feat_root)

    with open(feat_root / feat_dir / f'{uprot}.pkl', 'rb') as f_pkl:
        feat_dict = pickle.load(f_pkl)

    return np.stack(list(feat_dict.values())).T


def load_pssm(uprot, feat_root, feat_dir, option='ori'):
    if isinstance(feat_root, str):
        feat_root = Path(feat_root)

    with open(feat_root / feat_dir / f'{uprot}_{option}.pkl', 'rb') as f_pkl:
        pssm = pickle.load(f_pkl)

    return pssm.values


def load_coev_feats(uprot, feat_root, sca_dir, dca_dir):
    if isinstance(feat_root, str):
        feat_root = Path(feat_root)

    df_sca = pd.read_table(feat_root / sca_dir / f'{uprot}.sca', sep='\t', names=['pos1', 'pos2', 'sca'])
    df_dca = pd.read_table(feat_root / dca_dir / f'{uprot}.dca', sep='\t', names=['pos1', 'pos2', 'mi', 'di'])

    df_coev = df_sca.merge(df_dca, on=['pos1', 'pos2'])
    df_coev['key'] = df_coev.apply(lambda x: (int(x['pos1']), int(x['pos2'])), axis=1)

    return df_coev


def add_coev_edges(df_coev, feat_names, nlargest=20, src_name='pos1', dst_name='pos2'):
    df_top = df_coev.sort_values(feat_names, ascending=False).groupby(src_name).head(nlargest)
    if not isinstance(feat_names, list):
        feat_names = [feat_names]
    g_coev = dgl.from_networkx(nx.from_pandas_edgelist(df_top, src_name, dst_name, edge_attr=True).to_directed(),
                               edge_attrs=feat_names)
    return g_coev


def load_nsp_feats(uprot, feat_root, exclude, rewrite_pkl=False):
    if isinstance(feat_root, str):
        feat_root = Path(feat_root)

    df_nsp = pd.read_pickle(feat_root / f'{uprot}.pkl')
    rename_dict = dict(zip(df_nsp.columns, [s.strip() for s in df_nsp.columns]))
    df_nsp = df_nsp.rename(columns=rename_dict)

    if rewrite_pkl:
        df_nsp.to_pickle(feat_root / f'{uprot}.pkl')

    return df_nsp.drop(columns=exclude).values


def load_cosmis_feats(uprot, feat_root, cols=['cosmis', 'cosmis_pvalue'], suffix='_cosmis.tsv'):
    if isinstance(feat_root, str):
        feat_root = Path(feat_root)

    if suffix.split('.')[-1] == 'pkl':
        df_cosmis = pd.read_pickle(feat_root / f'{uprot}{suffix}')
    else:
        df_cosmis = pd.read_table(feat_root / f'{uprot}{suffix}')

    return df_cosmis[cols].values


def load_oe_feats(uprot, feat_root, cols=['obs_exp_mean', 'obs_exp_max']):
    if isinstance(feat_root, str):
        feat_root = Path(feat_root)
    
    df_oe = pd.read_csv(feat_root / f'{uprot}_oe.csv')

    return df_oe[cols].values


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

# Adapted from Dwivedi, Vijay Prakash, and Xavier Bresson.
# "A generalization of transformer networks to graphs." https://arxiv.org/abs/2012.09699

def laplacian_positional_encoding(g, pos_enc_dim):
    """
    Graph positional encoding v/ Laplacian eigenvectors
    Args:
        g: input graph
        pos_enc_dim: position encoding dimension

    Returns:

    """
    # Laplacian
    A = g.adjacency_matrix(scipy_fmt='csr').astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    if g.number_of_nodes() - 1 <= pos_enc_dim:
        return None
    # Eigenvectors with scipy
    try:
        EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim + 1, which='SR', tol=1e-2)  # for 40 PEs
        EigVec = EigVec[:, EigVal.argsort()]  # increasing order
    except:
        return None
    # g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1].real).float()

    return torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1].real)


def wl_positional_encoding(g):
    """
    WL-based absolute positional embedding
    """
    max_iter = 2
    node_color_dict = {}
    node_neighbor_dict = {}

    edge_list = torch.nonzero(g.adj().to_dense() != 0, as_tuple=False).numpy()
    node_list = g.nodes().numpy()

    # setting init
    for node in node_list:
        node_color_dict[node] = 1
        node_neighbor_dict[node] = {}

    for pair in edge_list:
        u1, u2 = pair
        if u1 not in node_neighbor_dict:
            node_neighbor_dict[u1] = {}
        if u2 not in node_neighbor_dict:
            node_neighbor_dict[u2] = {}
        node_neighbor_dict[u1][u2] = 1
        node_neighbor_dict[u2][u1] = 1

    # WL recursion
    iteration_count = 1
    exit_flag = False
    while not exit_flag:
        new_color_dict = {}
        for node in node_list:
            neighbors = node_neighbor_dict[node]
            neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
            color_string_list = [str(node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
            color_string = "_".join(color_string_list)
            hash_object = hashlib.md5(color_string.encode())
            hashing = hash_object.hexdigest()
            new_color_dict[node] = hashing
        color_index_dict = {k: v + 1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
        for node in new_color_dict:
            new_color_dict[node] = color_index_dict[new_color_dict[node]]
        if node_color_dict == new_color_dict or iteration_count == max_iter:
            exit_flag = True
        else:
            node_color_dict = new_color_dict
        iteration_count += 1

    return torch.LongTensor(list(node_color_dict.values()))


def fetch_struct_edges(start, end, chain, struct_graph, chain_res_list, seq2struct_pos):
    # struct2seq_pos = {':'.join([chain, v]): k for k, v in seq2struct_pos.items()}
    nodes_mapped = []
    for pos in range(start, end+1):
        if pos in seq2struct_pos:
            struct_pos = seq2struct_pos[pos]
            struct_node_idx = chain_res_list.index(f'{chain}:{struct_pos}')
            nodes_mapped.append(struct_node_idx)
            # nodes_mapped.extend(struct_graph.predecessors(struct_node_idx).tolist())
            # nodes = [struct_node_idx] + struct_graph.predecessors(struct_node_idx).tolist()
    nodes_mapped = list(set(nodes_mapped))
    subgraph = dgl.node_subgraph(struct_graph, nodes_mapped)

    return subgraph

    # struct_src, struct_dst = struct_graph.find_edges(subgraph.edata['_ID'])
    # src_mapped = []
    # dst_mapped = []

    # struct_edges = list(set(zip(struct_src, struct_dst)))
    # struct_src = torch.cat(struct_src)
    # struct_dst = torch.cat(struct_dst)
    # n_edges = subgraph.num_edges()
    # nodes_all = torch.cat([struct_src, struct_dst]).unique().tolist()
    #
    # for node_idx in nodes_all:
    #     seq_pos = struct2seq_pos[chain_res_list[node_idx]]
    # Structural nodes on sequence basis
    # TODO: map node & edge features
    # edge_remain_idx = []
    # for i in range(n_edges):
    #     try:
    #         src_seq_pos = struct2seq_pos[chain_res_list[struct_src[i].item()]]
    #         dst_seq_pos = struct2seq_pos[chain_res_list[struct_dst[i].item()]]
    #         # if src_seq_pos >= start and dst_seq_pos <= end:  # only keep residues within sequential neighborhood?
    #         if src_seq_pos in range(start, end+1) and dst_seq_pos in range(start, end+1):
    #             src_mapped.append(src_seq_pos - start)
    #             dst_mapped.append(dst_seq_pos - start)
    #             edge_remain_idx.append(i)
    #     except KeyError:  # structural residue not mappable to sequence
    #         continue

    # angle = subgraph.edata['angle'][edge_remain_idx]
    # dist = subgraph.edata['dist'][edge_remain_idx]
    # return src_mapped, dst_mapped, angle, dist
