import copy
import pickle, hashlib, torch, dgl
from pathlib import Path
from scipy import sparse as sp
import numpy as np

from dev.preprocess.utils import aa_to_index, fetch_prot_seq


def extract_variant_graph(uprot_pos, chain, chain_res_list, seq2struct_pos, prot_graph, feat_data, feat_stats, patch_radius=None):
    """
    Extract variant sub-graph from protein structure graph (target site at position 0)
    Args:
        uprot_pos:
        chain:
        chain_res_list:
        seq2struct_pos:
        prot_graph:
        feat_data:
        feat_stats:
        patch_radius:

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
    var_feats = torch.tensor(feat_data.iloc[map(lambda x: x - 1, seq_pos_remain), :].values)
    var_feats = normalize_data(var_feats, feat_stats)
    var_graph.ndata['feat'] = var_feats
    var_graph.edata['angle'] = normalize_data(var_graph.edata['angle'])
    var_graph.edata['dist'] = normalize_data(var_graph.edata['dist'])
    return var_graph, seq_pos_remain


def normalize_data(feat_raw, feat_stats=None):
    # target_idx = []
    # for col in feat_cols:
    #     if col in target_cols:
    #         target_idx.append(feat_cols.index(col))
    # mean = feat_stats['mean'][target_idx]
    feat_raw = copy.deepcopy(feat_raw)
    if not feat_stats:
        feat_stats = dict()
        feat_stats['mean'] = torch.tensor(np.nanmean(feat_raw.numpy()))
        feat_stats['min'] = torch.tensor(np.nanmin(feat_raw.numpy()))
        feat_stats['max'] = torch.tensor(np.nanmax(feat_raw.numpy()))
    # for key in feat_stats:
    #     feat_stats[key] = torch.tensor(feat_stats[key]).clone().detach()

    if torch.isnan(feat_raw).any():
        na_idx = torch.where(torch.isnan(feat_raw))
        feat_raw[na_idx] = torch.take(feat_stats['mean'], na_idx[1])

    return (feat_raw - feat_stats['min']) / (feat_stats['max'] - feat_stats['min'])


def load_features(uprot, feat_path):

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
    except TypeError:
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
    struct2seq_pos = {':'.join([chain, v]): k for k, v in seq2struct_pos.items()}
    nodes_mapped = []
    for pos in range(start, end+1):
        if pos in seq2struct_pos:
            struct_pos = seq2struct_pos[pos]
            struct_node_idx = chain_res_list.index(f'{chain}:{struct_pos}')
            nodes_mapped.append(struct_node_idx)
            nodes_mapped.extend(struct_graph.predecessors(struct_node_idx).tolist())
            # nodes = [struct_node_idx] + struct_graph.predecessors(struct_node_idx).tolist()
    nodes_mapped = list(set(nodes_mapped))
    subgraph = dgl.node_subgraph(struct_graph, nodes_mapped)
    struct_src, struct_dst = struct_graph.find_edges(subgraph.edata['_ID'])
    src_mapped = []
    dst_mapped = []

    # struct_edges = list(set(zip(struct_src, struct_dst)))
    # struct_src = torch.cat(struct_src)
    # struct_dst = torch.cat(struct_dst)
    n_edges = subgraph.num_edges()
    # nodes_all = torch.cat([struct_src, struct_dst]).unique().tolist()
    #
    # for node_idx in nodes_all:
    #     seq_pos = struct2seq_pos[chain_res_list[node_idx]]
    # Structural nodes on sequence basis
    # TODO: map node & edge features
    edge_remain_idx = []
    for i in range(n_edges):
        try:
            src_seq_pos = struct2seq_pos[chain_res_list[struct_src[i].item()]]
            dst_seq_pos = struct2seq_pos[chain_res_list[struct_dst[i].item()]]
            # if src_seq_pos >= start and dst_seq_pos <= end:  # only keep residues within sequential neighborhood?
            if src_seq_pos in range(start, end+1) and dst_seq_pos in range(start, end+1):
                src_mapped.append(src_seq_pos - start)
                dst_mapped.append(dst_seq_pos - start)
                edge_remain_idx.append(i)
        except KeyError:  # structural residue not mappable to sequence
            continue

    angle = subgraph.edata['angle'][edge_remain_idx]
    dist = subgraph.edata['dist'][edge_remain_idx]
    return src_mapped, dst_mapped, angle, dist
