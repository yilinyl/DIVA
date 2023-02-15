import os
import sys
import re
import gzip

import itertools
import numpy as np
from pathlib import Path
import logging
import pandas as pd

import json
import urllib.request

import torch
import dgl
import pickle

from scipy.spatial.distance import pdist, squareform
# from pathlib import Path
# from tqdm import tqdm
from .supp_data import *


def unzip_res_range(res_range):
    res_ranges = res_range.strip()[1:-1].split(',')
    index_list = []
    for r in res_ranges:
        if re.match('.+-.+', r):
            a, b = r.split('-')
            index_list += [str(n) for n in range(int(a), int(b)+1)]
        else:
            index_list.append(r)

    if index_list == ['']:
        return []
    else:
        return index_list


def load_pdbarray(f_path):
    """
    Load PDB structure file to numpy array (modified from dapeng's script)
    reference: https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html
    :param f_path: path to co-crystal structure file
    :return:
    """
    atom_list = []
    # ext = f_path.name.split('.')[-1]
    if f_path.suffix == '.gz':
        with gzip.open(f_path, 'rb') as f:
            file_content = f.read().decode('utf-8')
    else:
        with open(f_path, 'r') as f:
            file_content = f.read()

    for line in file_content.split('\n'):
        if not (line.startswith('ATOM') or line.startswith('HETATM')):
            if line.startswith('ENDMDL'):
                break  # Only take the first model.
            else:
                continue
                # "ATOM", atom seriel num, atom name, alternate loc, residue, chain
        columns = [line[0:6], line[6:11], line[12:16], line[16], line[17:20], line[21],
                # residue sequence num, coord-x, coord-y, coord-z, element symbol
                   line[22:27], line[30:38], line[38:46], line[46:54], line[76:78]]
        columns = [col.strip() for col in columns]
        atom_list.append(columns)
    return np.array(atom_list)


def get_residue_info(pdb_array):
    """
    Get boundary indexes of each residue, both sides inclusive
        e.g. [0, 2] indicates the 1st residue occupies 3 atoms (indexes 0, 1, 2) in the structure file
    :param pdb_array: numpy array from `load_pdbarray`
    :return:
    """
    atom_res_array = pdb_array[:, 6]  # residue sequence num
    boundary_list = []
    start_pointer = 0
    curr_pointer = 0
    curr_atom = atom_res_array[0]

    # One pass through the list of residue numbers and record row number boundaries. Both sides inclusive.
    while curr_pointer < atom_res_array.shape[0] - 1:
        curr_pointer += 1
        if atom_res_array[curr_pointer] != curr_atom:
            boundary_list.append([start_pointer, curr_pointer - 1])
            start_pointer = curr_pointer
            curr_atom = atom_res_array[curr_pointer]
    boundary_list.append([start_pointer, atom_res_array.shape[0] - 1])
    return np.array(boundary_list)


def get_distance_matrix(pdb_array, residue_index, distance_type='centroid', coord_array=None, coord_option=None):
    """
    Calculate pairwise distances between residues
    Args:
        pdb_array:
        residue_index: array of residue index boundaries
        distance_type: 'atoms_average' or 'centroid'
        coord_array:
        coord_option:

    Returns:

    """

    if distance_type == 'atoms_average':
        full_atom_dist = squareform(pdist(pdb_array[:, 7:10].astype(float)))  # coordinates
        residue_dm = np.zeros((residue_index.shape[0], residue_index.shape[0]))
        # pairwise distance across all residues
        for i, j in itertools.combinations(range(residue_index.shape[0]), 2):
            index_i = residue_index[i]
            index_j = residue_index[j]
            # average of pairwise distances between all atoms in both residues
            distance_ij = np.mean(full_atom_dist[index_i[0]:index_i[1] + 1, index_j[0]:index_j[1] + 1])
            residue_dm[i][j] = distance_ij
            residue_dm[j][i] = distance_ij

    elif distance_type == 'centroid':
        try:
            residue_dm = squareform(pdist(coord_array))
        except ValueError:
            coord_array = get_center_coords(pdb_array, residue_index, coord_option)
            residue_dm = squareform(pdist(coord_array))
    else:
        raise ValueError('Invalid distance type: %s' % distance_type)

    return residue_dm


def get_center_coords(pdb_array, residue_index, option=None):
    n_residues = residue_index.shape[0]
    coord_array = np.empty((n_residues, 3))

    for i in range(residue_index.shape[0]):
        res_start, res_end = residue_index[i]
        # average coordinates for each residue
        coord_i = pdb_array[res_start:res_end + 1, 7:10].astype(float)
        n_atoms = res_end - res_start + 1
        atoms = pdb_array[res_start:res_end + 1, -1]
        atom_weights = np.ones(n_atoms)
        if option == 'mass':
            atom_weights = list(map(lambda x: atom_mass_dict[x], atoms))

        coord_array[i] = np.average(coord_i, axis=0, weights=atom_weights)

    return coord_array


def get_normal(acid_plane):
    """
    Compute plane normal
    Args:
        acid_plane:

    Returns:

    """
    cp = np.cross(acid_plane[2] - acid_plane[1], acid_plane[0] - acid_plane[1])
    if np.all(cp == 0):
        return np.array([np.nan] * 3)
    normal = cp / np.linalg.norm(cp, 2)
    return normal


def fill_nan_mean(array, axis=0):
    if axis not in [0, 1]:
        raise ValueError('Invalid axis: %s' % axis)
    mean_array = np.nanmean(array, axis=axis)
    inds = np.where(np.isnan(array))
    array[inds] = np.take(mean_array, inds[1-axis])
    if np.any(np.isnan(array)):
        full_array_mean = np.nanmean(array)
        inds = np.unique(np.where(np.isnan(array))[1-axis])
        if axis == 0:
            array[:,inds] = full_array_mean
        else:
            array[inds] = full_array_mean
    return array


def get_all_pairwise_angle(pdb_array, residue_index):
    """
    Compute pairwise angles for all residues
    Args:
        pdb_array:
        residue_index:

    Returns:

    """
    n_residue = residue_index.shape[0]
    normal_vector_array = np.empty((n_residue, 3), dtype=float)
    for i, (res_start, res_end) in enumerate(residue_index):
        res_info = pdb_array[res_start:res_end + 1]
        res_acid_plane_index = np.where(np.logical_and(np.isin(res_info[:, 2], ['CA', 'C', 'O']),
                                                       np.isin(res_info[:, 3], ['', 'A'])))
        res_acid_plane = res_info[res_acid_plane_index][:, 7:10].astype(float)
        if res_acid_plane.shape[0] != 3:
            normal_vector_array[i] = np.array([np.nan] * 3)
            continue
        normal_vector = get_normal(res_acid_plane)
        if np.all(np.isnan(normal_vector)):
            normal_vector_array[i] = np.array([np.nan] * 3)
        else:
            normal_vector_array[i] = normal_vector

    pairwise_normal_dot = normal_vector_array.dot(normal_vector_array.T)

    # Correct floating point precision error
    pairwise_normal_dot[pairwise_normal_dot > 1] = 1
    pairwise_normal_dot[pairwise_normal_dot < -1] = -1

    pairwise_angle = np.arccos(pairwise_normal_dot) / np.pi

    pairwise_angle = fill_nan_mean(pairwise_angle, axis=0)

    return pairwise_angle


def structure_to_graph(pdb_array, num_neighbors=10, distance_type='centroid',
                       method='knn', radius=10, coord_option='mass'):

    """
    Construct protein graph
    Args:
        pdb_array:
        num_neighbors: number of neighbors for k-nearest neighbor graph (default=10)
        distance_type: 'centroid' (default) or 'atoms_average'
        method:
            - 'knn' for k-nearest neighbor graph (need to specify `num_neighbors`)
            - 'radius' to connect all nodes within a fixed distance (need to specify `radius`)
        radius: two nodes are connected if their Euclidean distance is within the radius (default=10)
        coord_option
    Returns:
        protein graph (dgl.Graph)
    """
    residue_index = get_residue_info(pdb_array)
    residue_coords = get_center_coords(pdb_array, residue_index, option=coord_option)
    residue_dm = get_distance_matrix(pdb_array, residue_index, distance_type, residue_coords, coord_option)

    n_residue = residue_index.shape[0]
    if method == 'knn':
        neighbor_index = residue_dm.argsort()[:, 1:num_neighbors + 1]  # get closest residue neighbors

        source = np.reshape(neighbor_index, (-1, 1)).squeeze(axis=1)
        target = np.repeat(np.arange(n_residue), neighbor_index.shape[1])
        edge_index = np.stack([target, source])
    else:  # connect all nodes with a fixed radius
        edge_index = np.where(residue_dm <= radius)  # self-loops exist

    edge_index = np.stack(edge_index)[:, edge_index[0] != edge_index[1]]

    res_idx_array = np.stack(residue_index)
    res_names = pdb_array[res_idx_array[:, 0], 4]  # 3-letter residue name
    # res_name_dict = dict(zip(np.arange(n_residue), res_names))

    pairwise_angle = get_all_pairwise_angle(pdb_array, residue_index)
    edge_angles = []
    dist = []
    n_edges = edge_index.shape[1]
    for i in range(n_edges):
        u, v = edge_index[:, i]
        edge_angles.append(pairwise_angle[u, v])
        dist.append(residue_dm[u, v])

    graph = dgl.graph(data=(torch.tensor(edge_index[0]), torch.tensor(edge_index[1])))
    graph.edata['angle'] = torch.tensor(edge_angles).unsqueeze(1)
    graph.edata['dist'] = torch.tensor(dist).unsqueeze(1)
    graph.ndata['ref_aa'] = torch.tensor(list(map(aa_to_index, res_names)), dtype=torch.int64)
    graph.ndata['coords'] = torch.tensor(residue_coords)

    chain_res_id = list(map(lambda x: ':'.join([x[5], x[6]]), pdb_array[res_idx_array[:, 0], :]))
    # res_idx_dict = dict(zip(chain_res_id, np.arange(res_idx_array.shape[0])))

    # g = nx.Graph()
    # g.add_edges_from(tuple(zip(*edge_index)))
    # g.remove_edges_from(nx.selfloop_edges(g))
    # nx.set_node_attributes(g, res_name_dict, 'ref_aa')
    # nx.set_edge_attributes(g, edge_angles, name='angle')  # TODO: keep small molecules or not

    return graph, chain_res_id

def pick_struct_file(record, cov_thres=0.5):
    if record['PDB_coverage'] > cov_thres:
        pdb_id = record['PDB']
        fname = 'pdb{}.ent.gz'.format(pdb_id.lower())
        # f_path = os.path.join(pdb_root_dir, pdb_id.lower()[1:3], fname)
    else:
        # AF-A0A024R1R8-F1-model_v1.pdb.gz
        fname = 'AF-{}-F1-model_v1.pdb.gz'.format(record['UniProt'])
        # f_path = os.path.join(af_dir, fname)
    return fname


def uprot2pdb_pos_map(uprot_pos_range, pdb_pos_range):
    uprot_pos = unzip_res_range(uprot_pos_range)
    pdb_pos = unzip_res_range(pdb_pos_range)

    return dict(zip(uprot_pos, pdb_pos))


def uprot2pdb_pos(record):
    pos_dict = dict(zip(record['MappableUniprotResidues'], record['MappablePDBResidues']))
    return pos_dict[str(record['Position'])]


def build_single_graph(record, model, pdb_root_dir, af_root_dir, save_dir, num_neighbors=10,
                       distance_type='centroid', method='radius', radius=10, df_ires=None, save=False,
                       anno_ires=False, coord_option=None):
    uprot = record['UniProt']
    pdb_root_path = Path(pdb_root_dir)
    af_root_path = Path(af_root_dir)
    struct_available = True
    if model == 'PDB':
        struct_id = record['PDB']
        fname = 'pdb{}.ent.gz'.format(struct_id.lower())
        # f_path = os.path.join(pdb_root_dir, struct_id.lower()[1:3], fname)
        f_path = pdb_root_path / struct_id.lower()[1:3] / fname
        if not f_path.exists():
            struct_available = False
    else:
        struct_id = record['UniProt']
        af_files = list(af_root_path.glob('AF-{}*.pdb*'.format(struct_id)))

        if len(af_files) == 0:
            struct_available = False
        else:
            f_path = af_files[0]

    f_save = '{}_{}_graph.pkl'.format(model, struct_id)
    save_path = Path(save_dir) / f_save
    if save_path.exists():  # protein graph already constructed
        with open(save_path, 'rb') as f_pkl:
            prot_graph, chain_res_list = pickle.load(f_pkl)
        return prot_graph, chain_res_list

    if not struct_available:
        logging.warning('Structure {}-{} not available for UniProt {}'.format(model, struct_id, uprot))
        return

    # if model == 'PDB':
    #     pdb_record = pdb_info.query('UniProt == @uprot & PDB == @struct_id').iloc[0]
    #     pos_map = uprot2pdb_pos_map(pdb_record['MappableUniprotResidues'], pdb_record['MappablePDBResidues'])

    pdb_array = load_pdbarray(f_path)

    pdb_array = pdb_array[pdb_array[:, 0] == 'ATOM', :]  # only keep standard AA residues
    prot_graph, chain_res_list = structure_to_graph(pdb_array, num_neighbors, distance_type, method, radius, coord_option)

    n_residue = prot_graph.num_nodes()
    res_sorted = sorted(chain_res_list)
    if anno_ires:  # annotate interface information
        ires_ref = df_ires.query('PDB == @struct_id')
        ires_label = get_ires_label(res_sorted, ires_ref)
        prot_graph.ndata['ires'] = torch.tensor(ires_label, dtype=torch.int64)

    if save:
        with open(save_path, 'wb') as f_pkl:
            pickle.dump((prot_graph, chain_res_list), f_pkl)

    return prot_graph, chain_res_list


def get_ires_label(res_sorted, ires_ref):
    """
    Get binary interface label for each residue (1=interface, 0=non-interface)
    Args:
        res_sorted:
        ires_ref:

    Returns: list

    """
    n_residue = len(res_sorted)
    ires_label = []
    ires_all = set()
    for i, record in ires_ref.iterrows():
        n_parts = len({record['ChainA'], record['ChainB']})
        suffix = ['A', 'B']
        for j in range(n_parts):
            chain = record['Chain' + suffix[j]]
            ires_list = unzip_res_range(record['PDBIres' + suffix[j]])
            ires_all.update([':'.join([chain, pos]) for pos in ires_list])
    if not ires_all:
        ires_label = [0] * n_residue
    else:
        for res in res_sorted:
            if res in ires_all:
                ires_label.append(1)
            else:
                ires_label.append(0)
    return ires_label


# def build_graphs_all(var_df, pdb_root_dir, af_dir, save_dir, cov_thres=0.7, num_neighbors=10,
#                      distance_type='centroid', method='radius', radius=10, save=False):
#     # pdbres_map = pd.read_csv(pdbresmap_path)
#     # pdbres_map = pdbres_map.dropna().reset_index(drop=True)
#     # uniprots_in_pdb = set(pdb_data["UniProt"])
#     # pdb_info = var_df.merge(pdbres_map)
#     for i, record in var_df.iterrows():
#         prot_graph = build_single_graph(record, pdb_root_dir, af_dir, save_dir, cov_thres,
#                                         num_neighbors, distance_type, method, radius, save)
#         if not prot_graph:
#             continue


def parse_fasta(fasta_file):
    """
    (from Charles)
    Load a fasta file of sequences into a dictionary. Supports both *.fasta and *.fasta.gz.

    Args:
      fasta_file: str, path to the fasta file.

    Returns:
      A dictionary of sequences.
    """
    result_dict = {}
    if fasta_file.endswith('.gz'):
        zipped = True
        f = gzip.open(fasta_file, 'rb')
    else:
        zipped = False
        f = open(fasta_file, 'r')
    seq = ''
    for line in f:
        if zipped:
            is_header = line.startswith(b'>')
        else:
            is_header = line.startswith('>')
        if is_header:
            if seq:
                result_dict[identifier] = seq
            if zipped:
                identifier = line.decode('utf-8').split('|')[1]  # YL: SwissProt ID as key
            else:
                identifier = line.split('|')[1]
            seq = ''
        else:
            if zipped:
                seq += line.decode('utf-8').strip()
            else:
                seq += line.strip()

    result_dict[identifier] = seq
    f.close()
    return result_dict


def fetch_prot_seq(pid):
    """
    Fetch protein sequence from UniProt protal

    Args:
        pid: a valid protein UniProt ID
    Returns:
        organism name corresponds to the input protein
    """

    url = "https://rest.uniprot.org/uniprotkb/{pid}?format=json".format(pid=pid)

    with urllib.request.urlopen(url) as response:
        content = response.read().decode("utf-8")
        # info = re.findall('<name type="common">(.*?)</name>', content)
    js_dict = json.loads(content)
    return js_dict['sequence']['value']


def get_prot_length(uprot_all, uprot2seq_dict):
    # uprot_all = var_df['UniProt'].drop_duplicates().values
    # resource_root = '/local/storage/yl986/data/uniprot_data_20220526/'
    # uprot2seq_dict = parse_fasta(os.path.join(resource_root, 'uniprot_sprot.fasta'))
    # isoform2seq_dict = parse_fasta(os.path.join(resource_root, 'uniprot_sprot_varsplic.fasta'))
    # uprot2seq_dict.update(isoform2seq_dict)
    uprot_all = set(uprot_all)
    print('Unique protein IDs: {}'.format(len(uprot_all)))
    uprot2length = dict()
    trembl = set()
    for uprot in uprot_all:
        try:
            seq = uprot2seq_dict[uprot]
        except KeyError:
            seq = fetch_prot_seq(uprot)
            trembl.add(uprot)
        uprot2length[uprot] = len(seq)

    return uprot2length

def aa_to_index(aa: str):
    """
    Encode amino acid to numerical values (prepare for one-hot encoding)
    """
    aa_index = {'ALA': 0,
                'CYS': 1,
                'ASP': 2,
                'GLU': 3,
                'PHE': 4,
                'GLY': 5,
                'HIS': 6,
                'ILE': 7,
                'LYS': 8,
                'LEU': 9,
                'MET': 10,
                'ASN': 11,
                'PRO': 12,
                'GLN': 13,
                'ARG': 14,
                'SER': 15,
                'THR': 16,
                'VAL': 17,
                'TRP': 18,
                'TYR': 19,
                'ASX': 20,
                'XAA': 20,
                'GLX': 20,
                'XLE': 20,
                'SEC': 20,
                'PYL': 20}

    return aa_index.get(aa.upper(), 20)


def calculate_expasy(seq, expasy_dict):
    """
    Calculate ExPaSy features (modified from PIONEER script)

    Args:
        seq (str): protein primary sequence
        expasy_dict (dict): ExPasy scales
    Returns:
        biochemical feature matrix (n_residues x 7)
    """

    # Calculate ExPaSy features
    feat_vec = []
    for feat in ['ACCE', 'AREA', 'BULK', 'COMP', 'HPHO', 'POLA', 'TRAN']:
        feat_vec.append(np.array([expasy_dict[feat][x] if x in expasy_dict[feat] else 0 for x in seq]))
    expasy_feat = np.column_stack(feat_vec)
    return expasy_feat


