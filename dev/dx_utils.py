import os
import re
import gzip
import itertools
import numpy as np

from scipy.spatial.distance import pdist, squareform

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
    
def get_interaction(file):
    interaction_set = set()
    with open(file, 'r') as f:
        for line in f:
            interaction_set.add(tuple(sorted(line.strip().split('\t'))))
    return interaction_set


def load_featarray(feature_file):
    feature_list = []
    with open(feature_file, 'r') as feature_f:
        for line in feature_f:
            columns = line.split('\t')
            columns = [i.strip() for i in columns]
            feature_list.append(columns)
    return np.array(feature_list)


def fill_feat_nan(feat_array, overall_mean):
    col_mean = np.nanmean(feat_array, axis=0)
    inds = np.where(np.isnan(feat_array))
    feat_array[inds] = np.take(col_mean, inds[1])
    if np.any(np.isnan(feat_array)):
        inds = np.where(np.isnan(feat_array))
        feat_array[inds] = np.take(overall_mean, inds[1])
    return feat_array


def min_max_normalize(feat_array, min_array=None, max_array=None):
    if min_array is None:
        normalized_array = feat_array - np.min(feat_array, axis=0)
    else:
        normalized_array = feat_array - min_array
    if max_array is None:
        normalized_array = normalized_array / np.max(normalized_array, axis=0)
    else:
        normalized_array = normalized_array / max_array
    return normalized_array


def pdb_txt2array(pdb_file):
    with open(pdb_file, 'r') as infile:
        lines = infile.readlines()
        
    pdb_info = []
    for line in lines:
        line = line.strip()
        line_list = [line[0:5], line[6:11], line[12:16], line[16], line[17:20], line[21], line[22:27], line[30:38], line[38:46], line[46:54]]
        pdb_info.append([i.strip() for i in line_list])
        
    return np.array(pdb_info, dtype='str')


def load_pdbarray(interaction, prot, model_fname, pdb_dir, models_to_use):
    for model in models_to_use['_'.join(interaction)][prot]:
        if model[3] == model_fname:
            break
    modelres = unzip_res_range(model[2]) # Residue numbers on PDB basis.
    
    atom_list = []
    with gzip.open(os.path.join(pdb_dir, model_fname), 'rb') as f:
        file_content = f.read().decode('utf-8')
    for line in file_content.split('\n'):
        if not (line.startswith('ATOM') or line.startswith('HETATM')):
            if line.startswith('ENDMDL'):
                break # Only take the first model.
            else:
                continue
        if model[0] == 'PDB' and line[21] != model[5]: # Chain names must match.
            continue
        if line[22:27].strip() not in modelres: # Residue + iCode must be one that has a corresponding UniProt residue.
            continue
        columns = [line[0:5], line[6:11], line[12:16], line[16], line[17:20], line[21], line[22:27], line[30:38], line[38:46], line[46:54]]
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
    atom_res_array = pdb_array[:,6]
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


def get_distance_matrix(pdb_array, residue_index, distance_type):
    """
    Calculate pairwise distances between residues
    :param pdb_array:
    :param residue_index: array of residue index boundaries
    :param distance_type: 'atoms_average' or 'centroid'
    :return:
    """
    if distance_type == 'atoms_average':
        full_atom_dist = squareform(pdist(pdb_array[:, 7:10].astype(float)))  # coordinates
        residue_dm = np.zeros((residue_index.shape[0], residue_index.shape[0]))
        # pairwise distance across all residues
        for i, j in itertools.combinations(range(residue_index.shape[0]), 2):
            index_i = residue_index[i]
            index_j = residue_index[j]
            # average of pairwise distances between all atoms in both residues
            distance_ij = np.mean(full_atom_dist[index_i[0]:index_i[1]+1, index_j[0]:index_j[1]+1])
            residue_dm[i][j] = distance_ij
            residue_dm[j][i] = distance_ij
        
    elif distance_type == 'centroid':
        coord_array = np.empty((residue_index.shape[0], 3))
        for i in range(residue_index.shape[0]):
            res_start, res_end = residue_index[i]
            # average coordinates for each residue
            coord_i = pdb_array[:,7:10][res_start:res_end+1].astype(np.float)
            coord_array[i] = np.mean(coord_i, axis=0)
        residue_dm = squareform(pdist(coord_array))
    
    else:
        raise ValueError('Invalid distance type: %s' % distance_type)
    return residue_dm


def get_neighbor_index(residue_dm, num_neighbors):
    return residue_dm.argsort()[:, 1:num_neighbors+1]


def get_normal(acid_plane):
    cp = np.cross(acid_plane[2] - acid_plane[1], acid_plane[0] - acid_plane[1])
    if np.all(cp == 0):
        return np.array([np.nan] * 3)
    normal = cp/np.linalg.norm(cp,2)
    return normal


def get_neighbor_angle(pdb_array, residue_index, neighbor_index):
    normal_vector_array = np.empty((neighbor_index.shape[0], 3), dtype=np.float)
    for i, (res_start, res_end) in enumerate(residue_index):
        res_info = pdb_array[res_start:res_end+1]
        res_acid_plane_index = np.where(np.logical_and(np.isin(res_info[:,2], ['CA', 'C', 'O']),
                                                       np.isin(res_info[:,3], ['', 'A'])))
        res_acid_plane = res_info[res_acid_plane_index][:,7:10].astype(np.float)
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
    
    angle_matrix = np.empty_like(neighbor_index, dtype=np.float)
    for i, index in enumerate(neighbor_index):
        angle_matrix[i] = pairwise_angle[i, index]
    
    angle_matrix = fill_nan_mean(angle_matrix, axis=0)
    return angle_matrix


def get_edge_data(residue_dm, neighbor_index, neighbor_angle):
    edge_matrix = np.zeros((neighbor_index.shape[0], neighbor_index.shape[1], 2))
    for i, dist in enumerate(residue_dm):
        edge_matrix[i][:,0] = dist[neighbor_index[i]]
        edge_matrix[i][:,1] = neighbor_angle[i]
    return edge_matrix


def get_edge_coo_data(pdb_array, num_neighbors, distance_type='atoms_average'):
    residue_index = get_residue_info(pdb_array)
    residue_dm = get_distance_matrix(pdb_array, residue_index, distance_type)
    neighbor_index = get_neighbor_index(residue_dm, num_neighbors)
    
    source = np.reshape(neighbor_index, (-1, 1)).squeeze(axis=1)
    target = np.repeat(np.arange(residue_index.shape[0]), neighbor_index.shape[1])
    edge_index = np.stack([source, target])
    return edge_index

