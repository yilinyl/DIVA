
import os
from utils import *


def compile_features(input_file, g_data_dir):
    df_var = pd.read_csv(input_file)
    uprot_all = df_var['UniProt'].drop_duplicates().values
    # prepare protein sequence dict
    # seq_all_dict = parse_fasta(uprot_fasta)  # UniProt ID => sequence
    for uprot in uprot_all:
        df_prot = df_var.query('UniProt == @uprot')


def load_prot_graph(record, g_data_dir, **kwargs):
    model = record['model']
    struct_id = record['PDB'] if model == 'PDB' else record['UniProt']
    f_graph = os.path.join(g_data_dir, '')  # path to pre-constructed protein graph
    if os.path.exists(f_graph):
        with open(f_graph, 'rb') as f_pkl:
            prot_graph, res_idx_dict = pickle.load(f_pkl)
    else:
        prot_graph, res_idx_dict = build_single_graph(record, **kwargs)  # TODO: check arguments
    # f_graph = '{}_{}_graph.gml.gz'.format(model, struct_id)
    # prot_graph = nx.read_gml(os.path.join(g_data_dir, f_graph))



    # if record['model'] == 'PDB':
    #     try:
    #         pdb_record = pdb_info_all.query('UniProt == @uprot & PDB == @struct_id').iloc[0]
    #     except IndexError:
    #         print('Cannot find PDB structure for variant: {}:{}:{}>{}'.format(uprot, record['Protein_position'],
    #                                                                           record['REF_AA'], record['ALT_AA']))
    #         return
    #     pos_map = uprot2pdb_pos_map(pdb_record['MappableUniprotResidues'], pdb_record['MappablePDBResidues'])
    #     var_info = var_info.loc[var_info['PDB'] == record['PDB']].reset_index(drop=True)


    # TODO: assign AA to protein structure (REF from PDB file; ALT from var_info)
