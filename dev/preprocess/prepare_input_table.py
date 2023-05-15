import os
import sys

# sys.path.append('..')
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.abspath('..')))

import pandas as pd
from utils import unzip_res_range, map_to_pdb, parse_fasta


def uprot2pdb_pos(record):
    pos_dict = dict(zip(record['MappableUniprotResidues'], record['MappablePDBResidues']))
    return pos_dict[str(record['Position'])]


def compile_input_table(df_raw, pdb_sift_path, fasta_file):
    pdb_data = pd.read_csv(pdb_sift_path, sep='\t').dropna().reset_index(drop=True)
    pdb_data["MappableUniprotResidues"] = pdb_data["MappableResInPDBChainOnUniprotBasis"].apply(lambda x: unzip_res_range(x))
    pdb_data["MappablePDBResidues"] = pdb_data["MappableResInPDBChainOnPDBBasis"].apply(lambda x: unzip_res_range(x))
    pdb_data["AllPDBResidues"] = pdb_data["AllResInPDBChainOnPDBBasis"].apply(lambda x: unzip_res_range(x))
    pdb_data['pdb_chain_length'] = pdb_data['AllPDBResidues'].apply(len)
    pdb_data['n_mapped'] = pdb_data.apply(lambda x: len(set(x['MappableUniprotResidues'])), axis=1)
    # uniprots_in_pdb = set(pdb_data["UniProt"])

    df_raw['prot_var_id'] = df_raw.apply(lambda x: '{}_{}_{}/{}'.format(x['UniProt'], x['Protein_position'], 
                                                                        x['REF_AA'], x['ALT_AA']), axis=1)
    if 'prot_length' not in df_raw.columns:
        uprot2seq_dict = parse_fasta(fasta_file)
        df_raw['prot_length'] = df_raw['UniProt'].apply(lambda x: len(uprot2seq_dict.get(x, '')))

    uprot_pdb = map_to_pdb(pdb_data, df_raw)
    uprot_pdb_merge = uprot_pdb.merge(pdb_data, how='left')
    uprot_pdb_merge['PDB_position'] = uprot_pdb_merge.apply(uprot2pdb_pos, axis=1)
    uprot_pdb_merge['prot_length'] = uprot_pdb_merge['UniProt'].apply(lambda x: len(uprot2seq_dict.get(x, '')))
    # uprot_pdb_merge['coverage'] = uprot_pdb_merge['n_mapped'] / uprot_pdb_merge['prot_length']

    cols = ['UniProt', 'Position', 'PDB', 'Chain', 'PDB_position', 'n_mapped', 'prot_length']
    df_merge = df_raw.merge(uprot_pdb_merge[cols].rename(columns={'Position': 'Protein_position'}), how='left')
    df_merge['PDB_coverage'] = df_merge['n_mapped'] / df_merge['prot_length']

    return df_merge


if __name__ == '__main__':
    pdb_sift_path = '/local/storage/yl986/data/pdb/sifts_20220526/pdbresiduemapping.txt'
    fasta_file = '/fs/cbsuhyfs1/storage/dx38/local_resource/uniprot_data_20220526/uniprot_sprot.fasta'
    test_file = sys.argv[1]
    outfile = sys.argv[2]

    df_in = pd.read_csv(test_file)

    df_prep = compile_input_table(df_in, pdb_sift_path, fasta_file)

    df_prep.to_csv(outfile, index=False)
    

