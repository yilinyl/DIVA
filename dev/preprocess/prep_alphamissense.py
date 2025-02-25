import os, sys
from pathlib import Path
import pandas as pd
import logging
import pickle
import json


def create_afmis_dict(afmis_fpath, out_path,
                      vid_col='prot_var_id', 
                      pid_col='uniprot_id', 
                      pos_col='Protein_position',
                      ref_col='REF_AA', alt_col='ALT_AA',
                      score_col='am_pathogenicity',
                      skiprows=0,
                      chunksize=100000):
    afmis_dict = dict()
    chunks = pd.read_csv(afmis_fpath, sep='\t', chunksize=chunksize, skiprows=skiprows)
    n_processed = 0
    prot_all = set()
    for i, chunk in enumerate(chunks):
        if vid_col not in chunk.columns:
            chunk[vid_col] = chunk.apply(lambda x: '{}_{}_{}/{}'.format(x[pid_col], x[pos_col], x[ref_col], x[alt_col]), axis=1)
        # Convert to dictionary
        chunk_dict = dict(zip(chunk[vid_col], chunk[score_col]))
        afmis_dict.update(chunk_dict)
        prot_all.update(chunk[pid_col].tolist())
        n_processed += chunk.shape[0]
        # Update progress
        logging.info(f"{n_processed} entries processed")
        
    logging.info('Save mapping to pickle file...')
    with open(out_path, 'wb') as f:
        pickle.dump(afmis_dict, f)
    return prot_all


def create_afmis_dict_from_raw(afmis_fpath, out_root,
                      vid_col='protein_variant', 
                      pid_col='uniprot_id', 
                      score_col='am_pathogenicity',
                      skiprows=0,
                      chunksize=100000):
    if isinstance(out_root, str):
        out_root = Path(out_root)
    if not out_root.exists():
        out_root.mkdir(parents=True)

    chunks = pd.read_csv(afmis_fpath, sep='\t', chunksize=chunksize, skiprows=skiprows)
    n_processed = 0
    prot_processed = set()
    for i, chunk in enumerate(chunks):
        chunk['prot_var_id'] = chunk.apply(lambda x: '{}_{}_{}/{}'.format(x[pid_col], x[vid_col][1:-1], x[vid_col][0], x[vid_col][-1]), axis=1)
        # Convert to dictionary
        prots = chunk[pid_col].unique()
        for pid in prots:
            out_fname = f'{pid}_sub_all.json'
            prot_var_dict = dict()
            if pid in prot_processed:
                with open(out_root / out_fname, 'r') as f:
                    prot_var_dict = json.load(f)
            cur_dict = dict(zip(chunk[chunk[pid_col] == pid]['prot_var_id'], chunk[chunk[pid_col] == pid][score_col]))
            prot_var_dict.update(cur_dict)
            with open(out_root / out_fname, 'w') as f:
                json.dump(prot_var_dict, f, indent=2)
            prot_processed.add(pid)

        n_processed += chunk.shape[0]
        # Update progress
        logging.info(f"{n_processed} entries processed")
    
    return prot_processed


if __name__ == '__main__':
    afmis_root = Path('/home/yl986/data/variant_data/AlphaMissense/')
    # afmis_file = '/home/yl986/data/variant_data/AlphaMissense/AlphaMissense_hg38_processed.tsv'
    afmis_file = afmis_root / 'AlphaMissense_aa_substitutions.tsv.gz'
    # afmis_file = '/home/yl986/data/variant_data/AlphaMissense/sample.tsv'
    # out_file = afmis_root / 'afmis_aa_substitutions.pkl'
    out_path = afmis_root / 'substitutions_by_prot'
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s: %(message)s')

    uniq_prots = create_afmis_dict_from_raw(afmis_file, out_path, skiprows=3)
    logging.info('Save list of proteins to file...')
    with open(afmis_root / 'afmis_substitutions_uprots.txt', 'w') as f:
        f.write('\n'.join(list(uniq_prots)))
    
    # logging.info('Save mapping to pickle file...')
    # with open(out_file, 'wb') as f:
    #     pickle.dump(afmis_dict, f)
    logging.info('Done!')