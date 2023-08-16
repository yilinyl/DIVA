import os, sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd


if __name__ == '__main__':
    cosmis_data_root = Path('/fs/cbsuhyfs1/storage1/yl986/data')
    data_root = Path('/local/storage/yl986/3d_vip/data_prepare/data/dataset/full_v1')
    fname_in = 'balanced_full.csv'

    cosmis_raw_path = cosmis_data_root / 'cosmis_scores_raw'
    out_dir = cosmis_data_root / 'cosmis_scores'
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    df_var = pd.read_csv(data_root / fname_in)
    cov_thres = 0.5

    df_var['model'], df_var['struct_id'] = zip(*df_var.apply(lambda x: ('PDB', x['PDB']) if x['PDB_coverage'] >= cov_thres 
                                                             else ('AF', x['UniProt']), axis=1))
    uprot2len = dict(zip(df_var['UniProt'], df_var['prot_length']))
    df_struct = df_var[['UniProt', 'model', 'struct_id', 'PDB_coverage', 'prot_length']].drop_duplicates().\
        sort_values(['UniProt', 'PDB_coverage'], ascending=False).reset_index(drop=True)

    for uprot in tqdm(uprot2len.keys()):
        uprot_records = df_struct.query('UniProt == @uprot')
        uprot2pos = pd.DataFrame({'uniprot_id': uprot, 'uniprot_pos': list(range(1, uprot2len[uprot]+1))})
        out_path = out_dir / f'{uprot}_cosmis.tsv'
        if out_path.exists():
            continue
        cosmis_all = []
        for i, record in uprot_records.iterrows():
            model = record['model']
            struct_id = record['struct_id']
            if model == 'PDB':
                fs = cosmis_raw_path / f'{model}-{struct_id}_{uprot}_cosmis.tsv'
                coverage = record['PDB_coverage']
            else:
                fs = cosmis_raw_path / f'{model}-{struct_id}_cosmis.tsv'
                coverage = 0  # no PDB coverage
            if not fs.exists():
                print(f'{fs} not avilable')
                continue
            df_tmp = pd.read_table(fs)
            if df_tmp.shape[0] == 0:
                print(f'Empty COSMIS data for {uprot}')
                continue
            df_tmp['PDB_coverage'] = coverage
            cosmis_all.append(df_tmp)
        if not cosmis_all:
            print('No COSMIS score for {}'.format(uprot))
            continue
        df_cosmis = pd.concat(cosmis_all).sort_values(['uniprot_pos', 'PDB_coverage'], ascending=[True, False])
        df_cosmis['cosmis'] = (df_cosmis['cs_mis_obs'] - df_cosmis['mis_pmt_mean']) / df_cosmis['mis_pmt_sd']
        cols = ['uniprot_id', 'uniprot_pos', 'cosmis', 'mis_p_value', 'mis_pmt_mean', 'mis_pmt_sd', "cs_gc_content", 
                # "cs_syn_prob", "cs_syn_obs", "syn_pmt_mean", "syn_pmt_sd", "syn_p_value", 
                'plddt']
        cosmis_raw = df_cosmis[cols].drop_duplicates(['uniprot_pos']).rename({'cosmis': 'cosmis_raw'}, axis=1)
        cosmis_raw['plddt'] = cosmis_raw['plddt'] / 100

        df_cosmis_agg = df_cosmis.groupby(['uniprot_id', 'uniprot_pos'])['cosmis'].\
                            agg(['min', 'mean', 'max']).reset_index()
        df_cosmis_agg = df_cosmis_agg.merge(cosmis_raw, how='left').rename(columns={'min': 'cosmis_min', 
                                                                                    'mean': 'cosmis_mean', 
                                                                                    'max': 'cosmis_max'})
        df_cosmis_agg = uprot2pos.merge(df_cosmis_agg, how='left').drop('uniprot_id', axis=1).rename(columns={'uniprot_pos': 'Protein_position'})
        df_cosmis_agg = df_cosmis_agg.fillna(df_cosmis_agg.mean())
        
        df_cosmis_agg.to_csv(out_path, index=False, sep='\t')
