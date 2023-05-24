import os, sys
from pathlib import Path
import pickle
import gzip
import json
import requests
import logging

import numpy as np
import pandas as pd

from pyensembl import EnsemblRelease

sys.path.append(os.path.dirname(os.path.abspath('..')))
sys.path.append(os.path.dirname(os.path.abspath('.')))
from dev.preprocess.utils import parse_fasta, fetch_prot_seq


def expand_pos_range(pos_ranges):
    pos_expand = set()
    for seg in pos_ranges:
        pos_expand.update(list(range(seg[0], seg[1]+1)))
    return sorted(list(pos_expand))


def fetch_oe_data(uprot_all, uprot2enst_dict, oe_raw_records, ensembl_db, prot_seq_dict):
    oe_list = []

    for uprot in uprot_all:
        if uprot not in uprot2enst_dict:  # No Ensembl transcript ID
            logging.warning('No Ensembl entry for UniProt {}'.format(uprot))
            continue
        enst_ids = uprot2enst_dict[uprot]
        # ts_all = list(map(lambda x: ensembl_db.transcript_by_id(x), enst_ids))
        ts_all = []
        for eid in enst_ids:
            try:
                ts_all.append(ensembl_db.transcript_by_id(eid))
            except ValueError:
                continue
        
        # ts = ensembl_db.transcript_by_id(enst_ids[0])  # TODO: which ensembl ID to use if there are multiple mappings?
        
        # if not prot_seq_dict:
        #     aa_seq = ts_all[0].protein_sequence
        # else:
        # try:
        #     aa_seq = prot_seq_dict[uprot]
        # except KeyError:
        #     aa_seq = fetch_prot_seq(uprot)
        # prot_length = len(aa_seq)
        
        oe_prot = []
        candidates = oe_raw_records[oe_raw_records['transcript'].isin(enst_ids)]
        for i, ts in enumerate(ts_all):
            pos_ranges = sorted(ts.coding_sequence_position_ranges)
            pos_range_combine = expand_pos_range(pos_ranges)
            if not ts.complete:
                continue
            prot_length = len(ts.protein_sequence)
            if len(pos_range_combine) // 3 == prot_length:
                # valid_ts_all.append(ts)
                # break
                if ts.strand == '-':
                    pos_range_combine = pos_range_combine[::-1]
                    
                # oe_records = candidates[candidates['transcript'].isin([ts.id])].dropna(subset=['pos_ext_hg38'])
                oe_records = candidates.query('transcript == "{}"'.format(ts.id))

                oe_cur = oe_records[oe_records['pos_ext_hg38'].isin(pos_range_combine)].\
                            drop(columns=['pos_ext_hg19', 'add_to_hg38', 'pos']).rename(columns={'pos_ext_hg38': 'pos_hg38'}).\
                            sort_values('pos_hg38').astype({'pos_hg38': int}).reset_index(drop=True)
                oe_cur['UniProt'] = uprot
                oe_cur['prot_length'] = prot_length
                coord_map = {pos: i // 3 + 1 for i, pos in enumerate(pos_range_combine)}

                oe_cur['Protein_position'] = oe_cur['pos_hg38'].apply(lambda x: coord_map[x])
                oe_prot.append(oe_cur)
        
        if len(oe_prot) == 0:
            logging.warning('Inconsistent sequence length for {}: CDS {} vs. protein {}'.format(uprot, len(pos_range_combine) // 3, prot_length))
        
        oe_list.append(pd.concat(oe_prot))

    return pd.concat(oe_list)



if __name__ == '__main__':
    data_root = Path('/local/storage/yl986/3d_vip/data_prepare/data')
    uprot2esnt_path = data_root / 'uprot_to_enst.json'
    # variant_file = './data_prepare/data/balance_var_struct_info.csv'
    oe_data_root = Path('/fs/cbsuhyfs1/storage1/yl986/data/oe_results/')
    # oe_data_root = Path('/home/yl986/data/3dvip/oe_output/hg38/')
    protein_fasta = ["/fs/cbsuhyfs1/storage/dx38/local_resource/uniprot_data_20220526/uniprot_sprot.fasta",
                     "/fs/cbsuhyfs1/storage/dx38/local_resource/uniprot_data_20220526/uniprot_sprot_varsplic.fasta"]
    # protein_fasta = ["/home/yl986/data/UniProt/uniprot_sprot.fasta",
    #                  "/home/yl986/data/UniProt/uniprot_sprot_varsplic.fasta"]
    
    uprot_by_chrom_file = data_root / 'uprot_by_chrom_tocheck.json'
    # target_chrom = sys.argv[1]
    out_dir = oe_data_root
    log_dir = '../logs'

    # out_path = Path(out_dir) / f'oe_prot_mapped_chr{target_chrom}.csv.gz'
    out_path = Path(out_dir) / 'oe_prot_mapped_add.csv.gz'

    formatter = '%(asctime)s - %(levelname)s: %(message)s'
    log_filename = f'{log_dir}/oe_agg_add.log'

    logging.basicConfig(
        filename=log_filename,
        format=formatter,
        level=logging.INFO,
        filemode="w",
    )
    
    ensembl = EnsemblRelease(108)

    with open(uprot2esnt_path, 'r') as f:
        uprot2enst = json.load(f)
    
    with open(uprot_by_chrom_file, 'r') as f:
        chrom2uprot = json.load(f)
    
    prot_seq_dict = dict()
    for fs in protein_fasta:
        prot_seq_dict.update(parse_fasta(fs))
    oe_list = []
    for target_chrom in chrom2uprot:
        # for chrom in chrom2uprot:
        oe_chrom = pd.read_csv(oe_data_root / 'hg38' / 'chr{}_oe_hg38_mapped.csv.gz'.format(str(target_chrom).lower()))
        uprot_list = chrom2uprot[target_chrom.upper()]
        df_oe_cur = fetch_oe_data(uprot_list, uprot2enst, oe_chrom, ensembl, prot_seq_dict)
        oe_list.append(df_oe_cur)
    
    df_oe = pd.concat(oe_list)
    df_oe.to_csv(out_path, index=False)
    