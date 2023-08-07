from pathlib import Path
import os, sys
import numpy as np
import pandas as pd

import pickle

# Aggregate PIONEER predicted interface score by protein

pio_ires_path = Path('/fs/cbsuhyfs1/storage1/yl986/data/PIONEER_pred')
pio_score_path = Path('/fs/cbsuhyfs1/storage/dl953/PIONEER_large_scale_prediction')


def fetch_pio_score(pio_files, pio_path):
    if isinstance(pio_path, str):
        pio_path = Path(pio_path)
    score_dict = dict()
    
    for prefix in pio_files:
        p1, p2, idx = prefix.split('_')
        partner = [p1, p2][1 - int(idx)]
        fname = prefix + '.pkl'
        try:
            cur_score = pd.read_pickle(pio_path / fname).sort_values('res_id', ascending=True)
        except FileNotFoundError:
            print('{} not found'.format(fname))
            continue
        score_dict[partner] = cur_score['prob'].tolist()
    
    return score_dict


if __name__ == '__main__':
    data_root = Path('/local/storage/yl986/3d_vip/data_prepare/data')
    #with open(data_root / 'IRES' / 'ires_by_prot_hm.pkl', 'rb') as f_pkl:
    #    prot2ires_hm = pickle.load(f_pkl)
    prot2pio_fname = pd.read_pickle(data_root / 'IRES' / 'prot_to_pio_file.pkl')
    
    ires_info = pd.read_csv(data_root / 'IRES' / 'ires_summary_all.txt', sep='\t')
    out_path = pio_ires_path / 'score_by_prot'

    prot_list = ires_info[ires_info['in_pioneer']]['UniProt'].tolist()

    for uprot in prot_list:
        if uprot not in prot2pio_fname:
            continue
        score_dict = fetch_pio_score(prot2pio_fname[uprot], pio_score_path)
        df_score = pd.DataFrame(score_dict)
        with open(out_path / f'{uprot}_pio.pkl', 'wb') as f_pkl:
            pickle.dump(df_score.values, f_pkl)
            #df_score.to_csv(out_path / f'{uprot}_pio.csv', index=False)
