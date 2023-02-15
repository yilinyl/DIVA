import pickle
import numpy as np
import pandas as pd
from pathlib import Path

FEAT_ROOT = '/fs/cbsuhyfs1/storage1/dx38/mutation_pathogenicity/PIONEER_features'

feat_root = Path(FEAT_ROOT)

df_info = pd.read_csv('/local/storage/yl986/3d_vip/data_prepare/data/pio_feat_info.csv')
df_info = df_info.drop_duplicates(['pid'])
feats = []
feat_cols = ['expasy_ACCE', 'expasy_AREA', 'expasy_BULK', 'expasy_COMP',
             'expasy_HPHO', 'expasy_POLA', 'expasy_TRAN', 'expasy_ACCE_norm',
             'expasy_AREA_norm', 'expasy_BULK_norm', 'expasy_COMP_norm',
             'expasy_HPHO_norm', 'expasy_POLA_norm', 'expasy_TRAN_norm', 'SS_H',
             'SS_E', 'SS_C', 'SS_H_norm', 'SS_E_norm', 'SS_C_norm', 'ACC_B', 'ACC_M',
             'ACC_E', 'ACC_B_norm', 'ACC_M_norm', 'ACC_E_norm', 'JS', 'JS_norm']

for i, record in df_info.iterrows():
    uprot = record['pid']
    feat_version = record['version']

    feat_path = feat_root / feat_version / 'sequence_features'
    with open(list(feat_path.glob('*{}*.pkl'.format(uprot)))[0], 'rb') as f:
        pair, pair_feat = pickle.load(f)
    idx = 0 if pair[0] == uprot else 1
    feat_data = pair_feat[idx][feat_cols].reset_index(drop=True)
    feats.append(feat_data)

feats = np.concatenate(feats, axis=0)
feat_stats = dict()
feat_stats['mean'] = np.nanmean(feats, axis=0)
feat_stats['min'] = np.nanmin(feats, axis=0)
feat_stats['max'] = np.nanmax(feats, axis=0)

with open('./feat_stats.pkl', 'wb') as f_pkl:
    pickle.dump([feat_stats, feat_cols], f_pkl)
