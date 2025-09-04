import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import logging
from tqdm import *
from preprocess.utils import *
# from data.data_utils import *
from utils import gpu_setup

torch.set_default_dtype(torch.float64)
from data.lm_utils import *


def get_all_lm_embed(df_in, tokenizer, model, seq_dict=None, esm_cache_dir='./esm_cache'):
    esm_cache_path = Path(esm_cache_dir)
    if not esm_cache_path.exists():
        esm_cache_path.mkdir(parents=True)
    ref_path = esm_cache_path / 'REF'
    alt_path = esm_cache_path / 'ALT'
    if not ref_path.exists():
        ref_path.mkdir()
    if not alt_path.exists():
        alt_path.mkdir()

    uprots = set(df_in['UniProt'])
    # for i, record in tqdm(df_in.iterrows(), total=df_in.shape[0]):
    for uprot in uprots:
        # uprot = record['UniProt']
        if uprot not in seq_dict:
            seq_dict[uprot] = fetch_prot_seq(uprot)
        seq = seq_dict[uprot]
        f_emb_ref = ref_path / f'{uprot}.pt'
        if not f_emb_ref.exists():
            emb = calc_esm_emb(seq, tokenizer, model)
            torch.save(emb.detach().cpu(), f_emb_ref)

        df_uprot = df_in.loc[df_in['UniProt'] == uprot]
        for i, record in df_uprot.iterrows():
            uprot_pos = record['Protein_position']
            alt_aa = record['ALT_AA']
            f_emb_alt = alt_path / f'{uprot}_{uprot_pos}{alt_aa}.pt'
            if not f_emb_alt.exists():
                seq_alt = seq[:uprot_pos - 1] + alt_aa + seq[uprot_pos:]
                emb = calc_esm_emb(seq_alt, tokenizer, model)
                torch.save(emb.detach().cpu(), f_emb_alt)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/esm_data_config.json', help="Config file path (.json)")
    parser.add_argument('--fname', default='train.csv', help="Input file name")
    parser.add_argument('--pretrained_name', default='facebook/esm2_t12_35M_UR50D', help="Name of pretrained ESM to use")
    parser.add_argument('--gpu_id', type=int)

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    data_params = config['data_params']
    data_path = Path(config['data_dir'])
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])

    # Graph cache config
    # graph_cache_root = Path(data_params['graph_cache_root'])
    # if data_params['method'] == 'radius':
    #     graph_cache = graph_cache_root / f'radius{data_params["radius"]}'
    # else:
    #     graph_cache = graph_cache_root / f'knn{data_params["num_neighbors"]}'

    # data_params['graph_cache'] = os.fspath(graph_cache)

    df_var = pd.read_csv(data_path / args.fname)

    seq_dict = dict()
    for fname in data_params['seq_fasta']:
        try:
            seq_dict.update(parse_fasta(fname))
        except FileNotFoundError:
            pass
    data_params['seq_dict'] = seq_dict

    esm_tokenizer, esm_model = init_pretrained_lm(args.pretrained_name)
    esm_model = esm_model.to(device)
    pretrained_full = args.pretrained_name
    esm_option = pretrained_full.split('/')[-1]
    get_all_lm_embed(df_var, esm_tokenizer, esm_model, seq_dict,
                     esm_cache_dir=f'{config["esm_cache_root"]}/{esm_option}')




