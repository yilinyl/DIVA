import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.abspath('..'))

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import yaml
import json
import pickle
import copy
import argparse

import numpy as np
import pandas as pd

import torch
from torch.utils.data.dataloader import DataLoader

from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertModel, BertForMaskedLM, EsmForMaskedLM, AutoModelForMaskedLM
from transformers import BertConfig, EsmConfig

from data.dis_var_dataset import ProteinVariantDatset, ProteinVariantDataCollator, PhenotypeDataset, TextDataCollator
import logging
from datetime import datetime
from utils import str2bool, setup_logger, set_seed, load_input_to_device, _save_scores
from metrics import *
from dev.preprocess.foldseek_util import get_struc_seq
from dev.disease_inference import load_config, env_setup, embed_phenotypes, save_pheno_results, inference
from models.dis_var_models import DiseaseVariantEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="configuration file", default='./configs/dis_var_pred_config.yaml')
    parser.add_argument("-c_fmt", "--config_fmt", help="configuration file format", default='yaml')
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)
    parser.add_argument('--tensorboard', type=str2bool, default=False,
                        help='Option to write log information in tensorboard')
    parser.add_argument('--data_dir', help='Data directory')
    parser.add_argument('--exp_dir', help='Directory for all training related files, e.g. checkpoints, log')
    parser.add_argument('--experiment', help='Experiment name')
    parser.add_argument('--log_level', default='info', help='Log level')
    parser.add_argument('--save_freq', type=int, default=1, help='Frequency to save models')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config, format=args.config_fmt)

    config, device = env_setup(args, config)

    data_configs = config['dataset']
    model_args = config['model']

    exp_path = Path(config['exp_dir'])

    # prot2seq = dict()
    prot2desc = dict()
    data_root = Path(data_configs['data_dir'])
    
    df_ref = pd.read_csv(data_configs['dms_reference_file'])  # DMS reference file
    dms_seq_dict = dict(zip(df_ref['DMS_id'], df_ref['target_seq']))
    dms_pdb_dict = dict(zip(df_ref['DMS_id'], df_ref['pdb_file']))

    df_input = pd.read_csv(data_root / data_configs['input_file']['test'])
    dms_source_list = df_input['source'].drop_duplicates().tolist()
    dms_id2uprot = dict(zip(df_input['source'], df_input[data_configs['pid_col']]))
    
    for fpath in data_configs['prot_meta_data']:
        try:
            df_meta = pd.read_csv(fpath, sep='\t').dropna(subset=['function'])
            # meta_info_all.append(df)
            prot_func_dict = dict(zip(df_meta['UniProt'], df_meta['function']))
            prot2desc.update(prot_func_dict)
        except FileNotFoundError:
            pass
        
    for prot, desc in prot2desc.items():
        # update description of isoforms
        if prot.find('-') >= 0:
            prot2desc[prot] = ' '.join([desc, prot2desc.get(prot.split('-')[0], '')]).strip()

    prot2comb_seq = dict()
    exclude_prots = []
    for dms_id in dms_source_list:
        if dms_id not in dms_seq_dict:
            exclude_prots.append(dms_id2uprot[dms_id])
            continue
        if data_configs['use_struct_vocab']:
            pdb_file = dms_pdb_dict[dms_id]
            if pdb_file.find('|') != -1:
                exclude_prots.append(dms_id2uprot[dms_id])
                continue
            pid = dms_id2uprot[dms_id]
            pdb_fpath = os.path.join(data_configs['struct_folder'], pdb_file)
            tmp_fpath = str(exp_path / 'tmp_struct_seq.tsv')
            struc_seq = get_struc_seq(data_configs['foldseek_bin'], pdb_fpath, ["A"], plddt_mask=True, plddt_threshold=70, tmp_save_path=tmp_fpath)["A"][1].lower()
            prot2comb_seq[pid] = "".join([a + b for a, b in zip(dms_seq_dict[dms_id], struc_seq)])
            os.remove(tmp_fpath)
    
    data_configs['seq_dict'] = {dms_id2uprot[k]: dms_seq_dict[k] for k in dms_id2uprot.keys()}
    data_configs['protein_info_dict'] = prot2desc
    
    afmis_root = None
    if data_configs['use_alphamissense']:
        afmis_root = Path(data_configs['alphamissense_score_dir'])

    # Initialize tokenizer
    protein_tokenizer = AutoTokenizer.from_pretrained(model_args['protein_lm_path'],
        do_lower_case=False
    )
    text_tokenizer = BertTokenizer.from_pretrained(model_args['text_lm_path'])

    # Load data
    with open(data_configs['phenotype_vocab_file'], 'r') as f:
        phenotype_vocab = f.read().splitlines()
    phenotype_vocab.insert(0, text_tokenizer.unk_token)  # add unknown token
    if data_configs['use_pheno_desc']:
        with open(data_configs['phenotype_desc_file']) as f:
            pheno_desc_dict = json.load(f)
    else:
        pheno_desc_dict = None

    pheno_dataset = PhenotypeDataset(phenotype_vocab, pheno_desc_dict, use_desc=data_configs['use_pheno_desc'])
    pheno_collator = TextDataCollator(text_tokenizer, padding=True)
    phenotype_loader = DataLoader(pheno_dataset, batch_size=config['pheno_batch_size'], collate_fn=pheno_collator, shuffle=False)
    
    seq_config = BertConfig.from_pretrained(model_args['protein_lm_path'])
    text_config = BertConfig.from_pretrained(model_args['text_lm_path'])

    seq_encoder = EsmForMaskedLM(seq_config)
    text_encoder = BertForMaskedLM(text_config)
    
    model = DiseaseVariantEncoder(seq_encoder=seq_encoder,
                                  text_encoder=text_encoder,
                                  n_residue_types=protein_tokenizer.vocab_size,
                                  hidden_size=512,
                                  use_desc=True,
                                  pad_label_idx=-100,
                                  dist_fn_name=model_args['dist_fn_name'],
                                  init_margin=model_args['margin'],
                                  use_struct_vocab=data_configs['use_struct_vocab'],
                                  use_alphamissense=data_configs['use_alphamissense'],
                                  adjust_logits=model_args['adjust_logits'],
                                  device=device)
    checkpt_dict = torch.load(config['model_path'], map_location='cpu')
    model.load_state_dict(checkpt_dict['state_dict'])

    for name, parameters in model.named_parameters():
        parameters.requires_grad = False
    
    model = model.to(device)
    all_pheno_embs = embed_phenotypes(model, device, phenotype_loader)
    all_pheno_embs = torch.tensor(all_pheno_embs, device=device)
    # if isinstance(data_configs['input_file']['test'], str):
    #     test_flist = [data_configs['input_file']['test']]
    # else:
    #     test_flist = data_configs['input_file']['test']

    # for test_file in test_flist:
    test_file = data_configs['input_file']['test']
    logging.info(f'Inference on {test_file}...')
    fname = os.path.basename(test_file).split('.')[0]
    test_dataset = ProteinVariantDatset(**data_configs, 
                                        variant_file=test_file, 
                                        split='test', 
                                        phenotype_vocab=phenotype_vocab, 
                                        protein_tokenizer=protein_tokenizer, 
                                        text_tokenizer=text_tokenizer,
                                        #  var_db=var_db,
                                        # prot_var_cache=prot_var_cache,
                                        mode='eval',
                                        update_var_cache=False,
                                        comb_seq_dict=prot2comb_seq,
                                        exclude_prots=exclude_prots,
                                        afmis_root=afmis_root,
                                        access_to_context=False)
    logging.info('{} variants loaded'.format(len(test_dataset)))
    test_collator = ProteinVariantDataCollator(test_dataset.get_protein_data(), protein_tokenizer, text_tokenizer, phenotype_vocab=phenotype_vocab, 
                                        use_prot_desc=True, truncate_protein=data_configs['truncate_protein'], 
                                        max_protein_length=data_configs['max_protein_seq_length'],
                                        use_struct_vocab=data_configs['use_struct_vocab'], use_alphamissense=data_configs['use_alphamissense'],
                                        use_pheno_desc=data_configs['use_pheno_desc'], pheno_desc_dict=pheno_desc_dict)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], collate_fn=test_collator)
    
    test_labels, test_scores, test_vars, test_adj_weights, test_pheno_results = inference(model, device, test_loader, pheno_vocab_emb=all_pheno_embs, topk=100)

    _save_scores(test_vars, test_labels, test_scores, fname, weights=test_adj_weights, epoch='', exp_dir=str(exp_path), mode='eval')
    logging.info('Done!')
