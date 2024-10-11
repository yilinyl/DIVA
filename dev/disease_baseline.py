import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.abspath('..'))

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import yaml
import json
import copy
import argparse

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertModel, BertForMaskedLM, EsmForMaskedLM, AutoModelForMaskedLM

from data.dis_var_dataset import ProteinVariantDatset, ProteinVariantDataCollator, PhenotypeDataset, TextDataCollator
import logging
from datetime import datetime
from metrics import *
from dev.preprocess.utils import parse_fasta_info
from dev.utils import load_config, env_setup, load_input_to_device
from dev.models.baseline_models import LMBaseModel
from dev.benchmark.utils import extract_text_emb


def compile_features(model, data_loader, device):

    var_emb_list = []
    var_pheno_label_all = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            seq_feat_dict = load_input_to_device(batch_data['seq_input_feat'], device)
            desc_feat_dict = load_input_to_device(batch_data['desc_input_feat'], device)
            variant_data = load_input_to_device(batch_data['variant'], device=device, exclude_keys=['var_names'])
            
            seq_pheno_emb_raw, _ = model(seq_feat_dict, variant_data, desc_feat_dict)
            var_emb_list.append(seq_pheno_emb_raw.detach().cpu().numpy())
            var_pheno_label_all.extend(variant_data['pos_pheno_idx'].detach().cpu().tolist())
            # mlm_logits_list.append(mlm_logits.detach().cpu())
    
        var_emb_raw = np.concatenate(var_emb_list, 0)
        
    return var_emb_raw, var_pheno_label_all


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="configuration file", default='./configs/dis_var_config_baseline.yaml')
    parser.add_argument("-c_fmt", "--config_fmt", help="configuration file format", default='yaml')
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)
    parser.add_argument('--data_dir', help='Data directory')
    parser.add_argument('--exp_dir', help='Directory for all training related files, e.g. checkpoints, log')
    parser.add_argument('--experiment', help='Experiment name')
    parser.add_argument('--log_level', default='info', help='Log level')
    # args, unparsed = parser.parse_known_args()
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config, format=args.config_fmt)

    config, device = env_setup(args, config, use_timestamp=False)

    data_configs = config['dataset']
    model_args = config['model']

    exp_dir = config['exp_dir']
    emb_output_path = Path(exp_dir) / 'embeddings'

    # pheno_result_path = result_path / 'phenotype'
    if not emb_output_path.exists():
        emb_output_path.mkdir(parents=True)
    
    prot2seq = dict()
    prot2desc = dict()
    data_root = Path(data_configs['data_dir'])
    for fname in data_configs['seq_fasta']:
        try:
            seq_dict, desc_dict = parse_fasta_info(fname)
            prot2seq.update(seq_dict)
            prot2desc.update(desc_dict)  # string of protein definition E.g. BRCA1_HUMAN Breast cancer type 1 susceptibility protein
        except FileNotFoundError:
            pass
    
    # prot2desc = dict()
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
    
    data_configs['seq_dict'] = prot2seq
    data_configs['protein_info_dict'] = prot2desc

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
    train_dataset = ProteinVariantDatset(**data_configs, 
                                         variant_file=data_configs['input_file']['train'], 
                                         split='train', 
                                         phenotype_vocab=phenotype_vocab, 
                                         protein_tokenizer=protein_tokenizer, 
                                         text_tokenizer=text_tokenizer,
                                         pheno_desc_dict=pheno_desc_dict,
                                        #  use_struct_neighbor=data_configs['use_struct_neighbor'],
                                         access_to_context=True)
    
    prot_var_cache = train_dataset.get_protein_cache()
    val_dataset = ProteinVariantDatset(**data_configs, 
                                         variant_file=data_configs['input_file']['val'], 
                                         split='val', 
                                         phenotype_vocab=phenotype_vocab, 
                                         protein_tokenizer=protein_tokenizer, 
                                         text_tokenizer=text_tokenizer,
                                         pheno_desc_dict=pheno_desc_dict,
                                         prot_var_cache=prot_var_cache,
                                         access_to_context=False)  # context variants in validation set not visible to each other
    

    prot_var_cache = val_dataset.get_protein_cache()
    
    test_dataset = ProteinVariantDatset(**data_configs, 
                                         variant_file=data_configs['input_file']['test'], 
                                         split='test', 
                                         phenotype_vocab=phenotype_vocab, 
                                         protein_tokenizer=protein_tokenizer, 
                                         text_tokenizer=text_tokenizer,
                                         pheno_desc_dict=pheno_desc_dict,
                                        #  var_db=var_db,
                                         prot_var_cache=prot_var_cache,
                                         access_to_context=False)

    # Initilize pretrained encoders:
    # seq_encoder = EsmForMaskedLM.from_pretrained(model_args['protein_lm_path'])
    seq_encoder = AutoModelForMaskedLM.from_pretrained(model_args['protein_lm_path'])
    text_encoder = BertForMaskedLM.from_pretrained(model_args['text_lm_path'])

    # Non-shuffle for all...
    train_collator = ProteinVariantDataCollator(train_dataset.get_protein_data(), protein_tokenizer, text_tokenizer, phenotype_vocab=phenotype_vocab, 
                                                use_prot_desc=True, max_protein_length=data_configs['max_protein_seq_length'], half_window_size=data_configs['half_window_size'],
                                                context_agg_opt=data_configs['context_agg_option'], use_pheno_desc=data_configs['use_pheno_desc'], 
                                                pheno_desc_dict=pheno_desc_dict, use_struct_vocab=data_configs['use_struct_vocab'], 
                                                use_struct_neighbor=data_configs['use_struct_neighbor'], struct_radius_cutoff=data_configs['struct_radius_cutoff'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], collate_fn=train_collator, shuffle=False)
    val_collator = ProteinVariantDataCollator(val_dataset.get_protein_data(), protein_tokenizer, text_tokenizer, phenotype_vocab=phenotype_vocab, 
                                              use_prot_desc=True, max_protein_length=data_configs['max_protein_seq_length'], half_window_size=data_configs['half_window_size'],
                                              context_agg_opt=data_configs['context_agg_option'], use_pheno_desc=data_configs['use_pheno_desc'], 
                                              pheno_desc_dict=pheno_desc_dict, use_struct_vocab=data_configs['use_struct_vocab'], 
                                              use_struct_neighbor=data_configs['use_struct_neighbor'], struct_radius_cutoff=data_configs['struct_radius_cutoff'])
    validation_loader = DataLoader(val_dataset, batch_size=config['batch_size'], collate_fn=val_collator)
    test_collator = ProteinVariantDataCollator(test_dataset.get_protein_data(), protein_tokenizer, text_tokenizer, phenotype_vocab=phenotype_vocab, 
                                               use_prot_desc=True, max_protein_length=data_configs['max_protein_seq_length'], half_window_size=data_configs['half_window_size'],
                                               context_agg_opt=data_configs['context_agg_option'], use_pheno_desc=data_configs['use_pheno_desc'], 
                                               pheno_desc_dict=pheno_desc_dict, use_struct_vocab=data_configs['use_struct_vocab'], 
                                               use_struct_neighbor=data_configs['use_struct_neighbor'], struct_radius_cutoff=data_configs['struct_radius_cutoff'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], collate_fn=test_collator)

    loader_dict = {'train': train_loader, 'test': test_loader, 'val': validation_loader}

    if model_args['frozen_bert']:
        # prot_unfreeze_layers = ['esm.encoder.layer.11']
        for name, parameters in seq_encoder.named_parameters():
            parameters.requires_grad = False
        # text_unfreeze_layers = ['bert.encoder.layer.11']
        for name, parameters in text_encoder.named_parameters():
            parameters.requires_grad = False

    seq_encoder = seq_encoder.to(device)
    text_encoder = text_encoder.to(device)
    base_lm_model = LMBaseModel(seq_encoder, text_encoder, use_desc=True)
    base_lm_model = base_lm_model.to(device)
    for split, loader in loader_dict.items():
        var_embs, var_dis_labels = compile_features(base_lm_model, loader, device=device)
        np.save(emb_output_path / f'{split}_var_embs.npy', var_embs)
        with open(emb_output_path / f'{split}_var_pheno_idx.txt', 'w') as f:
            f.write('\n'.join([str(s) for s in var_dis_labels]))

    all_pheno_embs = extract_text_emb(text_encoder, device, phenotype_loader)
    np.save(emb_output_path / 'phenotype_emb_raw.npy', all_pheno_embs)
