import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import yaml
import json
import argparse

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertModel, BertForMaskedLM, EsmForMaskedLM, AutoModelForMaskedLM
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

from data.dis_var_dataset import ProteinVariantDatset, ProteinVariantDataCollator
import logging
from datetime import datetime
from utils import str2bool, setup_logger, set_seed, load_input_to_device
from dev.preprocess.utils import parse_fasta_info
from models.protein_encoder import DiseaseVariantEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="configuration file", default='./configs/dis_var_config.yaml')
    parser.add_argument("-c_fmt", "--config_fmt", help="configuration file format", default='yaml')
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)
    parser.add_argument('--data_dir', help='Data directory')
    parser.add_argument('--exp_dir', help='Directory for all training related files, e.g. checkpoints, log')
    parser.add_argument('--experiment', help='Experiment name')
    parser.add_argument('--log_level', default='info', help='Log level')

    parser.add_argument('--inf-check', type=str2bool, default=False,
                        help='add hooks to check for infinite module outputs and gradients')
    # args, unparsed = parser.parse_known_args()
    args = parser.parse_args()

    return args


def train_epoch(model, optimizer, device, data_loader, diagnostic=None):
    model.train()
    running_loss = 0
    n_sample = 0
    for batch_idx, batch_data in enumerate(data_loader):
        # TODO: check batch_data structure
        # seq_input_data = {'input_ids': batch_data['seq_input_ids'].to(device),
        #                   'attention_mask': batch_data['seq_attention_mask'].to(device),
        #                   'token_type_ids': batch_data['seq_token_type_ids'].to(device)}
        
        # desc_input_data = {'input_ids': batch_data['desc_input_ids'].to(device),
        #                    'attention_mask': batch_data['desc_attention_mask'].to(device),
        #                    'token_type_ids': batch_data['desc_token_type_ids'].to(device)}
        # batch_var_idx = batch_data[3].to(device)
        seq_feat_dict = load_input_to_device(batch_data['seq_input_feat'], device)
        desc_feat_dict = load_input_to_device(batch_data['desc_input_feat'], device)
        batch_labels = batch_data['label']
        optimizer.zero_grad()
        # batch_pheno_feat = batch_data['phenotype']
        # TODO: parse phenotype information
        seq_emb, mlm_logits, desc_emb = model(seq_feat_dict, batch_data, desc_feat_dict)
        # shapes = batch_logits.size()
        # batch_logits = batch_logits.view(shapes[0]*shapes[1])

        loss = model.loss(batch_logits, batch_labels)
        loss.backward()
        optimizer.step()
        
        loss_ = loss.detach().item()
        size = batch_labels.size()[0]
        running_loss += loss_* size
        n_sample += size

        if diagnostic and batch_idx == 5:
            diagnostic.print_diagnostics()
            break
    epoch_loss = running_loss / n_sample
    
    return epoch_loss, optimizer


def load_config(cfg_file, format='yaml'):
    if format.lower() == 'yaml':
        with open(cfg_file, 'r') as f:
            config = yaml.safe_load(f)
    elif format.lower() == 'json':
        with open(cfg_file, 'r') as f:
            config = json.load(cfg_file)
    else:
        raise ValueError(f'{format} not supported! Please use one of [JSON, YAML]')
    
    return config

def gpu_setup(device='cpu'):
    """
    Setup GPU device
    """
    
    if torch.cuda.is_available() and device != 'cpu':
        device = torch.device(device)
    else:
        device = torch.device('cpu')
        logging.info('GPU not available, running on CPU')

    return device


def env_setup(args, config):
    
    device = gpu_setup(config['device'])
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")

    if args.exp_dir is not None:
        config['exp_dir'] = args.exp_dir
    if args.experiment is not None:
        config['experiment'] = args.experiment
    config['exp_dir'] = '{exp_root}/{name}/{date_time}'.format(exp_root=config['exp_dir'],
                                                               name=config['experiment'],
                                                               date_time=date_time)
    # Set up logging file
    setup_logger(config['exp_dir'], log_prefix=config['mode'], log_level=args.log_level)
    logging.info(json.dumps(config, indent=4))
    
    set_seed(args.seed, device)

    return config, device


def main():
    args = parse_args()
    config = load_config(args.config, format=args.config_fmt)

    config, device = env_setup(args, config)

    data_configs = config['dataset']
    model_args = config['model']

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

    train_dataset = ProteinVariantDatset(**data_configs, 
                                         variant_file=data_configs['input_file']['train'], 
                                         split='train', 
                                         phenotype_vocab=phenotype_vocab, 
                                         protein_tokenizer=protein_tokenizer, 
                                         text_tokenizer=text_tokenizer)
    # val_dataset = ProteinVariantDatset(**data_configs, variant_file=data_configs['input_file']['val'], split='val', phenotype_vocab=phenotype_vocab, protein_tokenizer=protein_tokenizer, phenotype_tokenizer=text_tokenizer)
    # test_dataset = ProteinVariantDatset(**data_configs, variant_file=data_configs['input_file']['test'], split='test', phenotype_vocab=phenotype_vocab, protein_tokenizer=protein_tokenizer, phenotype_tokenizer=text_tokenizer)

    # Initilize pretrained encoders:
    seq_encoder = EsmForMaskedLM.from_pretrained(model_args['protein_lm_path'])
    text_encoder = BertForMaskedLM.from_pretrained(model_args['text_lm_path'])

    var_collator = ProteinVariantDataCollator(protein_tokenizer, text_tokenizer, use_desc=True, pheno_descs=phenotype_vocab)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], collate_fn=var_collator)
    
    if model_args['frozen_bert']:
        # unfreeze_layers = ['layer.29', 'bert_encoder.pooler', 'classifier']
        for name, parameters in seq_encoder.named_parameters():
            parameters.requires_grad = False
        for name, parameters in text_encoder.named_parameters():
            parameters.requires_grad = False

    # if freeze_layer_ids is not None:
    #     for name, param in model.named_parameters():
    #         if any(f".{layer_id}." in name for layer_id in freeze_layer_ids):
    #             logging.info(f"freezing {name}")
    #             param.requires_grad = False
        
    seq_encoder = seq_encoder.to(device)
    text_encoder = text_encoder.to(device)
    model = DiseaseVariantEncoder(seq_encoder=seq_encoder,
                                  text_encoder=text_encoder,
                                  n_residue_types=protein_tokenizer.vocab_size,
                                  hidden_size=512,
                                  use_desc=True,
                                  pad_label_idx=-100)
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['init_lr'])
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
    #                                                  factor=net_params['lr_reduce_factor'],
    #                                                  patience=net_params['lr_schedule_patience'],
    #                                                  verbose=True)
    train_epoch(model, optimizer, device, train_loader)

    
    
if __name__ == '__main__':
    main()