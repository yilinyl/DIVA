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
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertModel, BertForMaskedLM, EsmForMaskedLM, AutoModelForMaskedLM
from transformers import BertConfig, EsmConfig
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

from data.dis_var_dataset import ProteinVariantDatset, ProteinVariantDataCollator, PhenotypeDataset, TextDataCollator
import logging
from datetime import datetime
from utils import str2bool, setup_logger, set_seed, load_input_to_device, _save_scores
from metrics import *
from dev.preprocess.utils import parse_fasta_info
from models.protein_encoder import DiseaseVariantEncoder


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
    # parser.add_argument('--inf-check', type=str2bool, default=False,
    #                     help='add hooks to check for infinite module outputs and gradients')
    # args, unparsed = parser.parse_known_args()
    args = parser.parse_args()

    return args


def embed_phenotypes(model, device, pheno_loader):
    model.eval()
    all_pheno_embs = []

    with torch.no_grad():
        for idx, batch_pheno in enumerate(pheno_loader):
            # pheno_input_dict = load_input_to_device(batch_pheno, device)
            pheno_input_dict = batch_pheno.to(device)
            pheno_embs = model.get_pheno_emb(pheno_input_dict, proj=True)
            all_pheno_embs.append(pheno_embs.detach().cpu().numpy())
    
    all_pheno_embs = np.concatenate(all_pheno_embs, 0)
    
    return all_pheno_embs


def inference(model, device, data_loader, pheno_vocab_emb, topk=None):
    model.eval()
    
    if not topk:
        topk = pheno_vocab_emb.shape[0]
    n_sample = 0
    n_pheno_sample = 0
    all_vars, all_scores, all_labels, all_pheno_scores = [], [], [], []
    all_pheno_emb_pred, all_pheno_emb_label, all_pos_pheno_descs, all_pos_pheno_idx = [], [], [], []
    all_patho_vars, all_pheno_emb_neg, all_neg_pheno_descs, all_pheno_neg_scores = [], [], [], []
    all_similarities, all_topk_scores, all_topk_indices = [], [], []
    # all_pheno_embs = []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            seq_feat_dict = load_input_to_device(batch_data['seq_input_feat'], device)
            desc_feat_dict = load_input_to_device(batch_data['desc_input_feat'], device)
            variant_data = load_input_to_device(batch_data['variant'], device=device, exclude_keys=['var_names'])
            batch_labels = batch_data['variant']['label'].unsqueeze(1).to(device)

            seq_pheno_emb, pos_emb_proj, neg_emb_proj, mlm_logits, logit_diff = model(seq_feat_dict, variant_data, desc_feat_dict)
            
            size = batch_labels.size()[0]
            cur_pheno_size = batch_labels.sum().item()

            # running_loss += loss_* size
            n_sample += size
            n_pheno_sample += cur_pheno_size
            batch_patho_scores = torch.sigmoid(logit_diff)

            all_scores.append(batch_patho_scores.squeeze(1).detach().cpu().numpy())
            all_vars.extend(batch_data['variant']['var_names'])
            all_labels.append(batch_labels.squeeze(1).detach().cpu().numpy())

            if batch_data['variant']['infer_phenotype']:
                # if pos_emb_proj is not None:
                #     pheno_score_pos = torch.cosine_similarity(seq_pheno_emb, pos_emb_proj)
                #     all_pheno_scores.append(pheno_score_pos.detach().cpu().numpy())
                #     all_pheno_emb_label.append(pos_emb_proj.detach().cpu().numpy())
                # pheno_score_neg = torch.cosine_similarity(seq_pheno_emb, neg_emb_proj)
                cos_sim_all = torch.cosine_similarity(seq_pheno_emb.unsqueeze(1), pheno_vocab_emb.unsqueeze(0), dim=-1)
                topk_scores, topk_indices = torch.topk(cos_sim_all, k=topk, dim=1)
                # all_pheno_neg_scores.append(pheno_score_neg.detach().cpu().numpy())

                all_pheno_emb_pred.append(seq_pheno_emb.detach().cpu().numpy())
                
                # all_pheno_emb_neg.append(neg_emb_proj.detach().cpu().numpy())

                all_topk_scores.append(topk_scores.detach().cpu().numpy())
                all_topk_indices.append(topk_indices.detach().cpu().numpy())
                all_similarities.append(cos_sim_all.detach().cpu().numpy())

                all_patho_vars.extend(batch_data['variant']['pheno_var_names'])
                if pos_emb_proj is not None:
                    pheno_score_pos = torch.cosine_similarity(seq_pheno_emb, pos_emb_proj)
                    all_pheno_scores.append(pheno_score_pos.detach().cpu().numpy())
                    all_pheno_emb_label.append(pos_emb_proj.detach().cpu().numpy())
                    all_pos_pheno_descs.extend(batch_data['variant']['pos_pheno_desc'])
                    all_pos_pheno_idx.extend(batch_data['variant']['pos_pheno_idx'])
                # all_neg_pheno_descs.extend(batch_data['variant']['neg_pheno_desc'])

        # epoch_loss = running_loss / n_sample
        all_labels = np.concatenate(all_labels, 0)
        all_scores = np.concatenate(all_scores, 0)
        
        # all_pheno_neg_scores = np.concatenate(all_pheno_neg_scores, 0)
        if all_pheno_emb_pred:
            all_pheno_emb_pred = np.concatenate(all_pheno_emb_pred, 0)
        # all_pheno_emb_neg = np.concatenate(all_pheno_emb_neg, 0)

            all_similarities = np.concatenate(all_similarities, 0)
            all_topk_scores = np.concatenate(all_topk_scores, 0)
            all_topk_indices = np.concatenate(all_topk_indices, 0)

            all_pheno_results = {'var_names': all_patho_vars,
                                'label': all_labels,
                                'pred_emb': all_pheno_emb_pred,
                                #  'neg_emb': all_pheno_emb_neg,
                                'similarities': all_similarities,
                                'topk_scores': all_topk_scores,
                                'topk_indices': all_topk_indices}
            if all_pos_pheno_descs:
                all_pheno_scores = np.concatenate(all_pheno_scores, 0)
                all_pheno_emb_label = np.concatenate(all_pheno_emb_label, 0)
                all_pheno_results.update({
                    'pos_emb': all_pheno_emb_label,
                    'pos_score': all_pheno_scores,
                    'pos_pheno_desc': all_pos_pheno_descs,
                    'pos_pheno_idx': all_pos_pheno_idx})
        else:
            all_pheno_results = dict()
    
    return all_labels, all_scores, all_vars, all_pheno_results


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

def save_pheno_results(pheno_result_dict, save_path, name, save_emb=False):
    if isinstance(save_path, str):
        save_path = Path(save_path)

    if 'pos_pheno_desc' in pheno_result_dict:
        df_pheno_results = pd.DataFrame({'prot_var_id': pheno_result_dict['var_names'], 
                                        'label': pheno_result_dict['label'],
                                        'pos_phenotype': pheno_result_dict['pos_pheno_desc'], 
                                        #  'neg_phenotype': pheno_result_dict['neg_pheno_desc'], 
                                        'pos_score': pheno_result_dict['pos_score']})
        df_pheno_results.to_csv(save_path / f'{name}_pheno_score.tsv', sep='\t', index=False)
    np.save(save_path / f'{name}_sim.npy', pheno_result_dict['similarities'])
    if save_emb:
        np.save(save_path / f'{name}_pheno_pred_emb.npy', pheno_result_dict['pred_emb'])
        # pd.DataFrame(pheno_result_dict['pos_emb']).to_csv(save_path / f'{split}_pheno_true_emb.tsv', sep='\t', index=False, header=False)
        # pd.DataFrame(pheno_result_dict['neg_emb']).to_csv(save_path / f'ep{epoch}_{split}_pheno_neg_emb.tsv', sep='\t', index=False, header=False)
    
    
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


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config, format=args.config_fmt)

    config, device = env_setup(args, config)

    data_configs = config['dataset']
    model_args = config['model']

    exp_path = Path(config['exp_dir'])
    # result_path = Path(exp_dir) / 'result'
    # if not result_path.exists():
    #     result_path.mkdir(parents=True)

    # pheno_result_path = result_path / 'phenotype'
    # if not pheno_result_path.exists():
    #     pheno_result_path.mkdir(parents=True)

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
    pheno_dataset = PhenotypeDataset(phenotype_vocab)
    pheno_collator = TextDataCollator(text_tokenizer, padding=True)
    phenotype_loader = DataLoader(pheno_dataset, batch_size=config['pheno_batch_size'], collate_fn=pheno_collator, shuffle=False)
    # TODO: load variant cache
    if os.path.exists(data_configs['variant_cache_file']):
        with open(data_configs['variant_cache_file']) as f:
            prot_var_cache = json.load(f)
    else:
        ref_dataset = ProteinVariantDatset(**data_configs, 
                                            variant_file=data_configs['input_file']['train'], 
                                            split='train', 
                                            phenotype_vocab=phenotype_vocab, 
                                            protein_tokenizer=protein_tokenizer, 
                                            text_tokenizer=text_tokenizer,
                                            mode='train',
                                            update_var_cache=True)
        # var_db = pd.read_csv(data_root / data_configs['input_file']['train']).query('label == 1').\
        #     drop_duplicates([data_configs['pid_col'], data_configs['pos_col'], data_configs['pheno_col']])
        prot_var_cache = ref_dataset.get_protein_cache()

        with open(data_configs['variant_cache_file'], 'w') as f:
            json.dump(prot_var_cache, f, indent=2)
    
    seq_config = BertConfig.from_pretrained(model_args['protein_lm_path'])
    text_config = BertConfig.from_pretrained(model_args['text_lm_path'])

    seq_encoder = EsmForMaskedLM(seq_config)
    text_encoder = BertForMaskedLM(text_config)
    # seq_encoder = seq_encoder.to(device)
    # text_encoder = text_encoder.to(device)
    model = DiseaseVariantEncoder(seq_encoder=seq_encoder,
                                  text_encoder=text_encoder,
                                  n_residue_types=protein_tokenizer.vocab_size,
                                  hidden_size=512,
                                  use_desc=True,
                                  pad_label_idx=-100,
                                  dist_fn_name=model_args['dist_fn_name'],
                                  init_margin=model_args['margin'],
                                  device=device)
    checkpt_dict = torch.load(config['model_path'], map_location='cpu')
    model.load_state_dict(checkpt_dict['state_dict'])

    for name, parameters in model.named_parameters():
        parameters.requires_grad = False
    
    model = model.to(device)
    all_pheno_embs = embed_phenotypes(model, device, phenotype_loader)
    all_pheno_embs = torch.tensor(all_pheno_embs, device=device)
    if isinstance(data_configs['input_file']['test'], str):
        test_flist = [data_configs['input_file']['test']]
    else:
        test_flist = data_configs['input_file']['test']

    for test_file in test_flist:
        logging.info(f'Inference on {test_file}...')
        fname = os.path.basename(test_file).split('.')[0]
        test_dataset = ProteinVariantDatset(**data_configs, 
                                            variant_file=test_file, 
                                            split='test', 
                                            phenotype_vocab=phenotype_vocab, 
                                            protein_tokenizer=protein_tokenizer, 
                                            text_tokenizer=text_tokenizer,
                                            #  var_db=var_db,
                                            prot_var_cache=prot_var_cache,
                                            mode='eval',
                                            update_var_cache=False)
        logging.info('{} variants loaded'.format(len(test_dataset)))
        test_collator = ProteinVariantDataCollator(test_dataset.get_protein_data(), protein_tokenizer, text_tokenizer, phenotype_vocab=phenotype_vocab, 
                                                use_prot_desc=True, max_protein_length=data_configs['max_protein_seq_length'], mode='eval', 
                                                has_phenotype_label=test_dataset.has_phenotype_label)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], collate_fn=test_collator)
        # train_labels, train_scores, train_vars, train_pheno_results = inference(model, device, train_loader, pheno_vocab_emb=all_pheno_embs, topk=100)
        # val_labels, val_scores, val_vars, val_pheno_results = inference(model, device, validation_loader, pheno_vocab_emb=all_pheno_embs, topk=100)
        test_labels, test_scores, test_vars, test_pheno_results = inference(model, device, test_loader, pheno_vocab_emb=all_pheno_embs, topk=100)

        _save_scores(test_vars, test_labels, test_scores, fname, epoch='', exp_dir=str(exp_path), mode='eval')
        # np.save(pheno_result_path / 'train_pheno_similarity.npy', train_topk_results['similarities'])
        # np.save(pheno_result_path / 'test_pheno_similarity.npy', test_topk_results['similarities'])
        # np.save(pheno_result_path / 'val_pheno_similarity.npy', val_topk_results['similarities'])

        result_dict = {'topk_scores': test_pheno_results['topk_scores'],
                    'topk_indices': test_pheno_results['topk_indices']}
        if 'pos_pheno_idx' in test_pheno_results:
            result_dict.update({'label': test_pheno_results['pos_pheno_idx']})

        with open(exp_path / f'{fname}_topk.pkl', 'wb') as f_pkl:
            pickle.dump(result_dict, f_pkl)

        save_pheno_results(test_pheno_results, exp_path, name=fname, save_emb=True)
        logging.info('Done!')

    np.save(exp_path / 'phenotype_emb.npy', all_pheno_embs.detach().cpu().numpy())
