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
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertModel, BertForMaskedLM, EsmForMaskedLM, AutoModelForMaskedLM

from data.dis_var_dataset import ProteinVariantDatset, ProteinVariantDataCollator, PhenotypeDataset, TextDataCollator
import logging
from datetime import datetime
from utils import str2bool, env_setup, load_input_to_device, _save_scores, load_config
from metrics import *
from dev.preprocess.utils import parse_fasta_info
from models.model_utils import sample_random_negative, select_hard_negatives, embed_phenotypes
from models.protein_encoder import DiseaseVariantAttnEncoder
from models.dis_var_models import DiseaseVariantEncoder
# torch.set_default_dtype(torch.float32)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="configuration file", default='./configs/dis_var_config.yaml')
    parser.add_argument("-c_fmt", "--config_fmt", help="configuration file format", default='yaml')
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)
    parser.add_argument('--tensorboard', type=str2bool, default=False,
                        help='Option to write log information in tensorboard')
    parser.add_argument('--data_dir', help='Data directory')
    parser.add_argument('--exp_dir', help='Directory for all training related files, e.g. checkpoints, log')
    parser.add_argument('--experiment', help='Experiment name')
    parser.add_argument('--log_level', default='info', help='Log level')
    parser.add_argument('--save_freq', type=int, default=1, help='Frequency to save models')
    parser.add_argument('--inf-check', type=str2bool, default=False,
                        help='add hooks to check for infinite module outputs and gradients')
    # args, unparsed = parser.parse_known_args()
    args = parser.parse_args()

    return args


def train_epoch(model, optimizer, device, data_loader, diagnostic=None, w_l=0.5, opt_patho=True, 
                pheno_loader=None, text_emb_update_freq=10, sample_method='random', n_neg_samples=100, hard_ratio=0.5):
    model.train()
    running_loss = 0
    running_patho_loss = 0
    running_pheno_loss = 0
    n_sample = 0
    n_pheno_sample = 0
    for batch_idx, batch_data in enumerate(data_loader):
        # TODO: check batch_data structure
        if not opt_patho and not batch_data['variant']['infer_phenotype']:
            continue
        if batch_data['use_alphamissense']:
            if batch_data['variant']['afmis_mask'].all():  # AlphaMissense score not available for all
                continue
        # if use_hardneg:
        if batch_idx % text_emb_update_freq == 0:
            all_pheno_emb_list = []
            with torch.no_grad():
                for idx, batch_pheno in enumerate(pheno_loader):
                    # pheno_input_dict = load_input_to_device(batch_pheno, device)
                    pheno_input_dict = batch_pheno.to(device)
                    batch_pheno_embs = model.get_pheno_emb(pheno_input_dict, proj=True, agg_opt='cls')
                    all_pheno_emb_list.append(batch_pheno_embs)
                all_pheno_embs = torch.cat(all_pheno_emb_list, dim=0)
            pheno_vocab_size = all_pheno_embs.size(0)

        # seq_feat_dict = load_input_to_device(batch_data['seq_input_feat'], device)
        desc_feat_dict = load_input_to_device(batch_data['desc_input_feat'], device)
        variant_data = load_input_to_device(batch_data['variant'], device=device, exclude_keys=['var_names'])
        for key in ['var_seq_input_feat', 'masked_seq_input_feat']:
            variant_data[key] = load_input_to_device(variant_data[key], device=device)
        batch_labels = batch_data['variant']['label'].unsqueeze(1).to(device)
        optimizer.zero_grad()
        # batch_pheno_feat = batch_data['phenotype']
        # TODO: parse phenotype information
        # seq_pheno_emb, pos_emb_proj, neg_emb_proj, mlm_logits, logit_diff = model(seq_feat_dict, variant_data)  # seq_pheno_emb: (pheno_vars_in_batch, hidden_size)
        patho_logits, batch_weights, seq_pheno_emb, pos_emb_proj, neg_emb_proj, struct_pheno_emb = model(variant_data, desc_feat_dict)  # seq_pheno_emb: (pheno_vars_in_batch, hidden_size)
        # patho_loss = model.patho_loss_fn(patho_logits, batch_labels.squeeze(-1))
        # shapes = batch_logits.size()
        # batch_logits = batch_logits.view(shapes[0]*shapes[1])

        # patho_loss = model.pathogenicity_loss(logit_diff, batch_labels)
        patho_loss = model.patho_loss_fn(patho_logits, batch_labels.float())
        if batch_data['variant']['infer_phenotype']:
            pos_pheno_idx = variant_data['pos_pheno_idx']
            if sample_method == 'hard':
                neg_pheno_idx = select_hard_negatives(seq_pheno_emb, all_pheno_embs, pos_pheno_idx, n_negatives=n_neg_samples)
            elif sample_method == 'mix':
                n_hard_neg = int(n_neg_samples * hard_ratio)
                n_rand_neg = n_neg_samples - n_hard_neg
                hard_neg_pheno_idx = select_hard_negatives(seq_pheno_emb, all_pheno_embs, pos_pheno_idx, n_negatives=n_hard_neg)
                mask_idx = torch.cat([pos_pheno_idx.unsqueeze(1), hard_neg_pheno_idx], dim=1)
                rand_neg_pheno_idx = sample_random_negative(pheno_vocab_size, mask_idx, n_neg=n_rand_neg)
                neg_pheno_idx = torch.cat([hard_neg_pheno_idx, rand_neg_pheno_idx], dim=1)
            else:
                # neg_pheno_idx = variant_data['neg_pheno_idx'].unsqueeze(1)
                neg_pheno_idx = sample_random_negative(pheno_vocab_size, pos_pheno_idx, n_neg=n_neg_samples)
        if opt_patho:
            if batch_data['variant']['infer_phenotype']:
                # pos_pheno_idx = variant_data['pos_pheno_idx']
                # if use_hardneg:
                contrast_loss = model.info_nce_loss(seq_pheno_emb, all_pheno_embs, pos_pheno_idx, neg_pheno_idx)
                    # neg_pheno_idx = select_hard_negatives(seq_pheno_emb, all_pheno_embs, pos_pheno_idx)
                # else:
                #     contrast_loss = model.contrast_nce_loss(seq_pheno_emb, pos_emb_proj, neg_emb_proj)
                    # seq_contrast_loss, struct_contrast_loss, contrast_loss = model.contrast_loss(seq_pheno_emb, pos_emb_proj, neg_emb_proj, struct_pheno_emb, struct_mask=variant_data['has_struct_context'])
                loss = contrast_loss + patho_loss * w_l
            else:
                loss = patho_loss
        else:
            # assert batch_data['variant']['infer_phenotype']
            # pos_pheno_idx = variant_data['pos_pheno_idx']
            # if use_hardneg:
            contrast_loss = model.info_nce_loss(seq_pheno_emb, all_pheno_embs, pos_pheno_idx, neg_pheno_idx)
            # else:
            #     contrast_loss = model.contrast_nce_loss(seq_pheno_emb, pos_emb_proj, neg_emb_proj)
                # seq_contrast_loss, struct_contrast_loss, contrast_loss = model.contrast_loss(seq_pheno_emb, pos_emb_proj, neg_emb_proj, struct_pheno_emb, struct_mask=variant_data['has_struct_context'])
            loss = contrast_loss

        loss.backward()
        optimizer.step()
        
        loss_ = loss.detach().item()
        size = batch_labels.size()[0]
        cur_pheno_size = batch_labels.sum().item()
        running_patho_loss += patho_loss.detach().item() * size
        if batch_data['variant']['infer_phenotype']:
            running_pheno_loss += contrast_loss.detach().item() * cur_pheno_size
        # running_loss += loss_* size
        n_sample += size
        n_pheno_sample += cur_pheno_size

        if diagnostic and batch_idx == 5:
            diagnostic.print_diagnostics()
            break
    epoch_patho_loss = running_patho_loss / n_sample
    epoch_pheno_loss = running_pheno_loss / n_pheno_sample
    epoch_loss = w_l * epoch_patho_loss + epoch_pheno_loss
    # epoch_loss = running_loss / n_sample
    
    return epoch_patho_loss, epoch_pheno_loss, epoch_loss, optimizer


def eval_epoch(model, device, data_loader, pheno_vocab_emb, w_l=0.5, use_struct_neighbor=False):
    model.eval()
    
    all_vars, all_scores, all_labels, all_pheno_scores = [], [], [], []
    all_pheno_pos_scores_seq, all_pheno_neg_scores_seq = [], []
    all_pheno_pos_scores_str, all_pheno_neg_scores_str = [], []
    all_seq_pheno_emb_pred, all_str_pheno_emb_pred, all_struct_mask = [], [], []
    all_pheno_emb_label, all_pos_pheno_descs, all_pos_pheno_idx = [], [], []
    all_patho_vars, all_pheno_emb_neg, all_pheno_neg_scores = [], [], []
    all_similarities, seq_similarities, str_similarities = [], [], []
    adjusted_weights = []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            # seq_feat_dict = load_input_to_device(batch_data['seq_input_feat'], device)
            desc_feat_dict = load_input_to_device(batch_data['desc_input_feat'], device)
            variant_data = load_input_to_device(batch_data['variant'], device=device, exclude_keys=['var_names'])
            for key in ['var_seq_input_feat', 'masked_seq_input_feat']:
                variant_data[key] = load_input_to_device(variant_data[key], device=device)
            batch_labels = batch_data['variant']['label'].unsqueeze(1).to(device)

            # seq_pheno_emb, pos_emb_proj, neg_emb_proj, mlm_logits, logit_diff = model(seq_feat_dict, variant_data, desc_feat_dict)
            logit_diff, batch_weights, seq_pheno_emb, pos_emb_proj, neg_emb_proj, struct_pheno_emb = model(variant_data, desc_feat_dict)  # seq_pheno_emb: (pheno_vars_in_batch, hidden_size)
            # patho_loss = model.patho_loss_fn(patho_logits, batch_labels.squeeze(-1))
            
            # patho_loss = model.pathogenicity_loss(logit_diff, batch_labels)
            # patho_loss = model.patho_loss_fn(logit_diff, batch_labels.float())
            if batch_data['variant']['infer_phenotype']:
                var_struct_mask = variant_data['has_struct_context']
            
            # batch_patho_scores = torch.softmax(patho_logits, 1)[:, 1]
            # batch_patho_scores = torch.sigmoid(logit_diff)
            batch_patho_scores = logit_diff
            if batch_patho_scores.ndim > 1:
                batch_patho_scores = batch_patho_scores.squeeze(-1)

            all_scores.append(batch_patho_scores.detach().cpu().numpy())
            all_vars.extend(batch_data['variant']['var_names'])
            all_labels.append(batch_labels.squeeze(1).detach().cpu().numpy())
            if model.adjust_logits:
                adjusted_weights.append(batch_weights.squeeze(-1).detach().cpu().numpy())
            else:
                adjusted_weights.append(np.ones(batch_labels.size(0)))

            if batch_data['variant']['infer_phenotype']:
                pheno_score_pos_seq = model.calibrate_dis_score(torch.cosine_similarity(seq_pheno_emb, pos_emb_proj))
                pheno_score_neg_seq = model.calibrate_dis_score(torch.cosine_similarity(seq_pheno_emb, neg_emb_proj))
                pheno_score_pos = pheno_score_pos_seq.clone()
                pheno_score_neg = pheno_score_neg_seq.clone()

                pheno_score_pos_str = torch.zeros_like(pheno_score_pos_seq, device=pheno_score_pos_seq.device)
                pheno_score_neg_str = torch.zeros_like(pheno_score_neg_seq, device=pheno_score_neg_seq.device)
                if variant_data['use_struct']:
                    seq_weight = torch.sigmoid(model.alpha.detach()).item() * model.seq_weight_scaler
                    # pheno_score_pos_str = torch.zeros_like(pheno_score_pos_seq, device=pheno_score_pos_seq.device)
                    pheno_score_pos_str[var_struct_mask] = model.calibrate_dis_score(torch.cosine_similarity(struct_pheno_emb, pos_emb_proj[var_struct_mask]))
                    pheno_score_pos[var_struct_mask] = seq_weight * pheno_score_pos_seq[var_struct_mask] + (1 - seq_weight) * pheno_score_pos_str[var_struct_mask]

                    # pheno_score_neg_str = torch.zeros_like(pheno_score_neg_seq, device=pheno_score_neg_seq.device)
                    pheno_score_neg_str[var_struct_mask] = model.calibrate_dis_score(torch.cosine_similarity(struct_pheno_emb, neg_emb_proj[var_struct_mask]))
                    pheno_score_neg[var_struct_mask] = seq_weight * pheno_score_neg_seq[var_struct_mask] + (1 - seq_weight) * pheno_score_neg_str[var_struct_mask]

                    all_str_pheno_emb_pred.append(struct_pheno_emb.detach().cpu().numpy())
                all_struct_mask.extend(var_struct_mask)

                all_pheno_scores.append(pheno_score_pos.detach().cpu().numpy())
                all_pheno_neg_scores.append(pheno_score_neg.detach().cpu().numpy())

                all_pheno_pos_scores_seq.append(pheno_score_pos_seq.detach().cpu().numpy())
                all_pheno_neg_scores_seq.append(pheno_score_neg_seq.detach().cpu().numpy())
                all_pheno_pos_scores_str.append(pheno_score_pos_str.detach().cpu().numpy())
                all_pheno_neg_scores_str.append(pheno_score_neg_str.detach().cpu().numpy())

                all_seq_pheno_emb_pred.append(seq_pheno_emb.detach().cpu().numpy())
                all_pheno_emb_label.append(pos_emb_proj.detach().cpu().numpy())
                all_pheno_emb_neg.append(neg_emb_proj.detach().cpu().numpy())

                all_patho_vars.extend(batch_data['variant']['pheno_var_names'])
                all_pos_pheno_descs.extend(batch_data['variant']['pos_pheno_name'])
                all_pos_pheno_idx.extend(batch_data['variant']['pos_pheno_idx'].detach().cpu().tolist())
                # all_neg_pheno_descs.extend(batch_data['variant']['neg_pheno_desc'])
                
                cos_sim_seq = torch.cosine_similarity(seq_pheno_emb.unsqueeze(1), pheno_vocab_emb.unsqueeze(0), dim=-1)
                cos_sim_all = cos_sim_seq.clone()
                if variant_data['use_struct']:
                    cos_sim_str = torch.zeros_like(cos_sim_seq, device=cos_sim_seq.device)
                    cos_sim_str[var_struct_mask] = torch.cosine_similarity(struct_pheno_emb.unsqueeze(1), pheno_vocab_emb.unsqueeze(0), dim=-1)
                    cos_sim_all[var_struct_mask] = seq_weight * cos_sim_seq[var_struct_mask] + (1 - seq_weight) * cos_sim_str[var_struct_mask]
                    
                    seq_similarities.append(cos_sim_seq.detach().cpu().numpy())
                    str_similarities.append(cos_sim_str.detach().cpu().numpy())
                
                # topk_scores, topk_indices = torch.topk(cos_sim_all, k=topk, dim=1)
                # all_topk_scores.append(topk_scores.detach().cpu().numpy())
                # all_topk_indices.append(topk_indices.detach().cpu().numpy())
                all_similarities.append(cos_sim_all.detach().cpu().numpy())

        all_labels = np.concatenate(all_labels, 0)
        all_scores = np.concatenate(all_scores, 0)
        all_weights = np.concatenate(adjusted_weights, 0)  # learned weights for pathogenicity (scaler if only PLM used)
        all_pheno_scores = np.concatenate(all_pheno_scores, 0)
        all_pheno_neg_scores = np.concatenate(all_pheno_neg_scores, 0)
        # all_pos_pheno_idx = np.concatenate(all_pos_pheno_idx, 0)

        all_seq_pheno_emb_pred = np.concatenate(all_seq_pheno_emb_pred, 0)
        all_pheno_emb_label = np.concatenate(all_pheno_emb_label, 0)
        all_pheno_emb_neg = np.concatenate(all_pheno_emb_neg, 0)

    all_pheno_results = {'var_names': all_patho_vars,
                         'label': all_labels,
                         'pos_pheno_desc': all_pos_pheno_descs,
                         'pos_pheno_idx': np.array(all_pos_pheno_idx),
                        #  'neg_pheno_desc': all_neg_pheno_descs,
                         'seq_pred_emb': all_seq_pheno_emb_pred,
                         'pos_emb': all_pheno_emb_label,
                         'neg_emb': all_pheno_emb_neg,
                         'pos_score': all_pheno_scores,
                         'neg_score': all_pheno_neg_scores,
                         'similarities': np.concatenate(all_similarities, 0)}
    
    if use_struct_neighbor:
        all_pheno_results.update({
            'seq_weight': seq_weight,
            'str_pred_emb': np.concatenate(all_str_pheno_emb_pred, 0),
            'pos_score_seq': np.concatenate(all_pheno_pos_scores_seq, 0),
            'neg_score_seq': np.concatenate(all_pheno_neg_scores_seq, 0),
            'pos_score_str': np.concatenate(all_pheno_pos_scores_str, 0),
            'neg_score_str': np.concatenate(all_pheno_neg_scores_str, 0),
            'seq_similarities': np.concatenate(seq_similarities, 0),
            'str_similarities': np.concatenate(str_similarities, 0),
            'use_struct_neighbor': np.array(all_struct_mask)
        })

    # if eval_topk:
    #     topk_results = {'similarities': np.concatenate(all_similarities, 0),
    #                     'topk_scores': np.concatenate(all_topk_scores, 0),
    #                     'topk_indices': np.concatenate(all_topk_indices, 0)}
    return all_labels, all_scores, all_vars, all_weights, all_pheno_results
    # return loss_dict, all_labels, all_scores, all_vars, all_pheno_results


def save_pheno_results(pheno_result_dict, save_path, epoch, split, save_emb=False, compute_tsne=False, use_struct=False):
    if isinstance(save_path, str):
        save_path = Path(save_path)

    df_pheno_results = pd.DataFrame({'prot_var_id': pheno_result_dict['var_names'], 
                                    #  'label': pheno_result_dict['label'],
                                     'phenotype': pheno_result_dict['pos_pheno_desc'], 
                                    #  'neg_phenotype': pheno_result_dict['neg_pheno_desc'], 
                                     'pos_score': pheno_result_dict['pos_score'],
                                     'neg_score': pheno_result_dict['neg_score']})
    if use_struct:
        for key in ['pos_score_seq', 'neg_score_seq', 'pos_score_str', 'neg_score_str', 'use_struct_neighbor']:
            df_pheno_results[key] = pheno_result_dict[key]

    if compute_tsne:
        pred_emb_tsne = compute_tsne(pheno_result_dict['seq_pred_emb'])
        df_pheno_results = pd.concat([df_pheno_results, pd.DataFrame(pred_emb_tsne)], axis=1).rename(columns={0: 'tsne1', 1: 'tsne2'})

    df_pheno_results.to_csv(save_path / f'ep{epoch}_{split}_pheno_score.tsv', sep='\t', index=False)
    if save_emb:
        # pd.DataFrame(pheno_result_dict['pred_emb']).to_csv(save_path / f'ep{epoch}_{split}_pheno_pred_emb.tsv', sep='\t', index=False, header=False)
        np.save(save_path / f'ep{epoch}_{split}_pheno_seq_pred_emb.npy', pheno_result_dict['seq_pred_emb'])
        if use_struct:
            np.save(save_path / f'ep{epoch}_{split}_pheno_str_pred_emb.npy', pheno_result_dict['str_pred_emb'])
        # pd.DataFrame(pheno_result_dict['neg_emb']).to_csv(save_path / f'ep{epoch}_{split}_pheno_neg_emb.tsv', sep='\t', index=False, header=False)


def main():
    args = parse_args()
    config = load_config(args.config, format=args.config_fmt)

    config, device = env_setup(args, config)

    data_configs = config['dataset']
    model_args = config['model']

    exp_dir = config['exp_dir']
    model_save_path = Path(exp_dir) / 'checkpoints'

    if not model_save_path.exists():
        model_save_path.mkdir(parents=True)

    result_path = Path(exp_dir) / 'result'
    # if not result_path.exists():
    #     result_path.mkdir(parents=True)

    pheno_result_path = result_path / 'phenotype'
    if not pheno_result_path.exists():
        pheno_result_path.mkdir(parents=True)

    if args.tensorboard:
        tb_writer = SummaryWriter(log_dir='{}/tensorboard'.format(config['exp_dir']))
    else:
        tb_writer = None

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

    prot2comb_seq = None
    if data_configs['use_struct_vocab']:
        if os.path.exists(data_configs['struct_seq_file']):
            with open(data_configs['struct_seq_file']) as f_js:
                prot2comb_seq = json.load(f_js)
        else:
            data_configs['use_struct_vocab'] = False

    if data_configs['use_struct_neighbor']:
        if not os.path.exists(data_configs['pdb_graph_dir']) and not os.path.exists(data_configs['af_graph_dir']):
            data_configs['use_struct_neighbor'] = False

    data_configs['seq_dict'] = prot2seq
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

    if data_configs['disease_name_map_file']:
        with open(data_configs['disease_name_map_file']) as f:
            dis_name_map_dict = json.load(f)
    else:
        dis_name_map_dict = None

    pheno_dataset = PhenotypeDataset(phenotype_vocab, pheno_desc_dict, use_desc=data_configs['use_pheno_desc'])
    pheno_collator = TextDataCollator(text_tokenizer, padding=True)
    phenotype_loader = DataLoader(pheno_dataset, batch_size=config['pheno_batch_size'], collate_fn=pheno_collator, shuffle=False)
    train_dataset = ProteinVariantDatset(**data_configs, 
                                         variant_file=data_configs['input_file']['train'], 
                                         split='train', 
                                         phenotype_vocab=phenotype_vocab, 
                                         disease_name_map_dict=dis_name_map_dict,
                                         protein_tokenizer=protein_tokenizer, 
                                         text_tokenizer=text_tokenizer,
                                         pheno_desc_dict=pheno_desc_dict,
                                        #  use_struct_neighbor=data_configs['use_struct_neighbor'],
                                         comb_seq_dict=prot2comb_seq,
                                         afmis_root=afmis_root,
                                         access_to_context=True)
    if data_configs['use_struct_neighbor']:
        logging.info('Average structural neighbors per variant in training: {:.2f}'.format(train_dataset.average_struct_neighbors()))
    # var_db = pd.read_csv(data_root / data_configs['input_file']['train']).query('label == 1').\
    #     drop_duplicates([data_configs['pid_col'], data_configs['pos_col'], data_configs['pheno_col']])
    prot_var_cache = train_dataset.get_protein_cache()
    val_dataset = ProteinVariantDatset(**data_configs, 
                                         variant_file=data_configs['input_file']['val'], 
                                         split='val', 
                                         phenotype_vocab=phenotype_vocab, 
                                         disease_name_map_dict=dis_name_map_dict,
                                         protein_tokenizer=protein_tokenizer, 
                                         text_tokenizer=text_tokenizer,
                                         pheno_desc_dict=pheno_desc_dict,
                                        #  var_db=var_db,
                                         prot_var_cache=prot_var_cache,
                                        #  use_struct_neighbor=data_configs['use_struct_neighbor'],
                                         comb_seq_dict=prot2comb_seq,
                                         afmis_root=afmis_root,
                                         access_to_context=False)  # context variants in validation set not visible to each other
    # val_variants = pd.read_csv(data_root / data_configs['input_file']['val']).query('label == 1').\
    #     drop_duplicates([data_configs['pid_col'], data_configs['pos_col'], data_configs['pheno_col']])
    # var_db = pd.concat([var_db, val_variants])
    if data_configs['use_struct_neighbor']:
        logging.info('Average structural neighbors per variant in validation: {:.2f}'.format(val_dataset.average_struct_neighbors()))

    prot_var_cache = val_dataset.get_protein_cache()
    
    test_dataset = ProteinVariantDatset(**data_configs, 
                                         variant_file=data_configs['input_file']['test'], 
                                         split='test', 
                                         phenotype_vocab=phenotype_vocab, 
                                         disease_name_map_dict=dis_name_map_dict,
                                         protein_tokenizer=protein_tokenizer, 
                                         text_tokenizer=text_tokenizer,
                                         pheno_desc_dict=pheno_desc_dict,
                                        #  var_db=var_db,
                                         prot_var_cache=prot_var_cache,
                                        #  use_struct_neighbor=data_configs['use_struct_neighbor'],
                                         comb_seq_dict=prot2comb_seq,
                                         afmis_root=afmis_root,
                                         access_to_context=False)
    if data_configs['use_struct_neighbor']:
        logging.info('Average structural neighbors per variant in test set: {:.2f}'.format(test_dataset.average_struct_neighbors()))

    # Initilize pretrained encoders:
    # seq_encoder = EsmForMaskedLM.from_pretrained(model_args['protein_lm_path'])
    seq_encoder = AutoModelForMaskedLM.from_pretrained(model_args['protein_lm_path'])
    text_encoder = BertForMaskedLM.from_pretrained(model_args['text_lm_path'])
    data_configs['use_alphamissense'] = train_dataset.use_alphamissense
    train_collator = ProteinVariantDataCollator(train_dataset.get_protein_data(), protein_tokenizer, text_tokenizer, phenotype_vocab=phenotype_vocab, 
                                                use_prot_desc=True, truncate_protein=data_configs['truncate_protein'], 
                                                max_protein_length=data_configs['max_protein_seq_length'], 
                                                half_window_size=data_configs['half_window_size'],
                                                context_agg_opt=data_configs['context_agg_option'], use_pheno_desc=data_configs['use_pheno_desc'], 
                                                pheno_desc_dict=pheno_desc_dict, use_struct_vocab=data_configs['use_struct_vocab'], use_alphamissense=data_configs['use_alphamissense'],
                                                use_struct_neighbor=data_configs['use_struct_neighbor'], struct_radius_cutoff=data_configs['struct_radius_cutoff'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], collate_fn=train_collator, shuffle=True)
    val_collator = ProteinVariantDataCollator(val_dataset.get_protein_data(), protein_tokenizer, text_tokenizer, phenotype_vocab=phenotype_vocab, 
                                              use_prot_desc=True, truncate_protein=data_configs['truncate_protein'], 
                                              max_protein_length=data_configs['max_protein_seq_length'], 
                                              half_window_size=data_configs['half_window_size'],
                                              context_agg_opt=data_configs['context_agg_option'], use_pheno_desc=data_configs['use_pheno_desc'], 
                                              pheno_desc_dict=pheno_desc_dict, use_struct_vocab=data_configs['use_struct_vocab'], use_alphamissense=data_configs['use_alphamissense'],
                                              use_struct_neighbor=data_configs['use_struct_neighbor'], struct_radius_cutoff=data_configs['struct_radius_cutoff'])
    validation_loader = DataLoader(val_dataset, batch_size=config['batch_size'], collate_fn=val_collator)
    test_collator = ProteinVariantDataCollator(test_dataset.get_protein_data(), protein_tokenizer, text_tokenizer, phenotype_vocab=phenotype_vocab, 
                                               use_prot_desc=True, truncate_protein=data_configs['truncate_protein'], 
                                               max_protein_length=data_configs['max_protein_seq_length'], 
                                               half_window_size=data_configs['half_window_size'],
                                               context_agg_opt=data_configs['context_agg_option'], use_pheno_desc=data_configs['use_pheno_desc'], 
                                               pheno_desc_dict=pheno_desc_dict, use_struct_vocab=data_configs['use_struct_vocab'], use_alphamissense=data_configs['use_alphamissense'],
                                               use_struct_neighbor=data_configs['use_struct_neighbor'], struct_radius_cutoff=data_configs['struct_radius_cutoff'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], collate_fn=test_collator)

    unfreeze_params = []
    if model_args['frozen_bert']:
        # prot_unfreeze_layers = ['esm.encoder.layer.11']
        for name, parameters in seq_encoder.named_parameters():
            parameters.requires_grad = False
            for tags in model_args['prot_bert_unfreeze']:
                if tags in name:
                    parameters.requires_grad = True
                    unfreeze_params.append(name)
                    break

        # text_unfreeze_layers = ['bert.encoder.layer.11']
        for name, parameters in text_encoder.named_parameters():
            parameters.requires_grad = False
            if model_args['num_warmup_epochs'] == 0:  # No warm-up
                for tags in model_args['text_bert_unfreeze']:
                    if tags in name:
                        parameters.requires_grad = True
                        unfreeze_params.append(name)
                        break
    # print(unfreeze_params)
    seq_encoder = seq_encoder.to(device)
    text_encoder = text_encoder.to(device)
    if data_configs['context_agg_option'] == 'stack':
        model = DiseaseVariantAttnEncoder(seq_encoder=seq_encoder,
                                    text_encoder=text_encoder,
                                    n_residue_types=protein_tokenizer.vocab_size,
                                    hidden_size=model_args['hidden_size'],
                                    use_desc=True,
                                    pad_label_idx=-100,
                                    dist_fn_name=model_args['dist_fn_name'],
                                    init_margin=model_args['margin'],
                                    freq_norm_factor=model_args['freq_norm_factor'],
                                    seq_weight_scaler=model_args['seq_weight_scaler'],
                                    pe_scalor=model_args['pe_scalor'],
                                    use_struct_vocab=data_configs['use_struct_vocab'],
                                    use_alphamissense=data_configs['use_alphamissense'],
                                    adjust_logits=model_args['adjust_logits'],
                                    device=device)
    else:  # concat
        model = DiseaseVariantEncoder(seq_encoder=seq_encoder,
                                    text_encoder=text_encoder,
                                    n_residue_types=protein_tokenizer.vocab_size,
                                    hidden_size=model_args['hidden_size'],
                                    use_desc=True,
                                    pad_label_idx=-100,
                                    calibration_fn_name=model_args['calibration_fn_name'],
                                    init_margin=model_args['margin'],
                                    nce_loss_temp=model_args['nce_loss_temp'],
                                    freq_norm_factor=model_args['freq_norm_factor'],
                                    seq_weight_scaler=model_args['seq_weight_scaler'],
                                    use_struct_vocab=data_configs['use_struct_vocab'],
                                    use_alphamissense=data_configs['use_alphamissense'],
                                    adjust_logits=model_args['adjust_logits'],
                                    device=device)
    total_param = 0
    total_param_with_grad = 0
    for p in model.parameters():
        if p.requires_grad:
            total_param_with_grad += p.numel()
        total_param += p.numel()
    logging.info(f'Model parameters (trainable/all): {total_param_with_grad} / {total_param}')

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['init_lr'])
    logging.info('Initializing optimizers...')

    logging.info("Training starts...")
    # best_ep_scores_train = None
    # best_ep_labels_train = None
    best_val_loss = float('inf')
    best_patho_loss = float('inf')
    best_pheno_loss = float('inf')
    best_weights = None
    best_optim = None
    best_epoch = 0
    best_results = {'train': None, 'test': None, 'val': None}
    best_topk_results = dict()
    best_val_acc = 0
    best_val_aupr = 0
    
    train_pathogenicity = config['train_pathogenicity']
    topk_max = max(model_args['topk'])
    
    # all_pheno_embs, pheno_sims = embed_phenotypes(model, device, phenotype_loader)  # TODO: uncomment later

    for epoch in range(config['epochs']):
        save_emb = False
        # logging.info('Epoch %d' % epoch)
        # if tb_writer:
        #     tb_writer.add_scalar("train/epoch", epoch)
        if epoch == model_args['num_warmup_epochs']:
            logging.info(f'Unfreeze selected text encoder parameters after epoch {epoch}')
            for name, parameters in text_encoder.named_parameters():
                for tags in model_args['text_bert_unfreeze']:
                    if tags in name:
                        parameters.requires_grad = True
                        unfreeze_params.append(name)
                        break
            total_param_with_grad = 0
            for p in model.parameters():
                if p.requires_grad:
                    total_param_with_grad += p.numel()
                total_param += p.numel()
            logging.info(f'Updated model parameters (trainable/all): {total_param_with_grad} / {total_param}')

        if (epoch > model_args['max_pathogenicity_epochs']) & train_pathogenicity:
            train_pathogenicity = False
            logging.info(f'Disable pathogenicity optimization after epoch {epoch}')
            for name, parameters in model.named_parameters():
                # protein_encoder.seq_encoder.esm.encoder.layer.32
                if name.startswith('protein_encoder.seq_encoder') and parameters.requires_grad:
                    parameters.requires_grad = False
                    logging.info(f'Freeze {name}')
            
            total_param_with_grad = 0
            for p in model.parameters():
                if p.requires_grad:
                    total_param_with_grad += p.numel()
                total_param += p.numel()
            logging.info(f'Updated model parameters (trainable/all): {total_param_with_grad} / {total_param}')
        logging.info('Epoch %d' % epoch)
        train_patho_loss, train_pheno_loss, train_loss, optimizer = train_epoch(model, optimizer, device, train_loader, w_l=model_args['w_l'], opt_patho=train_pathogenicity, 
                                                                                pheno_loader=phenotype_loader, text_emb_update_freq=model_args['pheno_emb_update_interval'], 
                                                                                sample_method=model_args['sample_method'], n_neg_samples=model_args['n_neg_samples'])
        # train_patho_loss, train_pheno_loss, train_loss, optimizer, contrast_optimizer = train_epoch_sep(model, optimizer, contrast_optimizer, device, train_loader, w_l=model_args['w_l'])
        all_pheno_embs = embed_phenotypes(model, device, phenotype_loader)
        # all_pheno_embs = all_pheno_embs.to(device)
        all_pheno_embs = torch.tensor(all_pheno_embs).to(device)
        train_labels, train_scores, train_vars, train_adj_weights, train_pheno_results = eval_epoch(model, device, train_loader, pheno_vocab_emb=all_pheno_embs, 
                                                                                                     w_l=model_args['w_l'], use_struct_neighbor=data_configs['use_struct_neighbor'])
        train_aupr = compute_aupr(train_labels, train_scores)
        train_auc = compute_roc(train_labels, train_scores)
        train_topk_acc = compute_topk_acc(train_pheno_results['pos_pheno_idx'], train_pheno_results['similarities'], topk_lst=model_args['topk'], label_lst=list(range(len(phenotype_vocab))))
        seq_weight = train_pheno_results.get('seq_weight', 1)
        data_name = 'train'
        logging.info(f'<{data_name}> loss={train_loss:.4f} patho-loss={train_patho_loss:.4f} pheno-loss={train_pheno_loss:.4f} '
                     f'auPR={train_aupr:.4f} auROC={train_auc:.4f} top{topk_max}_acc={train_topk_acc[topk_max]:.4f}')

        val_labels, val_scores, val_vars, val_adj_weights, val_pheno_results = eval_epoch(model, device, validation_loader, pheno_vocab_emb=all_pheno_embs, 
                                                                                          w_l=model_args['w_l'], use_struct_neighbor=data_configs['use_struct_neighbor'])
        # lr_scheduler.step(val_patho_loss)
        val_aupr = compute_aupr(val_labels, val_scores)
        val_auc = compute_roc(val_labels, val_scores)
        val_topk_acc = compute_topk_acc(val_pheno_results['pos_pheno_idx'], val_pheno_results['similarities'], topk_lst=model_args['topk'], label_lst=list(range(len(phenotype_vocab))))

        data_name = 'validation'
        logging.info(f'<{data_name}> auPR={val_aupr:.4f} auROC={val_auc:.4f} top{topk_max}_acc={val_topk_acc[topk_max]:.4f}')

        test_labels, test_scores, test_vars, test_adj_weights, test_pheno_results = eval_epoch(model, device, test_loader, pheno_vocab_emb=all_pheno_embs, 
                                                                                               w_l=model_args['w_l'], use_struct_neighbor=data_configs['use_struct_neighbor'])
        # print('# Loss: train= {0:.5f}; validation= {1:.5f}; test= {2:.5f};'.format(train_loss, val_loss, test_loss))
        test_aupr = compute_aupr(test_labels, test_scores)
        test_auc = compute_roc(test_labels, test_scores)
        test_topk_acc = compute_topk_acc(test_pheno_results['pos_pheno_idx'], test_pheno_results['similarities'], topk_lst=model_args['topk'], label_lst=list(range(len(phenotype_vocab))))

        data_name = 'test'
        logging.info(f'<{data_name}> auPR={test_aupr:.4f} auROC={test_auc:.4f} top{topk_max}_acc={test_topk_acc[topk_max]:.4f}')
        
        if val_aupr > best_val_aupr:
            best_val_aupr = val_aupr
            # best_patho_loss = val_loss_dict["epoch_patho_loss"]
            torch.save({'args': config, 
                        'state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict()},
                       model_save_path / 'best_patho_model.pt'.format(epoch))
            
        # if val_loss_dict["epoch_pheno_loss"] < best_pheno_loss:
        # if val_loss_dict["epoch_loss"] < best_val_loss:
        if val_topk_acc[topk_max] > best_val_acc:
            # best_val_loss = val_loss_dict["epoch_loss"]
            best_val_acc = val_topk_acc[topk_max]
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            best_optim = copy.deepcopy(optimizer.state_dict())
            
            best_results['train'] = (train_vars, train_labels, train_scores, train_adj_weights, train_pheno_results)
            best_results['test'] = (test_vars, test_labels, test_scores, test_adj_weights, test_pheno_results)
            best_results['val'] = (val_vars, val_labels, val_scores, val_adj_weights, val_pheno_results)
            np.save(pheno_result_path / f'ep{epoch}_phenotype_emb.npy', all_pheno_embs.detach().cpu().numpy())
            
            # best_pheno_loss = val_loss_dict["epoch_pheno_loss"]
            save_emb = True
            torch.save({'args': config, 
                        'state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict()},
                       model_save_path / 'best_pheno_model.pt'.format(epoch))
            
            best_topk_results['epoch'] = epoch
            best_topk_results['train'] = train_topk_acc
            best_topk_results['test'] = test_topk_acc
            best_topk_results['val'] = val_topk_acc
            if model_args['calibration_fn_name'] == 'logistic':
                best_topk_results['logistic_weight'] = model.calibrate_weight.detach().cpu().numpy().item()
                best_topk_results['logistic_bias'] = model.calibrate_bias.detach().cpu().numpy().item()

            with open(result_path / 'best_topk_result.json', 'w') as f_js:
                json.dump(best_topk_results, f_js, indent=2)

        if epoch % args.save_freq == 0:
            _save_scores(train_vars, train_labels, train_scores, 'train', weights=train_adj_weights, epoch=epoch, exp_dir=exp_dir)
            _save_scores(val_vars, val_labels, val_scores, 'val', weights=val_adj_weights, epoch=epoch, exp_dir=exp_dir)
            _save_scores(test_vars, test_labels, test_scores, 'test', weights=test_adj_weights, epoch=epoch, exp_dir=exp_dir)
            save_pheno_results(train_pheno_results, pheno_result_path, epoch, split='train', save_emb=save_emb, use_struct=data_configs['use_struct_neighbor'])
            save_pheno_results(val_pheno_results, pheno_result_path, epoch, split='val', save_emb=save_emb, use_struct=data_configs['use_struct_neighbor'])
            save_pheno_results(test_pheno_results, pheno_result_path, epoch, split='test', save_emb=save_emb, use_struct=data_configs['use_struct_neighbor'])

        if tb_writer:
            tb_writer.add_pr_curve('Train/PR-curve', train_labels, train_scores, epoch)
            tb_writer.add_pr_curve('Test/PR-curve', test_labels, test_scores, epoch)
            tb_writer.add_pr_curve('Val/PR-curve', val_labels, val_scores, epoch)

            tb_writer.add_scalar('train/loss', train_loss, epoch)
            tb_writer.add_scalar('train/patho_loss', train_patho_loss, epoch)
            tb_writer.add_scalar('train/pheno_loss', train_pheno_loss, epoch)
            # for loss_key in train_loss_dict.keys():
            #     tb_writer.add_scalar('train/{}'.format(loss_key.split('_', 1)[-1]), train_loss_dict[loss_key], epoch)
                # tb_writer.add_scalar('validation/{}'.format(loss_key.split('_', 1)[-1]), val_loss_dict[loss_key], epoch)
                # tb_writer.add_scalar('test/{}'.format(loss_key.split('_', 1)[-1]), test_loss_dict[loss_key], epoch)

            for k in model_args['topk']:
                tb_writer.add_scalar(f'train/top{k}_acc', train_topk_acc[k], epoch)
                tb_writer.add_scalar(f'validation/top{k}_acc', val_topk_acc[k], epoch)
                tb_writer.add_scalar(f'test/top{k}_acc', test_topk_acc[k], epoch)
            # tb_writer.add_embedding(best_results['train'][3]['pred_emb'], metadata=[best_results['train'][0]], 
            #                         metadata_header=['prot_var_id', 'phenotype'], tag='Train/Embedding')

    logging.info('Save best model at epoch {}:'.format(best_epoch))
    torch.save({'args': config, 'state_dict': best_weights,
                'optimizer_state_dict': best_optim},
               model_save_path / 'bestmodel-ep{}.pt'.format(best_epoch))
    for key in best_results:
        _save_scores(best_results[key][0], best_results[key][1], best_results[key][2], key, weights=best_results[key][3], epoch=best_epoch, exp_dir=exp_dir)
        save_pheno_results(best_results[key][-1], pheno_result_path, epoch=best_epoch, split=f'best_{key}', save_emb=True, use_struct=data_configs['use_struct_neighbor'])

    
if __name__ == '__main__':
    main()