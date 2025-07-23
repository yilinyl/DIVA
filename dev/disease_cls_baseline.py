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
from models.model_utils import embed_phenotypes
from models.baseline_models import VariantDiseaseClassifier
# torch.set_default_dtype(torch.float32)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="configuration file", default='./configs/dis_var_cls_config.yaml')
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


def train_epoch(model, optimizer, device, data_loader, diagnostic=None, w_l=0.5, opt_patho=True,):
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

        desc_feat_dict = load_input_to_device(batch_data['desc_input_feat'], device)
        variant_data = load_input_to_device(batch_data['variant'], device=device, exclude_keys=['var_names'])
        for key in ['var_seq_input_feat', 'masked_seq_input_feat']:
            variant_data[key] = load_input_to_device(variant_data[key], device=device)
        batch_labels = batch_data['variant']['label'].unsqueeze(1).to(device)
        optimizer.zero_grad()
        patho_logits, batch_weights, seq_pheno_emb, batch_dis_cls_logits = model(variant_data, desc_feat_dict)
        patho_loss = model.patho_loss_fn(patho_logits, batch_labels.float())
        if batch_data['variant']['infer_phenotype']:
            pos_pheno_idx = variant_data['pos_pheno_idx']

        if opt_patho:
            if batch_data['variant']['infer_phenotype']:
                dis_ce_loss = model.disease_loss_fn(batch_dis_cls_logits, pos_pheno_idx)
                loss = dis_ce_loss + patho_loss * w_l
            else:
                loss = patho_loss
        else:
            dis_ce_loss = model.disease_loss_fn(batch_dis_cls_logits, pos_pheno_idx)
            loss = dis_ce_loss

        loss.backward()
        optimizer.step()
        
        loss_ = loss.detach().item()
        size = batch_labels.size()[0]
        cur_pheno_size = batch_labels.sum().item()
        running_patho_loss += patho_loss.detach().item() * size
        if batch_data['variant']['infer_phenotype']:
            running_pheno_loss += dis_ce_loss.detach().item() * cur_pheno_size
        # running_loss += loss_* size
        n_sample += size
        n_pheno_sample += cur_pheno_size

        # if diagnostic and batch_idx == 5:
        #     diagnostic.print_diagnostics()
        #     break
    epoch_patho_loss = running_patho_loss / n_sample
    epoch_pheno_loss = running_pheno_loss / n_pheno_sample
    epoch_loss = w_l * epoch_patho_loss + epoch_pheno_loss
    # epoch_loss = running_loss / n_sample
    
    return epoch_patho_loss, epoch_pheno_loss, epoch_loss, optimizer


def eval_epoch(model, device, data_loader):
    model.eval()
    
    all_vars, all_scores, all_labels, all_pheno_scores = [], [], [], []
    all_pheno_pos_scores_seq, all_pheno_neg_scores_seq = [], []
    all_seq_pheno_emb_pred, all_dis_cls_probs = [], []
    all_pheno_emb_label, all_pos_pheno_descs, all_pos_pheno_idx = [], [], []
    all_patho_vars, all_pheno_emb_neg, all_pheno_neg_scores = [], [], []
    all_disease_scores = []
    adjusted_weights = []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            # seq_feat_dict = load_input_to_device(batch_data['seq_input_feat'], device)
            desc_feat_dict = load_input_to_device(batch_data['desc_input_feat'], device)
            variant_data = load_input_to_device(batch_data['variant'], device=device, exclude_keys=['var_names'])
            for key in ['var_seq_input_feat', 'masked_seq_input_feat']:
                variant_data[key] = load_input_to_device(variant_data[key], device=device)
            batch_labels = batch_data['variant']['label'].unsqueeze(1).to(device)
            logit_diff, batch_weights, seq_pheno_emb, batch_dis_cls_logits = model(variant_data, desc_feat_dict)
            
            # batch_patho_scores = torch.softmax(patho_logits, 1)[:, 1]
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
                pos_pheno_idx = batch_data['variant']['pos_pheno_idx']
                batch_dis_cls_probs = torch.softmax(batch_dis_cls_logits.detach(), 1)
                pheno_score_pos = batch_dis_cls_probs.gather(1, pos_pheno_idx.unsqueeze(1)).squeeze(1)
                all_pheno_scores.append(pheno_score_pos.detach().cpu().numpy())
                # all_dis_cls_probs.append(batch_dis_cls_probs.detach().cpu().numpy())
                all_seq_pheno_emb_pred.append(seq_pheno_emb.detach().cpu().numpy())
                all_disease_scores.append(batch_dis_cls_probs.detach().cpu().numpy())

                all_patho_vars.extend(batch_data['variant']['pheno_var_names'])
                all_pos_pheno_descs.extend(batch_data['variant']['pos_pheno_name'])
                all_pos_pheno_idx.extend(pos_pheno_idx.detach().cpu().tolist())

        all_labels = np.concatenate(all_labels, 0)
        all_scores = np.concatenate(all_scores, 0)
        all_weights = np.concatenate(adjusted_weights, 0)  # learned weights for pathogenicity (scaler if only PLM used)
        all_pheno_scores = np.concatenate(all_pheno_scores, 0)
        # all_pos_pheno_idx = np.concatenate(all_pos_pheno_idx, 0)

        all_seq_pheno_emb_pred = np.concatenate(all_seq_pheno_emb_pred, 0)
        # all_pheno_emb_label = np.concatenate(all_pheno_emb_label, 0)

    all_pheno_results = {'var_names': all_patho_vars,
                         'label': all_labels,
                         'pos_pheno_desc': all_pos_pheno_descs,
                         'pos_pheno_idx': np.array(all_pos_pheno_idx),
                         'seq_pred_emb': all_seq_pheno_emb_pred,
                         'pos_score': all_pheno_scores,
                         'disease_scores': np.concatenate(all_disease_scores, 0)}
    
    return all_labels, all_scores, all_vars, all_weights, all_pheno_results


def save_pheno_results(pheno_result_dict, save_path, epoch, split, save_emb=False, save_cls_scores=False):
    if isinstance(save_path, str):
        save_path = Path(save_path)

    df_pheno_results = pd.DataFrame({'prot_var_id': pheno_result_dict['var_names'], 
                                    #  'label': pheno_result_dict['label'],
                                     'phenotype': pheno_result_dict['pos_pheno_desc'], 
                                     'pos_score': pheno_result_dict['pos_score']})

    df_pheno_results.to_csv(save_path / f'ep{epoch}_{split}_pheno_score.tsv', sep='\t', index=False)
    if save_emb:
        np.save(save_path / f'ep{epoch}_{split}_pheno_seq_pred_emb.npy', pheno_result_dict['seq_pred_emb'])
    
    if save_cls_scores and 'disease_scores' in pheno_result_dict:
        np.save(save_path / f'ep{epoch}_{split}_dis_cls_probs.npy', pheno_result_dict['disease_scores'])
        
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

    # pheno_dataset = PhenotypeDataset(phenotype_vocab, pheno_desc_dict, use_desc=data_configs['use_pheno_desc'])
    vocab_size = len(phenotype_vocab)
    # pheno_collator = TextDataCollator(text_tokenizer, padding=True)
    # phenotype_loader = DataLoader(pheno_dataset, batch_size=config['pheno_batch_size'], collate_fn=pheno_collator, shuffle=False)
    train_dataset = ProteinVariantDatset(**data_configs, 
                                         variant_file=data_configs['input_file']['train'], 
                                         split='train', 
                                         phenotype_vocab=phenotype_vocab, 
                                         protein_tokenizer=protein_tokenizer, 
                                         text_tokenizer=text_tokenizer,
                                         pheno_desc_dict=pheno_desc_dict,
                                        #  use_struct_neighbor=data_configs['use_struct_neighbor'],
                                         afmis_root=afmis_root,
                                         access_to_context=True)
    
    prot_var_cache = train_dataset.get_protein_cache()
    val_dataset = ProteinVariantDatset(**data_configs, 
                                         variant_file=data_configs['input_file']['val'], 
                                         split='val', 
                                         phenotype_vocab=phenotype_vocab, 
                                         protein_tokenizer=protein_tokenizer, 
                                         text_tokenizer=text_tokenizer,
                                         pheno_desc_dict=pheno_desc_dict,
                                        #  var_db=var_db,
                                         prot_var_cache=prot_var_cache,
                                        #  use_struct_neighbor=data_configs['use_struct_neighbor'],
                                         afmis_root=afmis_root,
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
                                        #  use_struct_neighbor=data_configs['use_struct_neighbor'],
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

    model = VariantDiseaseClassifier(seq_encoder=seq_encoder,
                                    text_encoder=text_encoder,
                                    n_residue_types=protein_tokenizer.vocab_size,
                                    hidden_size=model_args['hidden_size'],
                                    n_classes=vocab_size,
                                    use_desc=True,
                                    pad_label_idx=-100,
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
        save_cls_probs = False
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
        train_patho_loss, train_pheno_loss, train_loss, optimizer = train_epoch(model, optimizer, device, train_loader, w_l=model_args['w_l'], opt_patho=train_pathogenicity)
        train_labels, train_scores, train_vars, train_adj_weights, train_pheno_results = eval_epoch(model, device, train_loader)
        train_aupr = compute_aupr(train_labels, train_scores)
        train_auc = compute_roc(train_labels, train_scores)
        train_topk_acc = compute_topk_acc(train_pheno_results['pos_pheno_idx'], train_pheno_results['disease_scores'], topk_lst=model_args['topk'], label_lst=list(range(len(phenotype_vocab))))
        # seq_weight = train_pheno_results.get('seq_weight', 1)
        data_name = 'train'
        logging.info(f'<{data_name}> loss={train_loss:.4f} patho-loss={train_patho_loss:.4f} pheno-loss={train_pheno_loss:.4f} '
                     f'auPR={train_aupr:.4f} auROC={train_auc:.4f} top{topk_max}_acc={train_topk_acc[topk_max]:.4f}')

        val_labels, val_scores, val_vars, val_adj_weights, val_pheno_results = eval_epoch(model, device, validation_loader)
        # lr_scheduler.step(val_patho_loss)
        val_aupr = compute_aupr(val_labels, val_scores)
        val_auc = compute_roc(val_labels, val_scores)
        val_topk_acc = compute_topk_acc(val_pheno_results['pos_pheno_idx'], val_pheno_results['disease_scores'], topk_lst=model_args['topk'], label_lst=list(range(len(phenotype_vocab))))

        data_name = 'validation'
        logging.info(f'<{data_name}> auPR={val_aupr:.4f} auROC={val_auc:.4f} top{topk_max}_acc={val_topk_acc[topk_max]:.4f}')

        test_labels, test_scores, test_vars, test_adj_weights, test_pheno_results = eval_epoch(model, device, test_loader)
        # print('# Loss: train= {0:.5f}; validation= {1:.5f}; test= {2:.5f};'.format(train_loss, val_loss, test_loss))
        test_aupr = compute_aupr(test_labels, test_scores)
        test_auc = compute_roc(test_labels, test_scores)
        test_topk_acc = compute_topk_acc(test_pheno_results['pos_pheno_idx'], test_pheno_results['disease_scores'], topk_lst=model_args['topk'], label_lst=list(range(len(phenotype_vocab))))

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
            # np.save(pheno_result_path / f'ep{epoch}_phenotype_emb.npy', all_pheno_embs.detach().cpu().numpy())
            
            # best_pheno_loss = val_loss_dict["epoch_pheno_loss"]
            save_emb = True
            save_cls_probs = True
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
            save_pheno_results(train_pheno_results, pheno_result_path, epoch, split='train', save_emb=save_emb, save_cls_scores=save_cls_probs)
            save_pheno_results(val_pheno_results, pheno_result_path, epoch, split='val', save_emb=save_emb, save_cls_scores=save_cls_probs)
            save_pheno_results(test_pheno_results, pheno_result_path, epoch, split='test', save_emb=save_emb, save_cls_scores=save_cls_probs)

        if tb_writer:
            tb_writer.add_pr_curve('Train/PR-curve', train_labels, train_scores, epoch)
            tb_writer.add_pr_curve('Test/PR-curve', test_labels, test_scores, epoch)
            tb_writer.add_pr_curve('Val/PR-curve', val_labels, val_scores, epoch)

            tb_writer.add_scalar('train/loss', train_loss, epoch)
            tb_writer.add_scalar('train/patho_loss', train_patho_loss, epoch)
            tb_writer.add_scalar('train/pheno_loss', train_pheno_loss, epoch)
            
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
        save_pheno_results(best_results[key][-1], pheno_result_path, epoch=best_epoch, split=f'best_{key}', save_emb=True)

    
if __name__ == '__main__':
    main()