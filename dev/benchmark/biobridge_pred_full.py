import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.abspath('..'))

from pathlib import Path
import json
import yaml
import copy
from typing import Dict, List, Optional, Tuple, Union
import argparse
import pickle
import logging
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertTokenizer, BertForMaskedLM, AutoModelForMaskedLM, PreTrainedModel, EsmModel

from dev.data.dis_var_dataset import ProteinVariantSeqDataset, ProteinVariantSeqCollator, TextDataCollator
from dev.utils import load_input_to_device, env_setup, load_config
from dev.preprocess.utils import parse_fasta_info
from dev.models.model_utils import sample_random_negative
from dev.benchmark.utils import extract_text_emb
from dev.benchmark.BioBridge.src.model import BindingModel
from dev.benchmark.BioBridge.src.inference import BridgeInference
from dev.benchmark.BioBridge.src.protein_encoder import load_protein_model


class ProteinDiseaseBridgeModel(nn.Module):
    def __init__(self,
                 seq_encoder: Union[nn.Module, PreTrainedModel], 
                 bridge_model: Union[nn.Module, BindingModel]
                 ):
        super(ProteinDiseaseBridgeModel, self).__init__()
        self.seq_encoder = seq_encoder
        self.bridge_model = bridge_model
        # assert seq_encoder.device == bridge_model.device
        self.device = seq_encoder.device
    
    def forward(self, variant_data):
        seq_feat_dict = {'input_ids': variant_data['var_seq_input_ids'],
                         'attention_mask': variant_data['var_seq_attention_mask']}
        prot_emb_raw = self.embed_seq(seq_feat_dict)
        # transformed protein embedding in disease space
        prot_emb_tr = self.bridge_transformation(
            emb_raw = prot_emb_raw,
            src_type = 1, # protein
            tgt_type = 2, # disease
            rel_type = 6, # associated with
        )
        return prot_emb_tr
    
    @torch.no_grad()
    def embed_seq(self, seq_feat_dict):
        self.seq_encoder.eval()
           # seq_emb_raw = torch.stack([seq_emb_raw[i, seq_feat_dict['attention_mask'][i, :]][1:-1].mean(dim=0) for i in range(len(seq_emb_raw))], dim=0)
        with torch.no_grad():
            output = self.seq_encoder(**seq_feat_dict)

        attention_mask = seq_feat_dict["attention_mask"]
        emb = output.last_hidden_state # (batch_size, seq_length, hidden_size)
        protein_attention_mask = attention_mask.bool()
        protein_embedding = torch.stack([emb[i, protein_attention_mask[i, :]][1:-1].mean(dim=0) for i in range(len(emb))], dim=0)
        return protein_embedding
    
    @torch.no_grad()
    def project_emb(self, emb_raw, src_type=2):
        emb_proj = self.bridge_model.projection(
            node_emb=emb_raw,
            node_type_id=src_type,  # 2 for disease
        )
        return emb_proj

    def bridge_transformation(
        self,
        emb_raw: torch.Tensor,
        src_type=1,  # protein
        tgt_type=2,  # disease
        rel_type=6  # associated with
        ):
        """Inference based on the trained Bridge model to project raw embeddings to the target space.

        Args:
            model (BindingModel): the trained Bridge model.
            x (torch.Tensor): the raw embeddings to be projected.
            src_type (int): the type of the source space.
            tgt_type (int): the type of the target space.
            rel_type (int): the type of the relation.
        """
        # if torch.cuda.is_available():
        #     x = x.to("cuda:0")
        #     model = model.to("cuda:0")
        
        self.bridge_model.eval()
        head_type_ids = torch.tensor([src_type] * len(emb_raw)).to(self.device)
        rel_type_ids = torch.tensor([rel_type] * len(emb_raw)).to(self.device)
        tail_type_ids = torch.tensor([tgt_type] * len(emb_raw)).to(self.device)
        output = self.bridge_model(
            head_emb=emb_raw,
            head_type_ids=head_type_ids,
            rel_type_ids=rel_type_ids,
            tail_type_ids=tail_type_ids,
        )
        return output['embeddings']


class BioBridgeTextDataset(Dataset):
    """
    Dataset for Protein sequence.

    Args:
        data_dir: the diractory need contain pre-train datasets.
        tokenizer: tokenizer used for encoding sequence.
    """

    def __init__(
        self,
        phenotypes: List[str],
        pheno_desc_dict: Dict[str, str] = None,
        use_desc: bool = True  # use phenotype description or not
    ):
        
        self.phenotypes = phenotypes
        self.pheno_desc_dict = pheno_desc_dict
        self.use_desc = use_desc and pheno_desc_dict is not None
        # self.tokenizer = tokenizer
        
    def __getitem__(self, index):
        pheno_name = self.phenotypes[index]
        if self.use_desc and self.pheno_desc_dict:
            try:
                desc = self.pheno_desc_dict[pheno_name]
            except KeyError:
                desc = ''
        return 'Name: {}. Definition: {}'.format(pheno_name, desc).strip()  # modified to match biobridge text data format

    def __len__(self):
        return len(self.phenotypes)
    

def inference(model, data_loader, dis_emb_proj, device):
    var_emb_list = []
    var_dis_label_all = []
    pos_dis_scores, neg_dis_scores = [], []
    all_similarities = []
    all_var_names, pos_disease_names, all_pos_pheno_idx = [], [], []
    vocab_size = dis_emb_proj.size(0)
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            variant_data = load_input_to_device(batch_data, device=device, exclude_keys=['var_names'])           
            var_seq_emb_proj = model(variant_data)

            pos_dis_idx = variant_data['pos_pheno_idx']
            neg_dis_idx = sample_random_negative(vocab_size, pos_dis_idx, n_neg=1).squeeze(1)
            
            dis_pos_score = torch.cosine_similarity(var_seq_emb_proj, dis_emb_proj[pos_dis_idx])
            dis_neg_score = torch.cosine_similarity(var_seq_emb_proj, dis_emb_proj[neg_dis_idx])
            batch_cos_sim = torch.cosine_similarity(var_seq_emb_proj.unsqueeze(1), dis_emb_proj.unsqueeze(0), dim=-1)

            all_var_names.extend(batch_data['var_names'])
            pos_disease_names.extend(batch_data['pos_pheno_name'])
            var_emb_list.append(var_seq_emb_proj.detach().cpu().numpy())
            pos_dis_scores.append(dis_pos_score.detach().cpu().numpy())
            neg_dis_scores.append(dis_neg_score.detach().cpu().numpy())
            var_dis_label_all.extend(pos_dis_idx.detach().cpu().tolist())
            all_similarities.append(batch_cos_sim.detach().cpu().numpy())

    result_dict = {'var_names': all_var_names,
                   'pos_pheno_name': pos_disease_names,
                   'pheno_idx': var_dis_label_all,
                   'var_proj_emb': np.concatenate(var_emb_list, 0),
                   'pos_score': np.concatenate(pos_dis_scores, 0),
                   'neg_score': np.concatenate(neg_dis_scores, 0),
                   'similarities': np.concatenate(all_similarities, 0)}
    
    return result_dict


def emb_disease_vocab(phenotype_vocab, pheno_desc_dict, text_tokenizer, config, device):
    text_encoder = BertForMaskedLM.from_pretrained(config['model']['text_lm_path'])
    for name, parameters in text_encoder.named_parameters():
        parameters.requires_grad = False
    text_encoder = text_encoder.to(device)

    pheno_dataset = BioBridgeTextDataset(phenotype_vocab, pheno_desc_dict, use_desc=config['dataset']['use_pheno_desc'])
    pheno_collator = TextDataCollator(text_tokenizer, padding=True)
    phenotype_loader = DataLoader(pheno_dataset, batch_size=config['pheno_batch_size'], collate_fn=pheno_collator, shuffle=False)
    
    dis_emb_raw = extract_text_emb(text_encoder, device, phenotype_loader)

    return dis_emb_raw


def save_results(result_dict, save_path, split, save_emb=True):
    if isinstance(save_path, str):
        save_path = Path(save_path)

    df_pheno_results = pd.DataFrame({'prot_var_id': result_dict['var_names'], 
                                     'phenotype': result_dict['pos_pheno_name'], 
                                     'pheno_idx': result_dict['pheno_idx'],
                                     'pos_score': result_dict['pos_score'],
                                     'neg_score': result_dict['neg_score']})
    
    df_pheno_results.to_csv(save_path / f'{split}_pheno_score.tsv', sep='\t', index=False)
    if save_emb:
        np.save(save_path / f'{split}_var_proj_emb.npy', result_dict['var_proj_emb'])
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="configuration file", default='./configs/biobridge_pred_full.yaml')
    parser.add_argument("-c_fmt", "--config_fmt", help="configuration file format", default='yaml')
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)
    parser.add_argument('--data_dir', help='Data directory')
    parser.add_argument('--exp_dir', help='Directory for all training related files, e.g. checkpoints, log')
    parser.add_argument('--checkpoint_dir', help='Directory to pre-trained model checkpoint')
    parser.add_argument('--experiment', help='Experiment name')
    parser.add_argument('--log_level', default='info', help='Log level')
    # args, unparsed = parser.parse_known_args()
    args = parser.parse_args()

    return args


def main():
    # load BioBridge model
    args = parse_args()
    config = load_config(args.config, format=args.config_fmt)

    config, device = env_setup(args, config, use_timestamp=False)
    data_configs = config['dataset']
    model_args = config['model']
    exp_dir = config['exp_dir']
    output_path = Path(exp_dir) / 'result'

    if not output_path.exists():
        output_path.mkdir(parents=True)

    checkpoint_dir = config['checkpoint_dir']
    with open(os.path.join(checkpoint_dir, "model_config.json"), "r") as f:
        model_config = json.load(f)
    biobridge_model = BindingModel(**model_config)
    biobridge_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "pytorch_model.bin")))
    # biobridge_model = BridgeInference(model)

    # Load disease vocabulary
    text_tokenizer = BertTokenizer.from_pretrained(config['model']['text_lm_path'])
    
    with open(data_configs['phenotype_vocab_file'], 'r') as f:
        phenotype_vocab = f.read().splitlines()
    phenotype_vocab.insert(0, text_tokenizer.unk_token)  # add unknown token

    if data_configs['use_pheno_desc']:
        with open(data_configs['phenotype_desc_file']) as f:
            pheno_desc_dict = json.load(f)
    else:
        pheno_desc_dict = None
        
    try:
        dis_emb_raw = np.load(config['disease_emb_cache'])
        dis_emb_raw = torch.tensor(dis_emb_raw, device=device)
    except FileNotFoundError:
        # compute disease vocabulary embedding
        dis_emb_raw = emb_disease_vocab(phenotype_vocab, pheno_desc_dict, text_tokenizer, config, device)
        np.save(output_path / Path(config['disease_emb_cache']).name, dis_emb_raw)

    logging.info('Disease vocabulary embedding generation complete!')
    # if config['encode_disease_only']:
    #     return
    
    # Load protein seq encoder
    protein_tokenizer = AutoTokenizer.from_pretrained(model_args['protein_lm_path'],
        do_lower_case=False
    )
    # seq_encoder = AutoModelForMaskedLM.from_pretrained(model_args['protein_lm_path'])
    seq_encoder = EsmModel.from_pretrained(model_args['protein_lm_path'])
    for name, parameters in seq_encoder.named_parameters():
        parameters.requires_grad = False
    seq_encoder = seq_encoder.to(device)
    biobridge_model = biobridge_model.to(device)
    prot_dis_model = ProteinDiseaseBridgeModel(seq_encoder, biobridge_model)
    # with torch.no_grad():
    dis_emb_raw = torch.tensor(dis_emb_raw, device=device)
    dis_emb_proj = prot_dis_model.project_emb(dis_emb_raw)
    np.save(output_path / 'dis_emb_proj.npy', dis_emb_proj.detach().cpu().numpy())
    # if config['encode_disease_only']:
    #     return
    
    prot2seq = dict()
    for fname in data_configs['seq_fasta']:
        try:
            seq_dict, _ = parse_fasta_info(fname)
            prot2seq.update(seq_dict)
        except FileNotFoundError:
            pass
    data_configs['seq_dict'] = prot2seq
    if isinstance(data_configs['input_file']['test'], str):
        test_flist = [data_configs['input_file']['test']]
    else:
        test_flist = data_configs['input_file']['test']
    
    for test_file in test_flist:
        logging.info(f'Inference on {test_file}...')
        fname = os.path.basename(test_file).split('.')[0]
        eval_dataset = ProteinVariantSeqDataset(**data_configs,
                                                variant_file=test_file, 
                                                protein_tokenizer=protein_tokenizer,
                                                phenotype_vocab=phenotype_vocab)
        logging.info('{} variants loaded'.format(len(eval_dataset)))

        prot2seq = eval_dataset.get_all_protein_seq()
        data_collator = ProteinVariantSeqCollator(eval_dataset, protein_tokenizer, prot2seq)
        data_loader = DataLoader(eval_dataset, batch_size=config['batch_size'], collate_fn=data_collator, shuffle=False)

        result_dict = inference(prot_dis_model, data_loader, dis_emb_proj, device)
        save_results(result_dict, output_path, split=fname, save_emb=True)


if __name__ == '__main__':
    main()