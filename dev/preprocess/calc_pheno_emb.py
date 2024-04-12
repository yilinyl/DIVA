import os, sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.abspath('..')))
sys.path.append(os.path.abspath('..'))

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

import argparse
import logging
import numpy as np
import pandas as pd

from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertModel, BertForMaskedLM, EsmForMaskedLM, AutoModelForMaskedLM
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

from dev.data.dis_var_dataset import PhenotypeDataset, TextDataCollator

from sklearn.metrics.pairwise import cosine_similarity

LM_PATH = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'

def embed_phenotypes(text_encoder, device, pheno_loader):
    """
    Get initial phenotype embeddings using frozen pretrained LM
    """
    text_encoder.eval()
    all_pheno_embs = []

    with torch.no_grad():
        for idx, batch_pheno in enumerate(pheno_loader):
            # pheno_input_dict = load_input_to_device(batch_pheno, device)
            pheno_input_dict = batch_pheno.to(device)
            # pheno_embs = text_encoder.get_pheno_emb(pheno_input_dict, proj=True)
            pheno_embs = text_encoder(
                pheno_input_dict['input_ids'],
                attention_mask=pheno_input_dict['attention_mask'],
                token_type_ids=pheno_input_dict['token_type_ids'],
                # token_type_ids=torch.zeros(pheno_input_dict['input_ids'].size(), dtype=torch.long, device=self.device),
                output_attentions=False,
                output_hidden_states=True,
                return_dict=None).hidden_states[-1]
            batch_size = pheno_input_dict['input_ids'].shape[0]
            # pheno_embs = torch.stack([pheno_embs[i, pheno_input_dict['attention_mask'][i, :].bool()].mean(dim=0) for i in range(batch_size)], dim=0)
            pheno_embs = torch.stack([pheno_embs[i, pheno_input_dict['attention_mask'][i, :].bool()][0] for i in range(batch_size)], dim=0)  # use embedding corresponding to the [CLS] token
            # print(pheno_embs.shape)
            all_pheno_embs.append(pheno_embs.detach().cpu().numpy())
            
    all_pheno_embs = np.concatenate(all_pheno_embs, 0)
    # all_pheno_embs_ts = torch.tensor(all_pheno_embs, device=device)
    # cos_sim = F.cosine_similarity(all_pheno_embs_ts.unsqueeze(1), all_pheno_embs_ts.unsqueeze(0), dim=-1)
    cos_sim = cosine_similarity(all_pheno_embs)
    print(all_pheno_embs.shape)
    print(cos_sim.shape)

    return all_pheno_embs, cos_sim


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phenotype_vocab_file", help="phenotype vocabulary text file")
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for loading phenotype data')
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)
    # parser.add_argument("--lm_path", help="path to pretrained LM")
    parser.add_argument('--device', help='device type')

    parser.add_argument('--output_dir', default='./scratch', help='Directory for output')
    parser.add_argument('--log_level', default='info', help='Log level')
    
    # args, unparsed = parser.parse_known_args()
    args = parser.parse_args()

    return args


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


if __name__ == '__main__':
    args = parse_args()
    device = gpu_setup(args.device)
    out_root = Path(args.output_dir)

    # Load data
    with open(args.phenotype_vocab_file, 'r') as f:
        phenotype_vocab = f.read().splitlines()
    
    text_tokenizer = BertTokenizer.from_pretrained(LM_PATH)

    pheno_dataset = PhenotypeDataset(phenotype_vocab)
    pheno_collator = TextDataCollator(text_tokenizer, padding=True)
    phenotype_loader = DataLoader(pheno_dataset, batch_size=args.batch_size, collate_fn=pheno_collator, shuffle=False)

    text_encoder = BertForMaskedLM.from_pretrained(LM_PATH)
    for name, parameters in text_encoder.named_parameters():
        parameters.requires_grad = False
    text_encoder = text_encoder.to(device)
    all_pheno_embs, all_cos_sim = embed_phenotypes(text_encoder, device, phenotype_loader)

    np.save(out_root / f'pheno_desc_rev1_emb.npy', all_pheno_embs)
    np.save(out_root / f'pheno_desc_rev1_cos_sim.npy', all_cos_sim)
    print('Done!')
