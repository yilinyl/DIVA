import os
import json
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import dataclasses
from dataclasses import dataclass
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from transformers import PreTrainedTokenizerBase, PreTrainedModel
import pandas as pd

from dev.models.protein_encoder import ProteinEncoder
from dev.models.base_model import DiseaseVariantBaseModel

class LMBaseModel(nn.Module):
    def __init__(self, 
                 seq_encoder: Union[nn.Module, PreTrainedModel], 
                 text_encoder: Union[nn.Module, PreTrainedModel],
                 use_desc: bool = False):
        super(LMBaseModel, self).__init__()

        self.seq_encoder = seq_encoder
        self.text_encoder = text_encoder
        self.use_desc = use_desc
        self.device = seq_encoder.device

        self.protein_encoder = ProteinEncoder(seq_encoder=seq_encoder,
                                              text_encoder=text_encoder,
                                              use_desc=use_desc,
                                              device=self.device)
    
    def forward(self, seq_input_feat, variant_data, desc_input_feat):
        seq_embs, mlm_logits, prot_desc_emb = self.protein_encoder(seq_input_feat, desc_input_feat)

        # Pathogenicity prediction
        desc_emb_agg = torch.stack([prot_desc_emb[i, desc_input_feat['attention_mask'][i, :].bool(), :][0] for i in range(prot_desc_emb.size(0))], dim=0)
        
        alt_seq_input_feat = {  # All ALT seq
            'input_ids': variant_data['var_seq_input_ids'],
            'attention_mask': seq_input_feat['attention_mask'][variant_data['prot_idx']]
        }
        alt_seq_embs = self.protein_encoder.embed_protein_seq(alt_seq_input_feat)
        alt_seq_embs = torch.stack([alt_seq_embs[i, alt_seq_input_feat['attention_mask'][i, :].bool(), :][0] for i in range(alt_seq_embs.size(0))], dim=0)
        alt_seq_func_embs = torch.cat([alt_seq_embs, desc_emb_agg[variant_data['prot_idx']]], dim=-1)  # concatenate alt-seq embedding & prot-desc embedding

        max_text_length = self.text_encoder.config.max_position_embeddings
        pheno_input_ids = variant_data['context_pheno_input_ids']
        pheno_attn_mask = variant_data['context_pheno_attention_mask']
        pheno_indices = variant_data['context_pheno_indices']
        n_uniq_phenos = pheno_input_ids.shape[0]
        # assert pheno_input_ids.shape[0] == n_pheno_vars

        if pheno_input_ids.shape[-1] > max_text_length:
            pheno_input_ids = pheno_input_ids[:, :max_text_length]
            pheno_attn_mask = pheno_attn_mask[:, :max_text_length]
        
        seq_context_pheno_emb_raw = self.text_encoder(
            pheno_input_ids,
            attention_mask=pheno_attn_mask,
            token_type_ids=torch.zeros(pheno_input_ids.size(), dtype=torch.long, device=self.device),
            output_attentions=False,
            output_hidden_states=True,
            return_dict=None
        ).hidden_states[-1]

        seq_context_pheno_emb_raw = torch.stack([seq_context_pheno_emb_raw[i, pheno_attn_mask[i, :].bool(), :][0] for i in range(n_uniq_phenos)], dim=0)
        seq_pheno_emb_raw = torch.cat([alt_seq_func_embs[variant_data['infer_pheno_vec'].bool()], seq_context_pheno_emb_raw], dim=-1)   # n_var, (seq_emb_dim + desc_emb_dim + pheno_emb_dim)

        return seq_pheno_emb_raw, mlm_logits
    

    def embed_protein_seq(self, seq_input_feat):
        seq_embs = self.seq_encoder(
            seq_input_feat['input_ids'],
            attention_mask=seq_input_feat['attention_mask'],
            # token_type_ids=token_type_ids,
            output_attentions=False,
            output_hidden_states=True,
        ).hidden_states[-1]

        return seq_embs


class VariantDiseaseClassifier(DiseaseVariantBaseModel):

    def __init__(self,
                 seq_encoder: Union[nn.Module, PreTrainedModel],
                 text_encoder: Union[nn.Module, PreTrainedModel],
                 n_residue_types,
                 hidden_size,
                 n_classes,
                 desc_proj_dim=128,
                 use_desc=True,
                 adjust_logits=False,
                 use_alphamissense=False,
                 device='cpu',
                 **kwargs):
        super(VariantDiseaseClassifier, self).__init__(seq_encoder, 
                                                       text_encoder, 
                                                       n_residue_types, 
                                                       hidden_size, 
                                                       desc_proj_dim, 
                                                       use_desc, 
                                                       adjust_logits=adjust_logits,
                                                       use_alphamissense=use_alphamissense,
                                                       device=device)

        self.disease_loss_fn = nn.CrossEntropyLoss()
        self.n_classes = n_classes
        self.output_layer = nn.Sequential(nn.ReLU(),
                                          nn.Linear(self.hidden_size, self.n_classes))

    
    def forward(self, variant_data, desc_input_feat):
        # protein desc embedding for all variants in batch
        # var_indices = (variant_data['prot_idx'], variant_data['var_idx'])
        prot_desc_emb = self.text_encoder(
            desc_input_feat['input_ids'],
            attention_mask=desc_input_feat['attention_mask'],
            token_type_ids=torch.zeros(desc_input_feat['input_ids'].size(), dtype=torch.long, device=self.device),
            output_attentions=False,
            output_hidden_states=True,
            return_dict=None
        ).hidden_states[-1]

        # Pathogenicity prediction
        desc_emb_agg = torch.stack([prot_desc_emb[i, desc_input_feat['attention_mask'][i, :].bool(), :][0] for i in range(prot_desc_emb.size(0))], dim=0)
        # use altnerated sequence for phenotype inference
        masked_seq_input_feat = variant_data['masked_seq_input_feat']        
        alt_seq_input_feat = variant_data['var_seq_input_feat']
        alt_seq_embs = self.protein_encoder.embed_protein_seq(alt_seq_input_feat)
        alt_seq_embs = torch.stack([alt_seq_embs[i, alt_seq_input_feat['attention_mask'][i, :].bool(), :][0] for i in range(alt_seq_embs.size(0))], dim=0)
        alt_seq_func_embs = torch.cat([alt_seq_embs, desc_emb_agg[variant_data['prot_idx']]], dim=-1)  # concatenate alt-seq embedding & prot-desc embedding
        # var_patho_logits = self.patho_output_layer(alt_seq_func_embs)
        _, var_logit_diff = self.binary_step(masked_seq_input_feat, variant_data)

        if not self.use_alphamissense:
            if self.adjust_logits:
                prompt_weights = self.func_prompt_mlp(desc_emb_agg[variant_data['prot_idx']])
            else:
                prompt_weights = 1
            weighted_bin_logits = prompt_weights * var_logit_diff
        else:
            prompt_weights = self.func_prompt_mlp(desc_emb_agg[variant_data['prot_idx']])
            prompt_weights = self.weight_act(prompt_weights)  # with activation (sigmoid)
            afmis_mask = variant_data['afmis_mask']  # True if alphamissense NOT available
            afmis_logits = torch.logit(variant_data['afmis_score'], eps=1e-6).unsqueeze(1)
            weighted_bin_logits = prompt_weights * var_logit_diff + (1 - prompt_weights) * afmis_logits
            weighted_bin_logits[afmis_mask] = var_logit_diff[afmis_mask]

        n_pheno_vars = variant_data['infer_pheno_vec'].sum().item()
        if n_pheno_vars == 0:  # Pathogenicity 
            return weighted_bin_logits, prompt_weights, None, None
    
        max_text_length = self.text_encoder.config.max_position_embeddings
        # pheno_input_ids = variant_data['context_pheno_input_ids'].view(n_pheno_vars, -1)
        # pheno_attn_mask = variant_data['context_pheno_attention_mask'].view(n_pheno_vars, -1)
        pheno_input_ids = variant_data['context_pheno_input_ids']
        pheno_attn_mask = variant_data['context_pheno_attention_mask']
        pheno_indices = variant_data['context_pheno_indices']
        n_uniq_phenos = pheno_input_ids.shape[0]
        # assert pheno_input_ids.shape[0] == n_pheno_vars

        if pheno_input_ids.shape[-1] > max_text_length:
            pheno_input_ids = pheno_input_ids[:, :max_text_length]
            pheno_attn_mask = pheno_attn_mask[:, :max_text_length]
        
        seq_context_pheno_emb_raw = self.text_encoder(
            pheno_input_ids,
            attention_mask=pheno_attn_mask,
            token_type_ids=torch.zeros(pheno_input_ids.size(), dtype=torch.long, device=self.device),
            output_attentions=False,
            output_hidden_states=True,
            return_dict=None
        ).hidden_states[-1]
        
        seq_context_pheno_emb_raw = torch.stack([seq_context_pheno_emb_raw[i, pheno_attn_mask[i, :].bool(), :][0] for i in range(n_uniq_phenos)], dim=0)        
        seq_pheno_emb_raw = torch.cat([alt_seq_func_embs[variant_data['infer_pheno_vec'].bool()], seq_context_pheno_emb_raw], dim=-1)   # n_var, (seq_emb_dim + desc_emb_dim + pheno_emb_dim)
        seq_pheno_emb = self.seq_pheno_comb(seq_pheno_emb_raw)  # n_var, hidden_size
        disease_cls_logits = self.output_layer(seq_pheno_emb)

        return weighted_bin_logits, prompt_weights, seq_pheno_emb, disease_cls_logits