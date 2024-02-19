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
from torch.nn import MultiheadAttention
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
from transformers import PreTrainedTokenizerBase, PreTrainedModel
import pandas as pd

from dev.utils import load_input_to_device


class ProteinEncoder(nn.Module):
    def __init__(self, 
                 seq_encoder: Union[nn.Module, PreTrainedModel], 
                 text_encoder: Union[nn.Module, PreTrainedModel],
                 use_desc: bool = False,
                 device: str = 'cpu'):
        super().__init__()

        self.seq_encoder = seq_encoder
        self.text_encoder = text_encoder
        self.use_desc = use_desc
        self.device = device
    
    def forward(self, seq_input_data, desc_input_data=None):
        seq_outputs = self.seq_encoder(
            seq_input_data['input_ids'],
            attention_mask=seq_input_data['attention_mask'],
            # token_type_ids=seq_input_data['token_type_ids'],
            output_attentions=False,
            output_hidden_states=True,
        )
        # attn_mask = seq_input_data['attention_mask'].bool()
        # num_batch_size = seq_input_data['attention_mask'].size(0)
        # seq_embs = torch.stack([seq_outputs.last_hidden_state[i, attn_mask[i, :], :][1:-1].mean(dim=0) for i in range(num_batch_size)], dim=0)
        seq_embs, mlm_logits = seq_outputs.hidden_states[-1], seq_outputs.logits

        # embedding protein functional description
        if self.use_desc:
            desc_outputs = self.text_encoder(
                desc_input_data['input_ids'],
                attention_mask=desc_input_data['attention_mask'],
                # token_type_ids=desc_input_data['token_type_ids'],
                output_attentions=False,
                output_hidden_states=True,
                return_dict=None,
            )

            # attn_mask = desc_input_data['attention_mask'].bool()
            # num_batch_size = desc_input_data['attention_mask'].size(0)
            desc_embs = desc_outputs.hidden_states[-1]  # (batch_size, max_desc_length, emb_dim)
            # desc_embs = torch.stack([desc_outputs.last_hidden_state[i, attn_mask[i, :], :][1:-1].mean(dim=0) for i in range(num_batch_size)], dim=0)

            return seq_embs, mlm_logits, desc_embs
        
        return seq_embs, mlm_logits, None
    
    def embed_protein_seq(self, seq_input_feat):
        seq_embs = self.seq_encoder(
            seq_input_feat['input_ids'],
            attention_mask=seq_input_feat['attention_mask'],
            # token_type_ids=token_type_ids,
            output_attentions=False,
            output_hidden_states=True,
        ).hidden_states[-1]

        return seq_embs


def clipped_sigmoid_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    clip_negative_at_logit: float,
    clip_positive_at_logit: float,
    epsilon: float = 1e-07,
    ):
    """Computes sigmoid xent loss with clipped input logits. (from AlphaMissense)

    Args:
    logits: The predicted values.
    labels: The ground truth values.
    clip_negative_at_logit: clip the loss to 0 if prediction smaller than this
        value for the negative class.
    clip_positive_at_logit: clip the loss to this value if prediction smaller
        than this value for the positive class.
    epsilon: A small increment to add to avoid taking a log of zero.

    Returns:
    Loss value.
    """
    prob = torch.sigmoid(logits)
    prob = torch.clip(prob, epsilon, 1. - epsilon)
    loss = -labels * torch.log(prob) - (1. - labels) * torch.log(1. - prob)  # cross-entropy

    loss_at_clip = np.log(np.exp(clip_negative_at_logit) + 1)
    loss = torch.where((1 - labels) * (logits < clip_negative_at_logit), loss_at_clip, loss)
    loss_at_clip = np.log(np.exp(-clip_positive_at_logit) + 1)
    loss = torch.where(labels * (logits < clip_positive_at_logit), loss_at_clip, loss)
    return loss

def sigmoid_cosine_distance_p(x, y, p=1):
    sig = torch.nn.Sigmoid()
    cosine_sim = torch.nn.CosineSimilarity()
    return (1 - sig(cosine_sim(x, y))) ** p


_dist_fn_map = {'euclidean': nn.PairwiseDistance(),
                'cosine_sigmoid': sigmoid_cosine_distance_p}

class DiseaseVariantEncoder(nn.Module):

    def __init__(self,
                 seq_encoder: Union[nn.Module, PreTrainedModel],
                 text_encoder: Union[nn.Module, PreTrainedModel],
                 n_residue_types,
                 hidden_size,
                 use_desc=True,
                 pad_label_idx=-100,
                 num_heads=4,
                 max_vars_per_batch=32,
                 dist_fn_name='cosine_sigmoid',
                 init_margin=1,
                 device='cpu',
                 **kwargs):
        super(DiseaseVariantEncoder, self).__init__()

        self.protein_encoder = ProteinEncoder(seq_encoder=seq_encoder,
                                              text_encoder=text_encoder,
                                              use_desc=False,
                                              device=device)  # requires initialization
        self.seq_encoder = seq_encoder
        self.text_encoder = text_encoder

        self.n_residue_types = n_residue_types
        self.use_desc = use_desc
        self.device = device

        self.seq_emb_dim = self.seq_encoder.config.hidden_size
        self.text_emb_dim = self.text_encoder.config.hidden_size
        self.hidden_size = hidden_size

        self.max_vars_per_batch = max_vars_per_batch

        self.seq_pheno_comb = nn.Linear(self.seq_emb_dim + self.text_emb_dim, self.hidden_size)

        self.proj_head = nn.Linear(self.text_emb_dim, self.hidden_size)

        self.dist_fn = _dist_fn_map[dist_fn_name]
        # self.patho_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_label_idx)
        self.patho_loss_fn = nn.BCEWithLogitsLoss()
        # self.desc_loss_fn = nn.CosineEmbeddingLoss()
        # self.cos_sim_loss_fn = nn.CosineEmbeddingLoss()
        self.contrast_loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=self.dist_fn, margin=init_margin)
    

    def forward(self, seq_input_feat, variant_data, desc_input_feat=None):
        
        # seq_embs, mlm_logits, _ = self.protein_encoder(seq_input_feat, desc_input_feat)  # mlm_logits: (batch_size, max_seq_length, vocab_size=33)
        seq_embs, mlm_logits, logit_diff = self.binary_step(seq_input_feat, variant_data, desc_input_feat)
        
        if variant_data['infer_phenotype']:
            # n_pheno_vars = len(variant_data['patho_var_prot_idx'])
            seq_pheno_emb, pos_emb_proj, neg_emb_proj = self.contrastive_step(seq_input_feat, variant_data)
        
        else:  # no valid phenotype label in the batch, skip phenotype inference
            seq_pheno_emb = None
            pos_emb_proj = None
            neg_emb_proj = None

        # var_indices = (variant_data['prot_idx'], variant_data['var_idx'])
        # logit_diff = self.log_diff_patho_score(mlm_logits[var_indices], variant_data['ref_aa'], variant_data['alt_aa'])

        return seq_pheno_emb, pos_emb_proj, neg_emb_proj, mlm_logits, logit_diff
    

    def binary_step(self, seq_input_feat, variant_data, desc_input_feat=None):
        seq_embs, mlm_logits, desc_embs = self.protein_encoder(seq_input_feat, desc_input_feat)  # mlm_logits: (batch_size, max_seq_length, vocab_size=33)
        var_indices = (variant_data['prot_idx'], variant_data['var_idx'])
        logit_diff = self.log_diff_patho_score(mlm_logits[var_indices], variant_data['ref_aa'], variant_data['alt_aa'])

        return seq_embs, mlm_logits, logit_diff


    def contrastive_step(self, seq_input_feat, variant_data):
        n_pheno_vars = variant_data['infer_pheno_vec'].sum().item()
        max_text_length = self.text_encoder.config.max_position_embeddings
        pheno_input_ids = variant_data['context_pheno_input_ids'].view(n_pheno_vars, -1)
        pheno_attn_mask = variant_data['context_pheno_attention_mask'].view(n_pheno_vars, -1)
        compute_neg = False
        if pheno_input_ids.shape[-1] > max_text_length:
            pheno_input_ids = pheno_input_ids[:, :max_text_length]
            pheno_attn_mask = pheno_attn_mask[:, :max_text_length]
        
        variant_pheno_emb = self.text_encoder(
            pheno_input_ids,
            attention_mask=pheno_attn_mask,
            token_type_ids=torch.zeros(pheno_input_ids.size(), dtype=torch.long, device=self.device),
            output_attentions=False,
            output_hidden_states=True,
            return_dict=None
        ).hidden_states[-1]
            
        pos_pheno_embs = self.text_encoder(
            variant_data['pos_pheno_input_ids'],
            attention_mask=variant_data['pos_pheno_attention_mask'],
            token_type_ids=torch.zeros(variant_data['pos_pheno_input_ids'].size(), dtype=torch.long, device=self.device),
            output_attentions=False,
            output_hidden_states=True,
            return_dict=None
        ).hidden_states[-1]  # n_var, max_pos_pheno_length, pheno_emb_dim

        if 'neg_pheno_input_ids' in variant_data:  # train contrastive
            compute_neg = True
            neg_pheno_embs = self.text_encoder(
                variant_data['neg_pheno_input_ids'],
                attention_mask=variant_data['neg_pheno_attention_mask'],
                token_type_ids=torch.zeros(variant_data['neg_pheno_input_ids'].size(), dtype=torch.long, device=self.device),
                output_attentions=False,
                output_hidden_states=True,
                return_dict=None
            ).hidden_states[-1]  # n_var, max_neg_pheno_length, pheno_emb_dim
        
        # use altnerated sequence for phenotype inference
        alt_seq_input_feat = {
            'input_ids': variant_data['var_seq_input_ids'][variant_data['infer_pheno_vec'].bool()],
            'attention_mask': seq_input_feat['attention_mask'][variant_data['patho_var_prot_idx']]
        }
        alt_seq_embs = self.protein_encoder.embed_protein_seq(alt_seq_input_feat)
        seq_var_embs = torch.stack([alt_seq_embs[i, alt_seq_input_feat['attention_mask'][i, :].bool(), :][1:-1].mean(dim=0) for i in range(n_pheno_vars)], dim=0)
        variant_pheno_emb = torch.stack([variant_pheno_emb[i, pheno_attn_mask[i, :].bool(), :].mean(dim=0) for i in range(n_pheno_vars)], dim=0)
        # seq_var_embs = seq_embs[(variant_data['prot_idx'], variant_data['var_idx'])]  # extract embedding for target position
        seq_pheno_emb_raw = torch.cat([seq_var_embs, variant_pheno_emb], dim=-1)  # n_var, (seq_emb_dim + pheno_emb_dim)
        seq_pheno_emb = self.seq_pheno_comb(seq_pheno_emb_raw)  # n_var, hidden_size

        pos_pheno_embs = torch.stack([pos_pheno_embs[i, variant_data['pos_pheno_attention_mask'][i, :].bool()].mean(dim=0) for i in range(n_pheno_vars)], dim=0)
        pos_emb_proj = self.proj_head(pos_pheno_embs)
        if compute_neg:
            neg_pheno_embs = torch.stack([neg_pheno_embs[i, variant_data['neg_pheno_attention_mask'][i, :].bool()].mean(dim=0) for i in range(n_pheno_vars)], dim=0)
            neg_emb_proj = self.proj_head(neg_pheno_embs)
        else:
            neg_emb_proj = None

        return seq_pheno_emb, pos_emb_proj, neg_emb_proj

    
    def get_pheno_emb(self, pheno_input_dict, proj=True):
        pheno_embs = self.text_encoder(
                pheno_input_dict['input_ids'],
                attention_mask=pheno_input_dict['attention_mask'],
                token_type_ids=pheno_input_dict['token_type_ids'],
                # token_type_ids=torch.zeros(pheno_input_dict['input_ids'].size(), dtype=torch.long, device=self.device),
                output_attentions=False,
                output_hidden_states=True,
                return_dict=None
            ).hidden_states[-1]  # n_var, max_pos_pheno_length, pheno_emb_dim
        batch_size = pheno_input_dict['input_ids'].shape[0]
        if proj:
            # pheno_embs = self.proj_head(pheno_embs)
            pheno_embs = torch.stack([pheno_embs[i, pheno_input_dict['attention_mask'][i, :].bool()].mean(dim=0) for i in range(batch_size)], dim=0)
            pheno_embs = self.proj_head(pheno_embs)
                    
        return pheno_embs

    def contrast_loss(self, var_emb, pos_emb, neg_emb):
        # pos_emb_dist = (var_emb - pos_emb).norm(p, dim=-1)
        # neg_emb_dist = (var_emb - neg_emb).norm(p, dim=-1)
        return self.contrast_loss_fn(var_emb, pos_emb, neg_emb)
        # targets = [torch.tensor([1], device=self.device, dtype=torch.long), 
        #           torch.tensor([-1], device=self.device, dtype=torch.long)]
        
        # return self.cos_sim_loss_fn(var_emb, pos_emb, targets[0]) + \
        #     self.cos_sim_loss_fn(var_emb, neg_emb, targets[1])

    # TODO: ref_aa, alt_aa, label format, how to convert into mask / 1-hot matrix 
    def log_diff_patho_score(self, logits, ref_aa, alt_aa):
        # modified from AlphaMissense
        ref_score = torch.einsum('ij, ij->i', logits, F.one_hot(ref_aa, num_classes=self.n_residue_types).float())
        alt_score = torch.einsum('ij, ij->i', logits, F.one_hot(alt_aa, num_classes=self.n_residue_types).float())
        logit_diff = (ref_score - alt_score).unsqueeze(1)
        # var_pathogenicity = torch.sum(logit_diff)

        return logit_diff
    
    def pathogenicity_loss(self, logit_diff, labels, clip_negative_at_logit=0.0, clip_positive_at_logit=-1.0):
        loss = clipped_sigmoid_cross_entropy(logits=logit_diff,
                                             labels=labels,
                                             clip_negative_at_logit=clip_negative_at_logit,
                                             clip_positive_at_logit=clip_positive_at_logit)
        
        loss = (torch.sum(loss, axis=(-2, -1)) / (1e-8 + torch.sum(labels.size(0), axis=(-2, -1))))
