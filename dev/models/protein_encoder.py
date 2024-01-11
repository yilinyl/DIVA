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
                 use_desc: bool = False):
        super().__init__()

        self.seq_encoder = seq_encoder
        self.text_encoder = text_encoder
        self.use_desc = use_desc
        self.device = self.seq_encoder.device
    
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


class LogitDiffPathogenicityHead(nn.Module):
  """Variant pathogenicity classification head. (modified from AlphaMissense)"""

  def __init__(self,
               n_residue_types: int,
               name: str = 'logit_diff_head',
               pad_label_idx: int = -100
               ):
    super().__init__(name=name)
   
    self.n_residue_types = n_residue_types
    self.variant_row = 1
    self.patho_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_label_idx)

  def forward(self, logits, ref_aa, alt_aa, variant_mask):
    ref_score = torch.einsum('ij, ij->i', logits, F.one_hot(ref_aa, num_classes=self.n_residue_types))
    variant_score = torch.einsum('ij, ij->i', logits, F.one_hot(alt_aa, num_classes=self.n_residue_types))
    logit_diff = ref_score - variant_score
    var_pathogenicity = torch.sum(logit_diff * variant_mask)

    return logit_diff, var_pathogenicity

  def loss(self, pred, labels, variant_mask):
    # loss = clipped_sigmoid_cross_entropy(logits=value['variant_row_logit_diff'],
    #                                      labels=batch['pathogenicity'],
    #                                      clip_negative_at_logit=0.0,
    #                                      clip_positive_at_logit=-1.0)
    # loss = (torch.sum(loss * batch['variant_mask'], axis=(-2, -1)) /
    #         (1e-8 + torch.sum(batch['variant_mask'], axis=(-2, -1))))
    loss = self.patho_loss_fn(pred, labels)
    # loss = (torch.sum(loss * variant_mask, axis=(-2, -1)) / (1e-8 + torch.sum(variant_mask, axis=(-2, -1))))
    
    return loss

class DiseaseVariantEncoder(nn.Module):

    def __init__(self,
                 seq_encoder: Union[nn.Module, PreTrainedModel],
                 text_encoder: Union[nn.Module, PreTrainedModel],
                 n_residue_types,
                 hidden_size,
                 use_desc=True,
                 pad_label_idx=-100,
                 num_heads=4,
                 **kwargs):
        super(DiseaseVariantEncoder, self).__init__()

        self.protein_encoder = ProteinEncoder(seq_encoder=seq_encoder,
                                              text_encoder=text_encoder,
                                              use_desc=use_desc)  # requires initialization
        self.seq_encoder = seq_encoder
        self.text_encoder = text_encoder

        self.n_residue_types = n_residue_types
        self.use_desc = use_desc
        self.device = self.protein_encoder.device

        self.seq_emb_dim = self.seq_encoder.config.hidden_size
        self.text_emb_dim = self.text_encoder.config.hidden_size
        self.hidden_size = hidden_size

        self.seq_pheno_comb = nn.Linear(self.seq_emb_dim + self.text_emb_dim, self.hidden_size)

        # multihead attention along sequence
        self.W_K = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_Q = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_V = nn.Linear(self.hidden_size, self.hidden_size)

        self.mha = MultiheadAttention(self.hidden_size, num_heads=num_heads, batch_first=True)

        self.patho_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_label_idx)
    

    def forward(self, seq_input_feat, batch_data, desc_input_feat=None):
        
        # with torch.no_grad():
        seq_embs, mlm_logits, desc_embs = self.protein_encoder(seq_input_feat, desc_input_feat)  # mlm_logits: (batch_size, max_seq_length, vocab_size=33)
        batch_size = batch_data['label'].size(0)
        batch_label = batch_data['label'].to(self.device)
        pheno_feat_dict = batch_data['phenotype']
        # TODO: mask/replace token at target position
        pheno_pos_feat = load_input_to_device(pheno_feat_dict['positive'], self.device, exclude_keys=['pheno_desc'])
        pheno_neg_feat = load_input_to_device(pheno_feat_dict['negative'], self.device, exclude_keys=['pheno_desc'])
        # pheno_pos_input_ids = pheno_feat_dict['pos_input_ids'].to(self.device)
        # pheno_neg_input_ids = pheno_feat_dict['neg_input_ids'].to(self.device)
        # seq_embs_agg = seq_embs.mean(dim=1)
        desc_embs_agg = desc_embs.mean(dim=1)  # batch_size, max_desc_length, desc_emb_dim --> batch_size, desc_emb_dim
        
        variant_idx = torch.where(batch_data['variant_mask'])
        
        pos_pheno_embs = self.text_encoder(
            pheno_pos_feat['input_ids'],
            attention_mask=pheno_pos_feat['attention_mask'],
            token_type_ids=torch.zeros_like(pheno_pos_feat['input_ids']),
            output_attentions=False,
            output_hidden_states=True,
            return_dict=None
        ).hidden_states[-1]

        neg_pheno_embs = self.text_encoder(
            pheno_neg_feat['input_ids'],
            attention_mask=pheno_neg_feat['attention_mask'],
            output_attentions=False,
            output_hidden_states=True,
            return_dict=None
        ).hidden_states[-1]

        variant_pheno_emb = self.text_encoder(
            pheno_pos_feat['mlm_input_ids'],
            attention_mask=pheno_pos_feat['mlm_attention_mask'],
            output_attentions=False,
            output_hidden_states=True,
            return_dict=None
        )

        # var_pheno_emb_lst = []
        # for b in range(batch_size):
        #     var_pheno_emb_cur = self.text_encoder(
        #         pheno_pos_feat['mlm_input_ids'][b],  # batch_size, max_seq_length, max_pheno_length
        #         attention_mask=pheno_pos_feat['mlm_attention_mask'][b],
        #         # token_type_ids=desc_input_data['token_type_ids'],
        #         output_attentions=False,
        #         output_hidden_states=True,
        #         return_dict=None,
        #     ).hidden_states[-1].mean(-2)
        #     var_pheno_emb_lst.append(var_pheno_emb_cur)
        # # batch_size, max_seq_length, max_pheno_length, text_emb_dim
        # variant_pheno_emb = torch.stack(var_pheno_emb_lst, 0)
        # variant_pheno_emb = torch.cat([desc_embs_agg, variant_pheno_emb], dim=1)  # concatenation of global and variant-specific functional embedding
        # pheno_pos_emb = pheno_pos_outputs.hidden_states[-1].mean(-2)
        seq_pheno_emb_raw = torch.cat([seq_embs, variant_pheno_emb], dim=-1)  # batch_size, max_seq_length, seq_emb_dim+text_emb_dim
        seq_pheno_emb = self.seq_pheno_comb(seq_pheno_emb_raw)  # batch_size, max_seq_length, hidden_size

        k = self.W_K(seq_pheno_emb)
        q = self.W_Q(seq_pheno_emb)
        v = self.W_V(seq_pheno_emb)

        attn_outputs, attn_weights = self.mha(q, k, v)

        return seq_embs, mlm_logits, desc_embs

    # TODO: ref_aa, alt_aa, label format, how to convert into mask / 1-hot matrix 
    def log_diff_patho_score(self, logits, ref_aa, alt_aa, variant_mask):
        # modified from AlphaMissense
        ref_score = torch.einsum('ij, ij->i', logits, F.one_hot(ref_aa, num_classes=self.n_residue_types))
        alt_score = torch.einsum('ij, ij->i', logits, F.one_hot(alt_aa, num_classes=self.n_residue_types))
        logit_diff = ref_score - alt_score
        var_pathogenicity = torch.sum(logit_diff * variant_mask)

        return logit_diff, var_pathogenicity

