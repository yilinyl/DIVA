import os
import json
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import math
import dataclasses
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from transformers import PreTrainedTokenizerBase, PreTrainedModel
import pandas as pd

from .loss import clipped_sigmoid_cross_entropy, sigmoid_cosine_distance_p
from .protein_encoder import ProteinEncoder

_dist_fn_map = {'euclidean': nn.PairwiseDistance(),
                'cosine_sigmoid': sigmoid_cosine_distance_p}

_calibration_fn_map = {'sigmoid': nn.Sigmoid(),
                       'none': None}


class DiseaseVariantBaseModel(nn.Module):
    def __init__(self,
                 seq_encoder: Union[nn.Module, PreTrainedModel],
                 text_encoder: Union[nn.Module, PreTrainedModel],
                 n_residue_types,
                 hidden_size,
                 desc_proj_dim=128,
                 use_desc=True,
                 pad_label_idx=-100,
                 dist_fn_name='cosine_sigmoid',
                 use_struct_vocab=False,
                 foldseek_vocab="pynwrqhgdlvtmfsaeikc#",
                 adjust_logits=False,
                 use_alphamissense=False,
                 device='cpu',
                 **kwargs):
        super(DiseaseVariantBaseModel, self).__init__()

        self.protein_encoder = ProteinEncoder(seq_encoder=seq_encoder,
                                              text_encoder=text_encoder,
                                              use_desc=False,
                                              device=device)  # requires initialization
        self.seq_encoder = seq_encoder
        self.text_encoder = text_encoder

        self.n_residue_types = n_residue_types
        self.use_desc = use_desc
        self.use_struct_vocab = use_struct_vocab
        self.foldseek_vocab_size = len(foldseek_vocab)
        self.device = device

        self.seq_emb_dim = self.seq_encoder.config.hidden_size
        self.text_emb_dim = self.text_encoder.config.hidden_size
        self.hidden_size = hidden_size

        self.seq_pheno_comb = nn.Linear(self.seq_emb_dim + self.text_emb_dim * 2, self.hidden_size) # concatenated embedding of alt_seq, prot_desc, context_pheno

        self.proj_head = nn.Linear(self.text_emb_dim, self.hidden_size)
        self.use_alphamissense = use_alphamissense
        self.prompt_hidden_size = self.text_emb_dim // 2
        self.adjust_logits = adjust_logits
        if self.use_alphamissense:
            self.adjust_logits = True
            
        if self.adjust_logits:
            self.func_prompt_l1 = nn.Linear(self.text_emb_dim, self.prompt_hidden_size)
            self.func_prompt_l2 = nn.Linear(self.prompt_hidden_size, 1)
            self.func_prompt_mlp = nn.Sequential(self.func_prompt_l1,  # learnable weight from protein function embedding
                                                nn.ReLU(),
                                                self.func_prompt_l2)
            if not self.use_alphamissense:  # scaler initialized ~1
                nn.init.normal_(self.func_prompt_l2.weight, std=(2 / self.prompt_hidden_size)**0.5)
                nn.init.ones_(self.func_prompt_l2.bias)
        self.weight_act = nn.Sigmoid()

        self.patho_output_layer = nn.Linear(self.seq_emb_dim + self.text_emb_dim, 2)  # use concatenated embedding of alt_seq and prot_desc
        
        self.alpha = nn.Parameter(torch.tensor(-1e-3))
        self.dist_fn = _dist_fn_map[dist_fn_name]
        self.patho_loss_fn = nn.BCEWithLogitsLoss()
    
    def binary_step(self, seq_input_feat, variant_data):
        # seq_embs, mlm_logits, desc_embs = self.protein_encoder(seq_input_feat, desc_input_feat)  # mlm_logits: (batch_size, max_seq_length, vocab_size=33)
        mlm_logits = self.protein_encoder.get_mlm_logits(seq_input_feat)
        probs = mlm_logits.softmax(dim=-1)
        logit_diff = 0
        # for single in mut_info.split(":"):
        var_idx = torch.tensor(variant_data['var_idx'], device=self.device) - variant_data['offset_idx']
        batch_idx = torch.arange(len(var_idx), device=self.device)
        if self.use_struct_vocab:  # from SaProt script
            ori_st = variant_data['ref_aa'].unsqueeze(1)  # starting index
            mut_st = variant_data['alt_aa'].unsqueeze(1)
            range_indices = torch.arange(self.foldseek_vocab_size, device=self.device).unsqueeze(0)
            ori_prob = probs[batch_idx, var_idx+1].gather(1, ori_st + range_indices).sum(1)
            mut_prob = probs[batch_idx, var_idx+1].gather(1, mut_st + range_indices).sum(1)
            # ori_prob = probs[batch_idx, var_idx+1, ori_st: ori_st + self.foldseek_vocab_size].sum()
            # mut_prob = probs[batch_idx, var_idx+1, mut_st: mut_st + self.foldseek_vocab_size].sum()
        else:
            ori_prob = probs[batch_idx, var_idx+1, variant_data['ref_aa']]
            mut_prob = probs[batch_idx, var_idx+1, variant_data['alt_aa']]
            
        # logit_diff = torch.log(mut_prob / ori_prob)  # smaller for pathogenic
        logit_diff = torch.log(ori_prob / mut_prob)  # larger for pathogenic

        return mlm_logits, logit_diff.unsqueeze(1)
    
    def forward(self, *args, **kwargs):
        pass