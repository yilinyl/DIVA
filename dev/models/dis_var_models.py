import os
import json
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import math
import dataclasses
from dataclasses import dataclass
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from transformers import BertModel, BertTokenizer
from transformers import PreTrainedTokenizerBase, PreTrainedModel
import pandas as pd
from dgl.nn.pytorch import GATConv

from .loss import clipped_sigmoid_cross_entropy, InfoNCELoss, sigmoid_cosine_distance_p
from .protein_encoder import ProteinEncoder

_dist_fn_map = {'euclidean': nn.PairwiseDistance(),
                'cosine_sigmoid': sigmoid_cosine_distance_p}

class DiseaseVariantEncoder(nn.Module):

    def __init__(self,
                 seq_encoder: Union[nn.Module, PreTrainedModel],
                 text_encoder: Union[nn.Module, PreTrainedModel],
                 n_residue_types,
                 hidden_size,
                 desc_proj_dim=128,
                 use_desc=True,
                 pad_label_idx=-100,
                 num_heads=4,
                 n_gnn_layers=2,
                 max_vars_per_batch=32,
                 dist_fn_name='cosine_sigmoid',
                 init_margin=1,
                 freq_norm_factor=None,
                 use_struct_vocab=False,
                 foldseek_vocab="pynwrqhgdlvtmfsaeikc#",
                 seq_weight_scaler=1,
                 adjust_logits=False,
                 use_alphamissense=False,
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
        self.use_struct_vocab = use_struct_vocab
        self.foldseek_vocab_size = len(foldseek_vocab)
        self.device = device

        self.seq_emb_dim = self.seq_encoder.config.hidden_size
        self.text_emb_dim = self.text_encoder.config.hidden_size
        self.hidden_size = hidden_size
        self.freq_norm_factor = freq_norm_factor

        self.max_vars_per_batch = max_vars_per_batch

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
        # self.desc_proj_dim = desc_proj_dim
        # if self.use_struct_vocab:
        #     self.patho_feat_dim = self.desc_proj_dim + 2 * self.foldseek_vocab_size
        # else:
        #     self.patho_feat_dim = self.desc_proj_dim + 2
        # self.desc_proj_linear = nn.Linear(self.text_emb_dim, self.desc_proj_dim)
        # self.patho_output_layer = nn.Linear(self.patho_feat_dim, 2)  # use concatenated embedding of alt_seq and prot_desc
        self.n_gnn_layers = n_gnn_layers
        self.gnn_layers = nn.ModuleList()
        if isinstance(num_heads, int):
            num_heads = [num_heads] * n_gnn_layers
        # num_heads[-1] = 1
        gnn_in_dim = self.text_emb_dim
        gnn_out_dim = self.hidden_size
        for l in range(n_gnn_layers):
            if l < n_gnn_layers - 1:
                act_fn = F.elu
            else:
                act_fn = None
            self.gnn_layers.append(GATConv(gnn_in_dim, 
                                           gnn_out_dim, 
                                           num_heads[l],
                                           activation=act_fn))
            gnn_in_dim = gnn_out_dim * num_heads[l]
        
        self.alpha = nn.Parameter(torch.tensor(-1e-3))
        self.seq_weight_scaler = seq_weight_scaler
        self.struct_pheno_comb = nn.Linear(self.seq_emb_dim + gnn_out_dim + self.text_emb_dim, self.hidden_size) # concatenated embedding of alt_seq, prot_desc, struct_context_pheno

        self.dist_fn = _dist_fn_map[dist_fn_name]
        # self.patho_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_label_idx)
        self.patho_loss_fn = nn.BCEWithLogitsLoss()
        # self.patho_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_label_idx)  # for softmax
        # self.desc_loss_fn = nn.CosineEmbeddingLoss()
        # self.cos_sim_loss_fn = nn.CosineEmbeddingLoss()
        self.contrast_loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=self.dist_fn, margin=init_margin, reduction='none')
        self.nce_loss_fn = InfoNCELoss()    

    # def binary_step(self, seq_input_feat, variant_data, desc_input_feat=None):
    #     # seq_embs, mlm_logits, desc_embs = self.protein_encoder(seq_input_feat, desc_input_feat)  # mlm_logits: (batch_size, max_seq_length, vocab_size=33)
    #     mlm_logits = self.protein_encoder.get_mlm_logits(seq_input_feat)
    #     var_indices = (variant_data['prot_idx'], variant_data['var_idx'])
    #     logit_diff = self.log_diff_patho_score(mlm_logits[var_indices], variant_data['ref_aa'], variant_data['alt_aa'])

    #     return mlm_logits, logit_diff
    def binary_step(self, seq_input_feat, variant_data):
        # seq_embs, mlm_logits, desc_embs = self.protein_encoder(seq_input_feat, desc_input_feat)  # mlm_logits: (batch_size, max_seq_length, vocab_size=33)
        mlm_logits = self.protein_encoder.get_mlm_logits(seq_input_feat)
        # var_indices = (variant_data['prot_idx'], variant_data['var_idx'])
        # logit_diff = self.log_diff_patho_score(mlm_logits[var_indices], variant_data['ref_aa'], variant_data['alt_aa'])
        probs = mlm_logits.softmax(dim=-1)
        logit_diff = 0
        # for single in mut_info.split(":"):
        var_idx = torch.tensor(variant_data['var_idx'], device=self.device) - variant_data['offset_idx']
        batch_idx = torch.arange(len(var_idx), device=self.device)
        # TODO: revise for struct-vocab
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

    def gnn_message_passing(self, graph, nfeats, efeats):
        h = nfeats
        for i, gnn_conv in enumerate(self.gnn_layers):
            h = gnn_conv(graph, h, efeats)  # n_nodes, num_heads, hidden_size
            if i < self.n_gnn_layers - 1:
                h = h.flatten(1)
            else:
                h = h.mean(1)

        return h      

    # def contrastive_step(self, seq_input_feat, variant_data, desc_input_feat):
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
            weighted_logits = prompt_weights * var_logit_diff
        else:
            prompt_weights = self.func_prompt_mlp(desc_emb_agg[variant_data['prot_idx']])
            prompt_weights = self.weight_act(prompt_weights)  # with activation (sigmoid)
            afmis_mask = variant_data['afmis_mask']  # True if alphamissense NOT available
            afmis_logits = torch.logit(variant_data['afmis_score'], eps=1e-6).unsqueeze(1)
            weighted_logits = prompt_weights * var_logit_diff + (1 - prompt_weights) * afmis_logits
            weighted_logits[afmis_mask] = var_logit_diff[afmis_mask]

        n_pheno_vars = variant_data['infer_pheno_vec'].sum().item()
        if n_pheno_vars == 0:  # Pathogenicity 
            return weighted_logits, prompt_weights, None, None, None, None
    
        max_text_length = self.text_encoder.config.max_position_embeddings
        # pheno_input_ids = variant_data['context_pheno_input_ids'].view(n_pheno_vars, -1)
        # pheno_attn_mask = variant_data['context_pheno_attention_mask'].view(n_pheno_vars, -1)
        pheno_input_ids = variant_data['context_pheno_input_ids']
        pheno_attn_mask = variant_data['context_pheno_attention_mask']
        pheno_indices = variant_data['context_pheno_indices']
        n_uniq_phenos = pheno_input_ids.shape[0]
        # assert pheno_input_ids.shape[0] == n_pheno_vars

        compute_pos = False
        compute_neg = False
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
        

        if 'pos_pheno_input_ids' in variant_data:  # positive phenotype label available
            compute_pos = True
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
        
        seq_context_pheno_emb_raw = torch.stack([seq_context_pheno_emb_raw[i, pheno_attn_mask[i, :].bool(), :][0] for i in range(n_uniq_phenos)], dim=0)
        # Update: use embedding corresponding to [CLS] token instead of average as sequence-level embedding 
        
        # variant_pheno_emb = torch.stack([variant_pheno_emb[i, pheno_attn_mask[i, :].bool(), :][0] for i in range(n_pheno_vars)], dim=0)
        # seq_var_embs = seq_embs[(variant_data['prot_idx'], variant_data['var_idx'])]  # extract embedding for target position
        # seq_pheno_emb_raw = torch.cat([seq_var_embs, variant_pheno_emb], dim=-1)  # n_var, (seq_emb_dim + pheno_emb_dim)
        seq_pheno_emb_raw = torch.cat([alt_seq_func_embs[variant_data['infer_pheno_vec'].bool()], seq_context_pheno_emb_raw], dim=-1)   # n_var, (seq_emb_dim + desc_emb_dim + pheno_emb_dim)
        seq_pheno_emb = self.seq_pheno_comb(seq_pheno_emb_raw)  # n_var, hidden_size

        # structural context
        struct_pheno_emb = None
        # combined_pheno_emb = seq_pheno_emb
        if variant_data['use_struct']:
            struct_pheno_input_ids = variant_data['struct_pheno_input_ids']
            struct_pheno_attention_mask = variant_data['struct_pheno_attention_mask']
            struct_context_pheno_emb_raw = self.text_encoder(
                struct_pheno_input_ids,
                attention_mask=struct_pheno_attention_mask,
                token_type_ids=torch.zeros(struct_pheno_input_ids.size(), dtype=torch.long, device=self.device),
                output_attentions=False,
                output_hidden_states=True,
                return_dict=None
            ).hidden_states[-1]
            struct_context_pheno_emb_agg = torch.stack([struct_context_pheno_emb_raw[i, struct_pheno_attention_mask[i, :].bool(), :][0] for i in range(struct_context_pheno_emb_raw.size(0))], dim=0)

            g_struct = variant_data['var_struct_graph']
            g_struct.ndata['pheno_emb'] = struct_context_pheno_emb_agg[g_struct.ndata['indice']]
            g_struct.ndata['pheno_emb'] = self.gnn_message_passing(g_struct, g_struct.ndata['pheno_emb'], g_struct.edata['distance'])
            # combein with protein sequence embedding & functional embedding
            struct_context_pheno_emb = g_struct.ndata['pheno_emb'][g_struct.ndata['mask'].bool()]
            struct_pheno_emb = torch.cat([alt_seq_func_embs[variant_data['infer_pheno_vec'].bool()][variant_data['has_struct_context']], struct_context_pheno_emb], dim=-1)
            struct_pheno_emb = self.struct_pheno_comb(struct_pheno_emb)
            # struct_mask = variant_data['has_struct_context']
            # combined_pheno_emb[struct_mask] = self.alpha * combined_pheno_emb[struct_mask] + (1 - self.alpha) * struct_pheno_emb

        if compute_pos:
            pos_pheno_embs = torch.stack([pos_pheno_embs[i, variant_data['pos_pheno_attention_mask'][i, :].bool()][0] for i in range(n_pheno_vars)], dim=0)
            pos_emb_proj = self.proj_head(pos_pheno_embs)
        else:
            pos_emb_proj = None
            
        if compute_neg:
            neg_pheno_embs = torch.stack([neg_pheno_embs[i, variant_data['neg_pheno_attention_mask'][i, :].bool()][0] for i in range(n_pheno_vars)], dim=0)
            neg_emb_proj = self.proj_head(neg_pheno_embs)
        else:
            neg_emb_proj = None

        return weighted_logits, prompt_weights, seq_pheno_emb, pos_emb_proj, neg_emb_proj, struct_pheno_emb

    
    def get_pheno_emb(self, pheno_input_dict, proj=True, agg_opt='mean'):
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
            if agg_opt == 'mean':
                pheno_embs = torch.stack([pheno_embs[i, pheno_input_dict['attention_mask'][i, :].bool()].mean(dim=0) for i in range(batch_size)], dim=0)
            else:  # take the dimension corresponding to [CLS] token as seq-level embedding
                pheno_embs = torch.stack([pheno_embs[i, pheno_input_dict['attention_mask'][i, :].bool()][0] for i in range(batch_size)], dim=0)
            pheno_embs = self.proj_head(pheno_embs)
                    
        return pheno_embs

    def contrast_loss(self, seq_var_emb, pos_emb, neg_emb, struct_var_emb=None, struct_mask=None):
        seq_contrast_loss = self.contrast_loss_fn(seq_var_emb, pos_emb, neg_emb)
        if isinstance(struct_var_emb, type(None)):
            return seq_contrast_loss.mean(), None, seq_contrast_loss.mean()
        struct_contrast_loss = self.contrast_loss_fn(struct_var_emb, pos_emb[struct_mask], neg_emb[struct_mask])
        combine_contrast_loss = seq_contrast_loss.clone()
        seq_weight = torch.sigmoid(self.alpha) * self.seq_weight_scaler
        # combine_contrast_loss[struct_mask] = torch.sigmoid(self.alpha) * combine_contrast_loss[struct_mask] + (1 - torch.sigmoid(self.alpha)) * struct_contrast_loss
        combine_contrast_loss[struct_mask] = seq_weight * combine_contrast_loss[struct_mask] + (1 - seq_weight) * struct_contrast_loss

        return seq_contrast_loss.mean(), struct_contrast_loss.mean(), combine_contrast_loss.mean()

        # return self.cos_sim_loss_fn(var_emb, pos_emb, targets[0]) + \
        #     self.cos_sim_loss_fn(var_emb, neg_emb, targets[1])
    
    def contrast_nce_loss(self, var_emb, pos_emb, neg_emb, temperature=0.07):
        sim_pos = torch.cosine_similarity(var_emb, pos_emb) / temperature
        sim_neg = torch.cosine_similarity(var_emb, neg_emb) / temperature
        sim_scores_all = torch.cat([sim_pos[:, None], sim_neg[:, None]], dim=-1)
        denom = torch.logsumexp(sim_scores_all, dim=-1)

        loss = -sim_pos + denom

        return loss.mean()

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

    def info_nce_loss(self, seq_var_emb, pheno_embs, positive_indices, negative_indices):
        
        return self.nce_loss_fn(seq_var_emb, pheno_embs, positive_indices, negative_indices)
