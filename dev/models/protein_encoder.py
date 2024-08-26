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
from transformers import BertModel, BertTokenizer
from transformers import PreTrainedTokenizerBase, PreTrainedModel
import pandas as pd
import dgl
from dgl.nn.pytorch import GATConv

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
    
    def get_mlm_logits(self, seq_input_data):
        seq_outputs = self.seq_encoder(
            seq_input_data['input_ids'],
            attention_mask=seq_input_data['attention_mask'],
            # token_type_ids=seq_input_data['token_type_ids'],
            output_attentions=False,
            output_hidden_states=False,
        )

        return seq_outputs.logits

    
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
                 n_gnn_layers=2,
                 max_vars_per_batch=32,
                 dist_fn_name='cosine_sigmoid',
                 init_margin=1,
                 freq_norm_factor=None,
                 seq_weight_scaler=1,
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
        self.freq_norm_factor = freq_norm_factor

        self.max_vars_per_batch = max_vars_per_batch

        self.seq_pheno_comb = nn.Linear(self.seq_emb_dim + self.text_emb_dim * 2, self.hidden_size) # concatenated embedding of alt_seq, prot_desc, context_pheno

        self.proj_head = nn.Linear(self.text_emb_dim, self.hidden_size)

        self.patho_output_layer = nn.Linear(self.seq_emb_dim + self.text_emb_dim, 2)  # use concatenated embedding of alt_seq and prot_desc
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
        # self.patho_loss_fn = nn.BCEWithLogitsLoss()
        self.patho_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_label_idx)  # for softmax
        # self.desc_loss_fn = nn.CosineEmbeddingLoss()
        # self.cos_sim_loss_fn = nn.CosineEmbeddingLoss()
        self.contrast_loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=self.dist_fn, margin=init_margin, reduction='none')
        self.nce_loss_fn = InfoNCELoss()

    # def forward(self, seq_input_feat, variant_data, desc_input_feat=None):
        
    #     # seq_embs, mlm_logits, _ = self.protein_encoder(seq_input_feat, desc_input_feat)  # mlm_logits: (batch_size, max_seq_length, vocab_size=33)
    #     # mlm_logits, logit_diff = self.binary_step(seq_input_feat, variant_data, desc_input_feat)
        
    #     if variant_data['infer_phenotype']:
    #         # n_pheno_vars = len(variant_data['patho_var_prot_idx'])
    #         seq_pheno_emb, pos_emb_proj, neg_emb_proj = self.contrastive_step(seq_input_feat, variant_data, desc_input_feat)
        
    #     else:  # no valid phenotype label in the batch, skip phenotype inference
    #         seq_pheno_emb = None
    #         pos_emb_proj = None
    #         neg_emb_proj = None

    #     # var_indices = (variant_data['prot_idx'], variant_data['var_idx'])
    #     # logit_diff = self.log_diff_patho_score(mlm_logits[var_indices], variant_data['ref_aa'], variant_data['alt_aa'])
    #     return seq_pheno_emb, pos_emb_proj, neg_emb_proj
        # return seq_pheno_emb, pos_emb_proj, neg_emb_proj, mlm_logits, logit_diff
    

    def binary_step(self, seq_input_feat, variant_data, desc_input_feat=None):
        # seq_embs, mlm_logits, desc_embs = self.protein_encoder(seq_input_feat, desc_input_feat)  # mlm_logits: (batch_size, max_seq_length, vocab_size=33)
        mlm_logits = self.protein_encoder.get_mlm_logits(seq_input_feat)
        var_indices = (variant_data['prot_idx'], variant_data['var_idx'])
        logit_diff = self.log_diff_patho_score(mlm_logits[var_indices], variant_data['ref_aa'], variant_data['alt_aa'])

        return mlm_logits, logit_diff

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
    def forward(self, seq_input_feat, variant_data, desc_input_feat):
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
        # alt_seq_input_feat = {
        #     'input_ids': variant_data['var_seq_input_ids'][variant_data['infer_pheno_vec'].bool()],
        #     'attention_mask': seq_input_feat['attention_mask'][variant_data['patho_var_prot_idx']]
        # }
        alt_seq_input_feat = {  # All ALT seq
            'input_ids': variant_data['var_seq_input_ids'],
            'attention_mask': seq_input_feat['attention_mask'][variant_data['prot_idx']]
        }
        alt_seq_embs = self.protein_encoder.embed_protein_seq(alt_seq_input_feat)
        alt_seq_embs = torch.stack([alt_seq_embs[i, alt_seq_input_feat['attention_mask'][i, :].bool(), :][0] for i in range(alt_seq_embs.size(0))], dim=0)
        alt_seq_func_embs = torch.cat([alt_seq_embs, desc_emb_agg[variant_data['prot_idx']]], dim=-1)  # concatenate alt-seq embedding & prot-desc embedding
        var_patho_logits = self.patho_output_layer(alt_seq_func_embs)

        n_pheno_vars = variant_data['infer_pheno_vec'].sum().item()
        if n_pheno_vars == 0:  # Pathogenicity 
            return var_patho_logits, None, None, None, None
    
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
        pheno_var_indices = torch.unique(pheno_indices)
        emb_agg_lst = []
        for var_idx in pheno_var_indices:
            cur_var_indices = pheno_indices == var_idx
            if cur_var_indices.sum() == 1:
                emb_agg_lst.append(seq_context_pheno_emb_raw[cur_var_indices])
            else:
                cur_counts = variant_data['context_pheno_counts'][cur_var_indices]
                if not self.freq_norm_factor:  # relative frequency
                    emb_agg_lst.append((seq_context_pheno_emb_raw[cur_var_indices] * cur_counts.unsqueeze(1) / cur_counts.sum()).sum(0).unsqueeze(0))
                else:  # absolute frequency (with normalization factor)
                    emb_agg_lst.append((seq_context_pheno_emb_raw[cur_var_indices] * cur_counts.unsqueeze(1) / self.freq_norm_factor).sum(0).unsqueeze(0))
        seq_context_pheno_emb_raw = torch.concat(emb_agg_lst, dim=0)
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

        return var_patho_logits, seq_pheno_emb, pos_emb_proj, neg_emb_proj, struct_pheno_emb

    
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


class InfoNCELoss(nn.Module):
    """
    Modified InfoNCE loss
    """
    def __init__(self, 
                 temperature=0.07,
                 reduce=True,
                 normalize=True):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        assert self.temperature > 0.0
        self.reduce = reduce
        self.normalize = normalize

    def forward(self, query_embs, ref_embs, positive_indices, negative_indices):
        """
        query_embs: Tensor of shape (batch_size, embedding_dim)
        ref_embs: Tensor of shape (vocab_size, embedding_dim)
        """        
        # Normalize embeddings to compute cosine similarity
        if self.normalize:
            query_embs_norm = F.normalize(query_embs, dim=-1)
            ref_embs_norm = F.normalize(ref_embs, dim=-1)
            # negatives = F.normalize(negatives, dim=-1)
        else:
            query_embs_norm = query_embs
            ref_embs_norm = ref_embs
        
        batch_size = query_embs.size(0)
        row_indices = torch.arange(batch_size).unsqueeze(1)
        indices_all = torch.cat([positive_indices.unsqueeze(1), negative_indices], dim=-1)
        
        # positive_embs = ref_embs_norm[positive_indices]  # (batch_size, batch_size)
        cos_sim_with_temp = torch.matmul(query_embs_norm, ref_embs_norm.t()) / self.temperature  # (batch_size, vocab_size)
        numer = cos_sim_with_temp[row_indices, positive_indices.unsqueeze(1)]  # (batch_size, )  
        denom = torch.logsumexp(cos_sim_with_temp[row_indices, indices_all], dim=-1).unsqueeze(1)

        loss = -numer + denom
        if self.reduce:
            return loss.mean()
        return loss
