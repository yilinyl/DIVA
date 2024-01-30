import os
from pathlib import Path
import json
import copy
import numpy as np
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import tqdm
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from transformers import BertModel, BertTokenizer
from transformers import PreTrainedTokenizerBase
import pandas as pd

from dev.preprocess.utils import fetch_prot_seq


def fetch_variant_on_protein(uprot, prot_var_db, outcome="pathogenic"):
    """
    Fetch all variants that lead to `outcome` on protein

    Returns: Dict[int->str]
    """
    if uprot not in prot_var_db:
        return None
    if outcome:
        prot_vars = prot_var_db[uprot].get(outcome, None)
    else:
        prot_vars = dict()
        for key in prot_var_db[uprot].keys():
            prot_vars.update(prot_var_db[uprot][key])  # variants of all outcomes
    
    return prot_vars


class ProteinDataset(Dataset):

    def __init__(self,
                 protein_data,
                 protein_tokenizer,
                 text_tokenizer,
                 max_protein_seq_length=None,
                 max_phenotype_length=None):
        
        super(ProteinDataset, self).__init__()
        self.data = protein_data
        self.protein_tokenizer = protein_tokenizer
        self.text_tokenizer = text_tokenizer
        self.max_protein_seq_length = max_protein_seq_length
        self.max_phenotype_length = max_phenotype_length
    

    def prep_seq_input(self, prot_seq):
        aa_list = list(prot_seq)
        if self.max_protein_seq_length is not None:
            aa_list = aa_list[:self.max_protein_seq_length]

        seq_lm_input = dict()
        prot_input_ids = self.protein_tokenizer.encode(aa_list, max_length=self.max_protein_seq_length, is_split_into_words=True)
        seq_lm_input['input_ids'] = prot_input_ids
        seq_lm_input['attention_mask'] = (seq_lm_input['input_ids'] != self.protein_tokenizer.pad_token_id).long()
        seq_lm_input['token_type_ids'] = torch.zeros_like(seq_lm_input['input_ids'], dtype=torch.long)

        return seq_lm_input

    def __getitem__(self, index):
        cur_protein = self.data[index]
        aa_list = list(cur_protein['seq'])
        if self.max_protein_seq_length is not None:
            aa_list = aa_list[:self.max_protein_seq_length]

        prot_input_ids = self.protein_tokenizer.encode(aa_list, 
                                                       max_length=self.max_protein_seq_length, 
                                                       is_split_into_words=True, padding='max_length')
        desc_input_ids = self.text_tokenizer.encode(cur_protein['desc'], max_length=self.max_phenotype_length, padding='max_length')

        return {'id': cur_protein['id'],
                'seq_input_ids': prot_input_ids,
                'desc_input_ids': desc_input_ids}

    def __len__(self):
        return len(self.data)


class ProteinVariantDatset(Dataset):
    """
    Dataset for Protein variants.

    Args:
        data_dir: the directory for input files
        seq_dict: mappings of each UniProt ID to primary sequence
        protein_info_dict
        variant_info: mappings of each UniProt ID to list of variant tuples (pos, phenotype_list, pathogenicity)
        f_uprot_list: file for UniProt IDs
        protein_tokenizer: pretrained-tokenizer to encode protein sequences
        phenotype_tokenizer: pretrained-tokenizer to encode phenotype terms        
    """

    def __init__(
        self,
        data_dir: str,
        seq_dict: dict,
        protein_info_dict: dict = None,
        # variant_info: dict = None,
        variant_file: str = None,
        # df_var: pd.DataFrame = None,
        use_protein_desc: bool = False,
        protein_tokenizer: PreTrainedTokenizerBase = None,
        text_tokenizer: PreTrainedTokenizerBase = None,
        max_protein_seq_length: int = None,
        max_phenotype_length: int = 1000,
        # unknown_label = -100,
        phenotype_vocab: List = None,  # phenotype vocabulary
        # pheno_embs: List = None,  # pre-computed phenotype embeddings
        pos_offset = 1,
        pid_col='UniProt',
        pos_col='Protein_position',
        label_col='label',
        pheno_col='phenotype',
        split='train',
        var_db=None,
        prot_var_cache=None,
        patho_only=False,
        **kwargs
    ):
        super(ProteinVariantDatset, self).__init__()
        self.data_root = Path(data_dir)
        
        self.seq_dict = seq_dict
        self.protein_info_dict = protein_info_dict
        
        self.protein_variants = dict()  # dictionary of protein
        self.variant_data = []  # list of variant dict
        self.use_protein_desc = use_protein_desc
        self.protein_ids = []
        self.prot_seq = []
        # self.var_pos = []
        self.pos_offset = pos_offset  # 1 for 1-based position
        self.aa_change = []
        self.pheno_descs = phenotype_vocab
        self.pheno_idx_all = np.arange(len(phenotype_vocab))
        self.protein_tokenizer = protein_tokenizer
        self.text_tokenizer = text_tokenizer
        self.max_protein_seq_length = max_protein_seq_length
        self.max_phenotype_length = max_phenotype_length
        # self.label_map = {'benign': 0, 'pathogenic': 1, self.phenotype_tokenizer.unk_token: unknown_label}
        # self.pheno_embs = []
        self.unk_pheno_mask = []
        self.label = []
        self.label_col = label_col
        self.pheno_col = pheno_col
        self.var_db = var_db  # external variant information
        self.prot_var_cache = dict()
        if prot_var_cache:
            self.prot_var_cache = prot_var_cache

        self.split = split
        self.patho_only = patho_only

        if not variant_file:
            variant_file = split + '.csv'
        
        df_var = pd.read_csv(self.data_root / variant_file)
        
        self._load_variant_data(df_var, pid_col, pos_col)
    

    def _load_variant_data(self, df_var, pid_col='UniProt', pos_col='Protein_position'):

        # assert self.var_input_file != None
        # df_var = pd.read_csv(self.var_input_file)
        seq_only = not self.use_protein_desc
        prots_all = df_var[pid_col].drop_duplicates().tolist()
        self.protein_ids = prots_all

        for i, uprot in enumerate(prots_all):
            cur_protein = dict()
            # cur_protein['id'] = uprot
            context_var_idx = []
            if self.prot_var_cache:
                cur_protein = self.prot_var_cache.get(uprot, None)
            if cur_protein:  # protein found in cache
                seq = cur_protein['seq']
                prot_desc = cur_protein['prot_desc']
                context_var_idx = cur_protein['context_var_idx']
                pos_pheno_desc = cur_protein['var_pheno_descs']
            else:
                # Load sequence (and functional description)
                if uprot not in self.seq_dict:
                    prot_info = fetch_prot_seq(uprot, seq_only=seq_only)
                    if seq_only:
                        seq = prot_info
                    else:
                        seq = prot_info['sequence']['value']
                        prot_desc_dict = prot_info['proteinDescription']
                        if 'recommendedName' in prot_desc_dict:
                            prot_desc = prot_desc_dict['recommendedName']['fullName']['value']
                        elif 'submissionNames' in prot_desc_dict:
                            prot_desc = prot_desc_dict['submissionNames'][0]['fullName']['value']
                        else:
                            prot_desc = self.text_tokenizer.unk_token

                        self.protein_info_dict[uprot] = prot_desc
                        
                        # cur_protein['desc'] = prot_desc
                    
                    # cur_protein['seq'] = seq
                    self.seq_dict[uprot] = seq
                    
                else:
                    seq = self.seq_dict[uprot]
                    if self.use_protein_desc:
                        prot_desc = self.protein_info_dict[uprot]
            
            df_prot = df_var[df_var[pid_col] == uprot]
            if self.patho_only:
                df_prot = df_prot[df_prot[self.label_col == 1]]
            if len(df_prot) == 0:
                continue
            is_var = np.zeros(len(seq))
            pos_pheno_idx = np.full(len(seq), fill_value=self.text_tokenizer.pad_token_id)  # positive phenotypes
            # neg_pheno_idx = np.full(len(seq), fill_value=self.text_tokenizer.pad_token_id)  # negative phenotypes
            # pos_pheno_desc = np.full(len(seq), fill_value=self.text_tokenizer.pad_token)
            pos_pheno_desc = [self.text_tokenizer.pad_token] * len(seq)
            # neg_pheno_desc = [self.text_tokenizer.pad_token] * len(seq)
            # neg_pheno_desc = np.full(len(seq), fill_value=self.text_tokenizer.pad_token)
            prot_var_pos = []
            # prot_var_idx_target = []
            # is_pathogenic = np.zeros(len(seq))
            # pos_phenotypes = []
            # neg_phenotypes = []
            ref_aa = []
            alt_aa = []
            var_names = []
            labels = []
            
            for j, record in df_prot.iterrows():
                pheno_cur = record[self.pheno_col]
                if record[pos_col] > self.max_protein_seq_length:  # skip out-of-bound variants for now
                    continue
                sample_mask = np.zeros(len(self.pheno_descs), dtype=bool)
                var_idx = record[pos_col] - self.pos_offset
                is_var[var_idx] = 1
                if self.label_col in record:
                    labels.append(record[self.label_col])

                infer_phenotype = False
                if pheno_cur != np.nan and record[self.label_col] == 1:  # pathogenic AND phenotype information available
                    try:
                        pheno_idx = self.pheno_descs.index(pheno_cur)
                        sample_mask[pheno_idx] = True
                        infer_phenotype = True
                    except ValueError:
                        pheno_idx = self.text_tokenizer.unk_token_id
                        pheno_cur = self.text_tokenizer.unk_token

                    # if self.split != 'test':
                    neg_sample_idx = np.random.choice(self.pheno_idx_all[~sample_mask], size=1, replace=False)[0]  # scalar
                    # neg_pheno_idx[var_idx] = neg_sample_idx
                    neg_pheno_cur = self.pheno_descs[neg_sample_idx]
                    # neg_pheno_desc.append(neg_pheno_cur)
                    # neg_pheno_desc[var_idx] = neg_pheno_cur
                    
                    pos_pheno_idx[var_idx] = pheno_idx
                    pos_pheno_desc[var_idx] = pheno_cur

                    context_var_idx.append(var_idx)  # context information for phenotype inference
                
                else:
                    pheno_idx = self.text_tokenizer.unk_token_id
                    pheno_cur = self.text_tokenizer.unk_token
                    # neg_sample_idx = self.text_tokenizer.unk_token_id
                    # neg_pheno_cur = self.text_tokenizer.unk_token
                
                ref_aa.append(record['REF_AA'])
                alt_aa.append(record['ALT_AA'])
                cur_var_name = '{}_{}_{}/{}'.format(uprot, record[pos_col], record['REF_AA'], record['ALT_AA'])
                var_names.append(cur_var_name)
                # var_names.append(''.join([record['REF_AA'], str(record[pos_col]), record['ALT_AA']]))
                prot_var_pos.append(record[pos_col])
                # prot_var_idx_target.append(var_idx)

                cur_variant = {'id': cur_var_name,
                               'uprot': uprot,
                               'var_idx': var_idx,
                               'var_pos': record[pos_col],
                               'label': record[self.label_col],
                               'ref_aa': self.protein_tokenizer.convert_tokens_to_ids(record['REF_AA']),
                               'alt_aa': self.protein_tokenizer.convert_tokens_to_ids(record['ALT_AA']),
                               'pos_pheno_desc': pheno_cur,
                               'pos_pheno_idx': pheno_idx,
                            #    'neg_pheno_desc': neg_pheno_cur,
                            #    'neg_pheno_idx': neg_sample_idx,
                               'infer_phenotype': infer_phenotype
                               }
                self.variant_data.append(cur_variant)
            # seq_input_ids = self.encode_protein_seq(seq)
            cur_protein = {'seq': seq,
                           'seq_length': len(seq),
                           'prot_desc': prot_desc,
                           'context_var_idx': context_var_idx,  
                           'var_pheno_descs': pos_pheno_desc,
                           }
            
            self.protein_variants[uprot] = cur_protein
            self.prot_var_cache.update(self.protein_variants)

    def encode_protein_seq(self, seq):
        aa_list = list(seq)
        if self.max_protein_seq_length is not None:
            aa_list = aa_list[:self.max_protein_seq_length]

        prot_input_ids = self.protein_tokenizer.encode(aa_list, padding=True,
                                                    #    max_length=self.max_protein_seq_length, 
                                                       is_split_into_words=True)
        
        return prot_input_ids

    def __getitem__(self, index):
        # return self.protein_variants[index]
        return self.variant_data[index]

    def __len__(self):

        # return len(self.protein_variants)
        return len(self.variant_data)
    
    @property
    def n_protein(self):
        assert len(self.protein_ids) == len(self.protein_variants)
        return len(self.protein_ids)
    
    def get_protein_list(self):
        return self.protein_ids

    def get_protein_data(self):
        return self.protein_variants
    
    def get_protein_cache(self):
        return self.prot_var_cache


@dataclass
class ProteinVariantDataCollator:
    """
    Data collator used for language model. Inputs are dynamically padded to the maximum length
    of a batch if they are not all of the same length.
    The class is rewrited from 'Transformers.data.data_collator.DataCollatorForLanguageModeling'.
        
    Agrs:
        tokenizer: the tokenizer used for encoding sequence.
        mlm: Whether or not to use masked language modeling. If set to 'False', the labels are the same as the
            inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
            non-masked tokens and the value to predict for the masked token.
        mlm_probability: the probablity of masking tokens in a sequence.
        are_protein_length_same: If the length of proteins in a batch is different, protein sequence will
                                 are dynamically padded to the maximum length in a batch.
    """
    protein_data: Dict
    protein_tokenizer: PreTrainedTokenizerBase
    text_tokenizer: PreTrainedTokenizerBase = None
    mlm: bool = False  # for masked language model (not implemented yet)
    mlm_probability: float = 0.15
    same_length: bool = False
    use_desc: bool = False
    label_pad_idx: int = -100
    phenotype_vocab: List = None
    window_size: int = 64
    max_protein_length: int = 1024

    def __post_init__(self):
        if self.mlm and self.protein_tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )
        
    def __call__(
        self,
        batch_data_raw: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        batch = protein_variant_collate_fn(batch_data_raw, self.protein_tokenizer, self.text_tokenizer, self.protein_data,
                                           pheno_vocab=self.phenotype_vocab, use_desc=self.use_desc, 
                                           window_size=self.window_size, max_protein_length=self.max_protein_length)

        return batch
    

    def mask_variants(
        self,
        inputs: torch.Tensor,
        tokenizer: PreTrainedTokenizerBase,
        variant_mask: Optional[torch.Tensor] = None,  # True for variant positions
        vocab_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = inputs.clone()
        if not vocab_size:
            vocab_size = tokenizer.vocab_size
        
        variant_mask = variant_mask.bool()
        n_variants = variant_mask.sum().item()

        # only compute loss on masked tokens.
        labels[variant_mask] = self.text_tokenizer.mask_token_id
        # 80% of the time, replace masked input tokens with tokenizer.mask_token
        indices_replaced = torch.bernoulli(torch.full(labels.shape, fill_value=0.8)).bool() & variant_mask
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, fill_value=0.5)).bool() & variant_mask & ~indices_replaced
        random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels, variant_mask, indices_replaced, indices_random
    
    def mask_tokens(
        self,
        inputs: torch.Tensor,
        tokenizer: PreTrainedTokenizerBase,
        special_tokens_mask: Optional[torch.Tensor] = None,
        vocab_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling:
        default: 80% MASK, 10%  random, 10% original
        """
        labels = inputs.clone()
        if not vocab_size:
            vocab_size = tokenizer.vocab_size
        probability_matrix = torch.full(labels.size(), fill_value=self.mlm_probability)
        # if `special_tokens_mask` is None, generate it by `labels`
        if special_tokens_mask is None:
            special_tokens_mask = [
                tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        # tokens are zero-out for masking if it has special_tokens_mask == True
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # only compute loss on masked tokens.
        labels[~masked_indices] = self.text_tokenizer.mask_token_id

        # 80% of the time, replace masked input tokens with tokenizer.mask_token
        indices_replaced = torch.bernoulli(torch.full(labels.shape, fill_value=0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, fill_value=0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels, masked_indices, indices_replaced, indices_random
    

def protein_seq_collate_fn(
    batch_data_raw: List[int], 
    tokenizer: PreTrainedTokenizerBase,
    same_length: bool = False
):
    input_id_tensors = [torch.LongTensor(elem['input_ids']) for elem in batch_data_raw]
    batch_pos = torch.LongTensor([elem['var_pos'] for elem in batch_data_raw])
    
    if same_length:
        return torch.stack(input_id_tensors, dim=0)

    batch_size = len(input_id_tensors)

    max_length = max(x.size(0) for x in input_id_tensors)
    input_ids_pad = input_id_tensors[0].new_full([batch_size, max_length], fill_value=tokenizer.pad_token_id)
    for i, input_ids in enumerate(input_id_tensors):
        if tokenizer.padding_side == 'right':
            input_ids_pad[i, :input_ids.size(0)] = input_ids
        else:
            input_ids_pad[i, -input_ids.size(0):] = input_ids
    
    batch_data_dict = {'uprot': [elem['uprot'] for elem in batch_data_raw],
                       'input_ids': input_ids_pad,
                       'seq': [elem['seq'] for elem in batch_data_raw],
                       'var_pos': batch_pos,
                       'ref_aa': [elem['ref_aa'] for elem in batch_data_raw],
                       'alt_aa': [elem['alt_aa'] for elem in batch_data_raw],
                       }
    if 'label' in batch_data_raw[0]:
        batch_data_dict['label'] = torch.LongTensor([elem['label'] for elem in batch_data_raw])
    
    return batch_data_dict


def fetch_phenotypes_in_frame(
            var_idx: int,
            seq_length: int,
            context_var_idx: List[int],
            pos_pheno_descs: List[str],
            window_size: int = 64,
            max_num: int = None,
            mask_token: str = '[MASK]'
    ):
        # assert len(pos_pheno_descs) == len(var_mask)

        start = max(var_idx - window_size, 0)
        end = min(var_idx + window_size, seq_length - 1)
        target_idx = var_idx - start

        pheno_in_frame = []
        for idx in sorted(set(context_var_idx)):
            if idx == var_idx:
                pheno_in_frame.append(mask_token)
            elif idx in range(start, end+1):
                pheno_in_frame.append(pos_pheno_descs[idx])
            if idx > end:
                break
        target_pheno_loc = pheno_in_frame.index(mask_token)
        if max_num:
            if len(pheno_in_frame) > max_num:
                if target_pheno_loc <= max_num - 1:
                    pheno_in_frame = pheno_in_frame[:max_num]  # right truncation
                else:
                    pheno_in_frame = pheno_in_frame[-max_num:]
        
        # pheno_in_frame.pop(target_pheno_loc)

        return pheno_in_frame


def protein_variant_collate_fn(
    batch_data_raw: List[Dict],
    protein_tokenizer: PreTrainedTokenizerBase,
    text_tokenizer: PreTrainedTokenizerBase,
    protein_data: Dict,
    pheno_vocab: List[str],
    use_desc: bool = True,
    window_size: int = 64,
    max_protein_length: int = None,
    max_context_phenos: int = None
):   
    """
    Collate function for protein using both sequence and description text
    """
    batch_prot_ids = [elem['uprot'] for elem in batch_data_raw]
    batch_size = len(batch_data_raw)
    prot_unique = list(set(batch_prot_ids))
    # protein sequence
    seq_lst = [protein_data[pid]['seq'] for pid in prot_unique]
    prot_lengths = [len(seq) for seq in seq_lst]
    # max_seq_length = max(prot_lengths)
    if not max_protein_length:
        max_protein_length = protein_tokenizer.model_max_length
    else:
        max_protein_length = min(max_protein_length, protein_tokenizer.model_max_length)

    # seq_lst = [elem['seq'] for elem in batch_data_raw]
    batch_seq_tokenized = protein_tokenizer(seq_lst, padding=True, truncation=True, return_tensors='pt', max_length=max_protein_length)
    seq_tokenized_length = batch_seq_tokenized['input_ids'].shape[-1]
    # batch_var_mask = [torch.LongTensor(elem['variant_mask']) for elem in batch_data_raw]
    # batch_patho_mask = [torch.LongTensor(elem['is_pathogenic']) for elem in batch_data_raw]
    # batch_desc_input_ids = [torch.LongTensor(elem['desc_input_ids']) for elem in batch_data_raw]
    # max_desc_length = max([len(elem['desc_input_ids']) for elem in batch_data_raw])

    patho_var_prot_idx = []  # index of protein in the batch for pathogenic variants
    prot_idx_all = []
    var_idx_all = []
    ref_aa = []
    alt_aa = []
    pos_pheno_label_all = []
    neg_pheno_label_all = []
    phenos_in_frame_all = []
    var_names_all = []
    batch_label = []
    pheno_var_names = []
    infer_pheno_vec = []
    # phenos_in_frame_input_ids = [] # List[torch.Tensor] --> length=n_variants
    for b, elem in enumerate(batch_data_raw):
        # var_idx_lst = elem['var_idx']
        uprot = elem['uprot']
        prot_idx_cur = prot_unique.index(uprot)
        
        protein_info_cur = protein_data[uprot]
        prot_context_var_idx = protein_info_cur['context_var_idx']
        prot_desc = protein_info_cur['prot_desc']
        pos_pheno_descs = elem['pos_pheno_desc']
        # neg_pheno_descs = elem['neg_pheno_desc']
        ref_aa.append(elem['ref_aa'])
        alt_aa.append(elem['alt_aa'])
        var_names_all.append(elem['id'])
        var_idx = elem['var_idx']
        batch_label.append(elem['label'])
        infer_pheno_vec.append(int(elem['infer_phenotype']))
        
        var_idx_all.append(var_idx)
        prot_idx_all.append(prot_idx_cur)
        # for var_idx in var_idx_lst:
        if elem['infer_phenotype']:
            phenos_in_frame_cur = fetch_phenotypes_in_frame(var_idx, prot_lengths[prot_idx_cur], prot_context_var_idx, 
                                                            protein_info_cur['var_pheno_descs'], window_size, max_context_phenos, 
                                                            mask_token=text_tokenizer.mask_token)
            patho_var_prot_idx.append(prot_idx_cur)
            pheno_var_names.append(elem['id'])
            pos_pheno_label_all.append(pos_pheno_descs)
            # neg_pheno_label_all.append(neg_pheno_descs)
            phenos_in_frame_all.append([prot_desc] + phenos_in_frame_cur)
            neg_pheno_idx, neg_pheno_desc = sample_negative(pheno_vocab, pos_pheno_descs)
            neg_pheno_label_all.append(neg_pheno_desc)
        # phenos_all.extend([prot_desc] + phenos_in_frame_cur)
            # max_pheno_length = max(max_pheno_length, phenos_input_ids_cur.shape[-1])
            # phenos_in_frame_input_ids.append(phenos_input_ids_cur) 
    if use_desc:
            desc_lst = [protein_data[pid]['prot_desc'] for pid in prot_unique]
            batch_desc_tokenized = text_tokenizer(desc_lst, padding=True, return_tensors='pt')   
            # max_desc_length = batch_desc_tokenized['input_ids'].shape[-1]

    if sum(infer_pheno_vec) == 0:
        variant_dict = {
            'var_pos': [elem['var_pos'] for elem in batch_data_raw],
            'var_idx': var_idx_all,
            'var_names': var_names_all,
            'ref_aa': torch.LongTensor(ref_aa),
            'alt_aa': torch.LongTensor(alt_aa),
            'infer_phenotype': False,
            # 'n_variants': [len(elem['alt_aa']) for elem in batch_data_raw],
            'prot_idx': prot_idx_all
        }
    else:
        pos_pheno_tokenized = text_tokenizer(pos_pheno_label_all, padding=True, return_tensors='pt')
        neg_pheno_tokenized = text_tokenizer(neg_pheno_label_all, padding=True, return_tensors='pt')
        # max_pheno_length = max(max_desc_length, pos_pheno_tokenized['input_ids'].shape[-1])
        max_phenos_in_frame = max([len(ph) for ph in phenos_in_frame_all])
        phenos_in_frame_padded = []
        for i, phenos_in_frame_cur in enumerate(phenos_in_frame_all):
            n_phenos = len(phenos_in_frame_cur)
            if n_phenos < max_phenos_in_frame:
                phenos_in_frame_cur = phenos_in_frame_cur + [text_tokenizer.pad_token] * (max_phenos_in_frame - n_phenos)
            phenos_in_frame_padded.extend(phenos_in_frame_cur)
            
        batch_pheno_tokenized = text_tokenizer(phenos_in_frame_padded, padding=True, return_tensors='pt')
        batch_pheno_input_ids = batch_pheno_tokenized['input_ids'].view(sum(infer_pheno_vec), max_phenos_in_frame, -1)
        # batch_pheno_input_ids = torch.stack(pheno_input_ids_padded)  # n_variants, max_phenos_in_frame + 1, max_pheno_length
        batch_pheno_attenton_mask = (batch_pheno_input_ids != text_tokenizer.pad_token_id).long()
        variant_dict = {
            'var_pos': [elem['var_pos'] for elem in batch_data_raw],
            'var_idx': var_idx_all,
            'var_names': var_names_all,
            'pheno_var_names': pheno_var_names,
            'ref_aa': torch.LongTensor(ref_aa),
            'alt_aa': torch.LongTensor(alt_aa),
            'infer_phenotype': True,
            'pos_pheno_desc': pos_pheno_label_all,
            'pos_pheno_input_ids': pos_pheno_tokenized['input_ids'],
            'pos_pheno_attention_mask': pos_pheno_tokenized['attention_mask'],
            'neg_pheno_desc': neg_pheno_label_all,
            'neg_pheno_input_ids': neg_pheno_tokenized['input_ids'],
            'neg_pheno_attention_mask': neg_pheno_tokenized['attention_mask'],
            'context_pheno_input_ids': batch_pheno_input_ids,
            'context_pheno_attention_mask': batch_pheno_attenton_mask,
            # 'n_variants': [len(elem['alt_aa']) for elem in batch_data_raw],
            'patho_var_prot_idx': patho_var_prot_idx,
            'prot_idx': prot_idx_all
        }
    
    variant_dict['label'] = torch.tensor(batch_label)
    variant_dict['infer_pheno_vec'] = torch.tensor(infer_pheno_vec)
    
    return {
        # 'id': batch_prot_ids,
        'seq_length': prot_lengths, 
        # 'variant_mask': batch_var_mask,  # true for variant (batch_size, max_length)
        # 'pathogenic_mask': batch_patho_mask,  # true for pathogenic (batch_size, max_length)
        'label': batch_label,  # 1: pathogenic, 0: benign, -100: non-variant spots
        
        'seq_input_feat': {
            'input_ids': batch_seq_tokenized['input_ids'],  # batch_size x max_length
            'attention_mask': batch_seq_tokenized['attention_mask'],  # 1 if token should be attended to
            'token_type_ids': torch.zeros_like(batch_seq_tokenized['input_ids'], dtype=torch.long)
        },
        'desc_input_feat': {
            'input_ids': batch_desc_tokenized['input_ids'],
            'attention_mask': batch_desc_tokenized['attention_mask'],
            'token_type_ids': torch.zeros_like(batch_desc_tokenized['input_ids'], dtype=torch.long)
        },
        'variant': variant_dict
    }


def sample_negative(pheno_vocab, pos_pheno_desc):
    sample_mask = np.zeros(len(pheno_vocab), dtype=bool)
    pheno_idx_all = np.arange(len(pheno_vocab))
    try:
        pos_idx = pheno_vocab.index(pos_pheno_desc)
        sample_mask[pos_idx] = True
    except ValueError:
        pass
    neg_sample_idx = np.random.choice(pheno_idx_all[~sample_mask], size=1, replace=False)[0]  # scalar
    neg_pheno_desc = pheno_vocab[neg_sample_idx]

    return neg_sample_idx, neg_pheno_desc


def pad_phenotype_input(raw_input_ids: List[List[int]],  
                        # var_idx_list, 
                        max_text_length, 
                        seq_length,
                        padding_side='right', 
                        fill_val=0):
    
    # assert len(var_idx_list) == len(raw_input_ids)

    x_pad = torch.full((seq_length, max_text_length), fill_value=fill_val, dtype=torch.long)
    for i, cur_input in enumerate(raw_input_ids):
        cur_length = len(cur_input)
        if padding_side == 'right':
            x_pad[i, :cur_length] = torch.tensor(cur_input)
        else:
            x_pad[i, -cur_length:] = torch.tensor(cur_input)
            
    return x_pad


def prep_padded_input(x: List[torch.Tensor], max_length, padding_side='right', init_value=0):
    batch_size = len(x)
    x_pad = torch.full((batch_size, max_length), fill_value=init_value)
    for i, x_cur in enumerate(x):
        cur_length = x_cur.size(0)
        if cur_length <= max_length:
            if padding_side == 'right':
                x_pad[i, :cur_length] = x_cur
            else:
                x_pad[i, -cur_length:] = x_cur
        elif cur_length > max_length:
            x_pad[i] = x_cur[:max_length]
    
    return x_pad
            

class PhenotypeDataset(Dataset):
    """
    Dataset for Protein sequence.

    Args:
        data_dir: the diractory need contain pre-train datasets.
        tokenizer: tokenizer used for encoding sequence.
    """

    def __init__(
        self,
        phenotypes: List[str],
        # tokenizer: PreTrainedTokenizerBase = None,
        # max_phenotype_length: int = None,
        # embed_all = True
    ):
        
        self.phenotypes = phenotypes
        # self.tokenizer = tokenizer
        
    def __getitem__(self, index):
        return self.phenotypes[index]

    def __len__(self):
        return len(self.phenotypes)

@dataclass
class TextDataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    # truncation: bool = True
    # max_length: int = None

    def __call__(self, batch):
        batch_tokenized = self.tokenizer(batch, padding=self.padding, return_tensors='pt')
        return batch_tokenized

@dataclass
class DataCollatorForLanguageModeling:
    """
    Data collator used for language model. Inputs are dynamically padded to the maximum length
    of a batch if they are not all of the same length.
    The class is rewrited from 'Transformers.data.data_collator.DataCollatorForLanguageModeling'.
        
    Agrs:
        tokenizer: the tokenizer used for encoding sequence.
        mlm: Whether or not to use masked language modeling. If set to 'False', the labels are the same as the
            inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
            non-masked tokens and the value to predict for the masked token.
        mlm_probability: the probablity of masking tokens in a sequence.
        are_protein_length_same: If the length of proteins in a batch is different, protein sequence will
                                 are dynamically padded to the maximum length in a batch.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = False
    mlm_probability: float = 0.15
    same_length: bool = False

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )
    
    def __call__(
        self,
        examples: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        batch = {'input_ids': protein_seq_collate_fn(examples, self.tokenizer, self.same_length)}
        special_tokens_mask = batch.pop('special_tokens_mask', None)  # always None

        if self.mlm:
            batch['input_ids'], batch['labels'] = self.mask_tokens(
                batch['input_ids'], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch['input_ids'].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch['labels'] = labels

        batch['attention_mask'] = (batch['input_ids'] != self.tokenizer.pad_token_id).long()
        batch['token_type_ids'] = torch.zeros_like(batch['input_ids'], dtype=torch.long)
        return batch
    

    def mask_tokens(
        self,
        inputs: torch.Tensor,
        special_tokens_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling:
        default: 80% MASK, 10%  random, 10% original
        """
        labels = inputs.clone()
        probability_matrix = torch.full(labels.size(), fill_value=self.mlm_probability)
        # if `special_tokens_mask` is None, generate it by `labels`
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # only compute loss on masked tokens.
        labels[~masked_indices] = -100

        # 80% of the time, replace masked input tokens with tokenizer.mask_token
        indices_replaced = torch.bernoulli(torch.full(labels.shape, fill_value=0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, fill_value=0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

