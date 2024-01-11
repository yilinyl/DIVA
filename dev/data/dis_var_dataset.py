import os
from pathlib import Path
import json
import copy
import numpy as np
import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
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
        **kwargs
    ):
        super(ProteinVariantDatset, self).__init__()
        self.data_root = Path(data_dir)
        
        self.seq_dict = seq_dict
        self.protein_info_dict = protein_info_dict
        
        self.protein_variants = []  # list of protein dict
        # self.variant_data = []  # list of variant dict
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

        self.split = split

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

        cur_protein = dict()

        for i, uprot in enumerate(prots_all):
            # cur_protein['id'] = uprot

            # Load sequence (and functional description)
            if uprot not in self.seq_dict:
                prot_info = fetch_prot_seq(uprot, seq_only=seq_only)
                if seq_only:
                    seq = prot_info
                else:
                    seq = prot_info['sequence']['value']
                    prot_desc = prot_info['proteinDescription']['recommendedName']['fullName']['value']
                    self.protein_info_dict[uprot] = prot_desc
                    # cur_protein['desc'] = prot_desc
                
                # cur_protein['seq'] = seq
                self.seq_dict[uprot] = seq
                
            else:
                seq = self.seq_dict[uprot]
                if self.use_protein_desc:
                    prot_desc = self.protein_info_dict[uprot]
            
            df_prot = df_var[df_var[pid_col] == uprot]
            is_var = np.zeros(len(seq))
            pos_pheno_idx = np.full(len(seq), fill_value=self.text_tokenizer.pad_token_id)  # positive phenotypes
            neg_pheno_idx = np.full(len(seq), fill_value=self.text_tokenizer.pad_token_id)  # negative phenotypes
            # pos_pheno_desc = np.full(len(seq), fill_value=self.text_tokenizer.pad_token)
            pos_pheno_desc = [self.text_tokenizer.pad_token] * len(seq)
            neg_pheno_desc = [self.text_tokenizer.pad_token] * len(seq)
            # neg_pheno_desc = np.full(len(seq), fill_value=self.text_tokenizer.pad_token)
            prot_var_pos = []
            prot_var_idx = []
            is_pathogenic = np.zeros(len(seq))
            # pos_phenotypes = []
            # neg_phenotypes = []
            ref_aa = []
            alt_aa = []
            labels = []
            # patho_var_pos = []
            # patho_var_pheno_idx = []
            for j, record in df_prot.iterrows():
                pheno_cur = record[self.pheno_col]
                sample_mask = np.zeros(len(self.pheno_descs), dtype=bool)
                try:
                    pheno_idx = self.pheno_descs.index(pheno_cur)
                    sample_mask[pheno_idx] = True
                except ValueError:
                    pheno_idx = self.text_tokenizer.unk_token_id
                    pheno_cur = self.text_tokenizer.unk_token
                                
                var_idx = record[pos_col] - self.pos_offset
                is_var[var_idx] = 1

                if self.split != 'test':
                    neg_sample_idx = np.random.choice(self.pheno_idx_all[~sample_mask], size=1, replace=False)[0]  # scalar
                    neg_pheno_idx[var_idx] = neg_sample_idx
                    neg_pheno_cur = self.pheno_descs[neg_sample_idx]
                    # neg_pheno_desc.append(neg_pheno_cur)
                    neg_pheno_desc[var_idx] = neg_pheno_cur
                    # neg_phenotypes.append(neg_sample_idx[0])
                else:
                    neg_sample_idx = None
                
                if self.label_col in record:
                    labels.append(record[self.label_col])
                
                pos_pheno_idx[var_idx] = pheno_idx
                pos_pheno_desc[var_idx] = pheno_cur
                # pos_pheno_desc.append(self.pheno_descs[pheno_idx])

                # pos_phenotypes.append(pheno_idx)
                ref_aa.append(record['REF_AA'])
                alt_aa.append(record['ALT_AA'])
                prot_var_pos.append(record[pos_col])
                prot_var_idx.append(var_idx)

                if self.label_col in record and record[self.label_col] == 1:
                    is_pathogenic[var_idx] = 1
            
            seq_input_ids = self.encode_protein_seq(seq)
            cur_protein = {'id': uprot,
                           'seq': seq,
                           'seq_length': len(seq),
                           'desc': prot_desc,
                           'seq_input_ids': seq_input_ids,
                           'var_pos': prot_var_pos,
                           'var_idx': prot_var_idx,
                           'variant_mask': is_var,
                           'ref_aa': self.protein_tokenizer.convert_tokens_to_ids(ref_aa),
                           'alt_aa': self.protein_tokenizer.convert_tokens_to_ids(alt_aa),
                           'is_pathogenic': is_pathogenic,
                           'pos_pheno_idx': pos_pheno_idx,  # array with shape (seq_length, )
                           'pos_pheno_desc': pos_pheno_desc,
                           'neg_pheno_idx': neg_pheno_idx,  # array with shape (seq_length, )
                           'neg_pheno_desc': neg_pheno_desc,
                           'label': labels
                           }
            if not seq_only:
                cur_protein['desc_input_ids'] = self.text_tokenizer.encode(prot_desc)
                # if self.split != 'test':
                #     pos_pheno_input_ids = self.text_tokenizer(pos_pheno_desc)['input_ids']  # nested list: length=seq_length
                #     neg_pheno_input_ids = self.text_tokenizer(neg_pheno_desc)['input_ids']
                #     cur_protein['max_pheno_length'] = max([max(len(pos_pheno_input_ids[vi]), len(neg_pheno_input_ids[vi])) for vi in prot_var_idx])
                #     cur_protein['pos_pheno_input_ids'] = pos_pheno_input_ids
                #     cur_protein['neg_pheno_input_ids'] = neg_pheno_input_ids

            self.protein_variants.append(cur_protein)


    def encode_protein_seq(self, seq):
        aa_list = list(seq)
        if self.max_protein_seq_length is not None:
            aa_list = aa_list[:self.max_protein_seq_length]

        prot_input_ids = self.protein_tokenizer.encode(aa_list, padding='max_length',
                                                       max_length=self.max_protein_seq_length, 
                                                       is_split_into_words=True)
        
        return prot_input_ids


    def __getitem__(self, index):
        return self.protein_variants[index]

    def __len__(self):

        return len(self.protein_variants)
    
    @property
    def n_protein(self):
        assert len(self.protein_ids) == len(self.protein_variants)
        return len(self.protein_ids)
    
    def get_protein_list(self):
        return self.protein_ids

    def get_protein_data(self):
        return self.protein_variants


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

    protein_tokenizer: PreTrainedTokenizerBase
    text_tokenizer: PreTrainedTokenizerBase = None
    mlm: bool = False  # for masked language model (not implemented yet)
    mlm_probability: float = 0.15
    same_length: bool = False
    use_desc: bool = False
    label_pad_idx: int = -100
    pheno_descs: List = None
    n_phenotypes: int = None  # total number of phenotype terms

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
        # batch = {'input_ids': protein_seq_collate_fn(examples, self.tokenizer, self.same_length)}
        # if not self.use_desc:
        #     batch = protein_seq_collate_fn(examples, self.protein_tokenizer, self.same_length)
        #     batch['attention_mask'] = (batch['input_ids'] != self.protein_tokenizer.pad_token_id).long()
        #     batch['token_type_ids'] = torch.zeros_like(batch['input_ids'], dtype=torch.long)
        # else:
        batch = protein_variant_collate_fn(batch_data_raw, self.protein_tokenizer, self.text_tokenizer, self.same_length, self.use_desc, label_pad_idx=self.label_pad_idx)   

        # seq_token_mask = batch['seq_input_feat']['attention_mask'].logical_not()
        # only pahtogenic positions are eligible for phenotype prediction
        # mlm_token_mask = seq_token_mask.logical_or(batch['pathogenic_mask'].logical_not())
        if not self.n_phenotypes:
            self.n_phenotypes = len(self.pheno_descs)
        
        seq_lengths = batch['seq_length']
        batch_size = len(seq_lengths)

        # TODO: decide to use variant_mask or pathogenic_mask
        pheno_pos_masked_idx, pheno_mlm_label, variant_mask, indices_replaced, indices_random = \
            self.mask_variants(batch['phenotype']['positive']['pheno_idx'], self.text_tokenizer, variant_mask=batch['variant_mask'], vocab_size=self.n_phenotypes)
        
        # mlm_token_mask = batch['pathogenic_mask'].logical_not()
        # pheno_masked_idx, pheno_mlm_label, pheno_masked_indices, indices_replaced, indices_random = \
        #     self.mask_tokens(batch['phenotype']['pos_pheno_idx'], self.text_tokenizer, special_tokens_mask=mlm_token_mask, vocab_size=self.n_phenotypes)
        pos_pheno_desc_lst = [' '.join(s) for s in batch['phenotype']['positive']['pheno_desc']]
        neg_pheno_desc_lst = [' '.join(s) for s in batch['phenotype']['negative']['pheno_desc']]

        pos_pheno_tokenized = self.text_tokenizer(pos_pheno_desc_lst, padding=True, return_tensors='pt')
        neg_pheno_tokenized = self.text_tokenizer(neg_pheno_desc_lst, padding=True, return_tensors='pt')

        pos_mlm_pheno_descs = copy.deepcopy(batch['phenotype']['positive']['pheno_desc'])
        
        for b in range(batch_size):
            idx_replace_cur = torch.where(indices_replaced[b])[0].tolist()
            idx_random_cur = torch.where(indices_random[b])[0].tolist()

            for idx in idx_random_cur:
                pos_mlm_pheno_descs[b][idx] = self.pheno_descs[pheno_pos_masked_idx[b, idx].item()]
            
            for idx in idx_replace_cur:
                pos_mlm_pheno_descs[b][idx] = self.text_tokenizer.mask_token

        pos_pheno_mlm_tokenized = self.text_tokenizer([' '.join(pheno_seq) for pheno_seq in pos_mlm_pheno_descs], padding=True,
                                                      return_tensors='pt')
        
        batch['phenotype']['positive'].update({
            'input_ids': pos_pheno_tokenized['input_ids'],
            'attention_mask': pos_pheno_tokenized['attention_mask'],
            'mlm_input_ids': pos_pheno_mlm_tokenized['input_ids'],
            'mlm_attention_mask': pos_pheno_mlm_tokenized['attention_mask'],
            'mlm_label': pheno_mlm_label
        })

        batch['phenotype']['negative'].update({
            'input_ids': neg_pheno_tokenized['input_ids'],
            'attention_mask': neg_pheno_tokenized['attention_mask'],
        })

        # random_phenos = [self.pheno_descs[idx.item()] for idx in pheno_pos_masked_idx[indices_random]]  # MLM for positive label
        # pos_pheno_mlm_input = batch['phenotype']['positive']['input_ids'].clone()
        # pos_pheno_mlm_attn_mask = batch['phenotype']['positive']['attention_mask'].clone()
        
        # if random_phenos:
        #     # for b, mlm_pheno_cur in enumerate(pos_mlm_pheno_descs):
            
        #     random_pheno_tokenized = self.text_tokenizer(random_phenos, padding='max_length', truncation=True,
        #                                                  max_length=batch['phenotype']['max_length'], return_tensors='pt')
        #     pos_pheno_mlm_input[indices_random] = random_pheno_tokenized['input_ids']
        #     pos_pheno_mlm_attn_mask[indices_random] = random_pheno_tokenized['attention_mask']
            
        # pos_pheno_mlm_input[indices_replaced] = self.text_tokenizer.encode(self.text_tokenizer.mask_token, padding='max_length', truncation=True,
        #                                                                    max_length=batch['phenotype']['max_length'], return_tensors='pt')
        # pos_pheno_mlm_attn_mask[indices_replaced] = (pos_pheno_mlm_input[indices_replaced] != self.text_tokenizer.pad_token_id).long()

        # batch['phenotype']['positive']['mlm_input_ids'] = pos_pheno_mlm_tokenized['input_ids']  # masked input ids for mlm
        # batch['phenotype']['positive']['mlm_attention_mask'] = pos_pheno_mlm_tokenized['attention_mask']
        # batch['phenotype']['positive']['mlm_label'] = pheno_mlm_label

        # batch['phenotype']['pos_input_ids'] = pos_pheno_masked

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


def protein_variant_collate_fn(
    batch_data_raw: List[Dict],
    protein_tokenizer: PreTrainedTokenizerBase,
    text_tokenizer: PreTrainedTokenizerBase,
    same_length: bool = False,
    use_desc: bool = True,
    label_pad_idx: int = -100
):   
    """
    Collate function for protein using both sequence and description text
    """
    batch_prot_ids = [elem['id'] for elem in batch_data_raw]
    batch_size = len(batch_prot_ids)
    # protein sequence
    prot_lengths = [elem['seq_length'] for elem in batch_data_raw]
    max_seq_length = max(prot_lengths)

    batch_seq_input_ids = [torch.tensor(elem['seq_input_ids'], dtype=torch.long) for elem in batch_data_raw] # list -> tensor
    batch_var_mask = [torch.LongTensor(elem['variant_mask']) for elem in batch_data_raw]
    batch_patho_mask = [torch.LongTensor(elem['is_pathogenic']) for elem in batch_data_raw]
    batch_desc_input_ids = [torch.LongTensor(elem['desc_input_ids']) for elem in batch_data_raw]
    max_desc_length = max([len(elem['desc_input_ids']) for elem in batch_data_raw])
    # TODO: pad phenotype input ids to same length: (batch_size, max_seq_length, max_pheno_length)
    # max_pheno_length = max([elem['max_pheno_length'] for elem in batch_data_raw])
           
    if same_length:
        batch_seq_input_ids = torch.stack(batch_seq_input_ids, dim=0)
        batch_var_mask = torch.stack(batch_var_mask, dim=0)
        batch_patho_mask = torch.stack(batch_patho_mask, dim=0)
    # protein sequence padding, pad id = 0
    else:  # after padding: (batch_size, max_seq_length)
        batch_seq_input_ids = prep_padded_input(batch_seq_input_ids, max_seq_length, 
                                                padding_side=protein_tokenizer.padding_side, 
                                                init_value=protein_tokenizer.pad_token_id)
        batch_var_mask = prep_padded_input(batch_var_mask, max_seq_length, 
                                           padding_side=protein_tokenizer.padding_side, 
                                           init_value=0)
        batch_patho_mask = prep_padded_input(batch_patho_mask, max_seq_length, 
                                             padding_side=protein_tokenizer.padding_side, 
                                             init_value=0)
        
    
    pos_pheno_input_all = []
    neg_pheno_input_all = []
    pos_pheno_idx_all = [torch.LongTensor(elem['pos_pheno_idx']) for elem in batch_data_raw]
    neg_pheno_idx_all = [torch.LongTensor(elem['neg_pheno_idx']) for elem in batch_data_raw]
    pos_attention_mask_all = []
    neg_attention_mask_all = []
    # for b, prot_data in enumerate(batch_data_raw):
    #     # prot_var_idx = torch.where(batch_var_mask[b])[0].tolist()

    #     pos_input_padded = pad_phenotype_input(prot_data['pos_pheno_input_ids'],  # (max_seq_length, max_pheno_length)
    #                                         #    prot_var_idx,
    #                                            max_pheno_length,
    #                                            max_seq_length,
    #                                            padding_side=text_tokenizer.padding_side,
    #                                            fill_val=text_tokenizer.pad_token_id)
    #     pos_attention_mask = (pos_input_padded != text_tokenizer.pad_token_id).long()

    #     neg_input_padded = pad_phenotype_input(prot_data['neg_pheno_input_ids'],
    #                                         #    prot_var_idx,
    #                                            max_pheno_length,
    #                                            max_seq_length,
    #                                            padding_side=text_tokenizer.padding_side,
    #                                            fill_val=text_tokenizer.pad_token_id)
    #     neg_attention_mask = (neg_input_padded != text_tokenizer.pad_token_id).long()
        
    #     pos_pheno_input_all.append(pos_input_padded)
    #     neg_pheno_input_all.append(neg_input_padded)
    #     pos_attention_mask_all.append(pos_attention_mask)
    #     neg_attention_mask_all.append(neg_attention_mask)
        # pos_pheno_idx_all.append(pos_id_padded)
        # neg_pheno_idx_all.append(neg_id_padded)
    
    # batch_pos_input_ids = torch.stack(pos_pheno_input_all)  # (batch_size, max_seq_length, max_pheno_length)
    # batch_neg_input_ids = torch.stack(neg_pheno_input_all)
    # batch_pos_attention_mask = torch.stack(pos_attention_mask_all)
    # batch_neg_attention_mask = torch.stack(neg_attention_mask_all)
    
    batch_pos_pheno_idx = prep_padded_input(pos_pheno_idx_all, max_seq_length,
                                      padding_side=protein_tokenizer.padding_side,
                                      init_value=text_tokenizer.pad_token_id)
    
    batch_neg_pheno_idx = prep_padded_input(neg_pheno_idx_all, max_seq_length,
                                        padding_side=protein_tokenizer.padding_side,
                                        init_value=text_tokenizer.pad_token_id)
    # batch_pos_pheno_idx = torch.stack(pos_pheno_idx_all)
    # batch_neg_pheno_idx = torch.stack(neg_pheno_idx_all)
    
    batch_seq_attention_mask = (batch_seq_input_ids != protein_tokenizer.pad_token_id).long()
    batch_seq_token_type_ids = torch.zeros_like(batch_seq_input_ids, dtype=torch.long)

    # batch_desc_input_ids = torch.tensor([elem['desc_input_ids'] for elem in batch_data_raw], dtype=torch.long)
    batch_label = torch.full_like(batch_patho_mask, fill_value=label_pad_idx)
    batch_label[batch_var_mask.bool()] = 0  # benign
    batch_label[batch_patho_mask.bool()] = 1  # pathogenic

    if use_desc:
        batch_desc_input_ids = prep_padded_input(batch_desc_input_ids, max_desc_length,  # (batch_size, max_desc_length)
                                                 padding_side=text_tokenizer.padding_side, 
                                                 init_value=text_tokenizer.pad_token_id)
        batch_desc_attention_mask = (batch_desc_input_ids != text_tokenizer.pad_token_id).long()
        batch_desc_token_type_ids = torch.zeros_like(batch_desc_input_ids, dtype=torch.long)
        
    return {
        'id': batch_prot_ids,
        'seq_length': prot_lengths, 
        'variant_mask': batch_var_mask,  # true for variant (batch_size, max_length)
        'pathogenic_mask': batch_patho_mask,  # true for pathogenic (batch_size, max_length)
        'label': batch_label,  # 1: pathogenic, 0: benign, -100: non-variant spots
        'var_pos': [elem['var_pos'] for elem in batch_data_raw],
        'var_idx': [elem['var_idx'] for elem in batch_data_raw],
        'ref_aa': [elem['ref_aa'] for elem in batch_data_raw],
        'alt_aa': [elem['alt_aa'] for elem in batch_data_raw],
        'seq_input_feat': {
            'input_ids': batch_seq_input_ids,  # batch_size x max_length
            'attention_mask': batch_seq_attention_mask,  # 1 if token should be attended to
            'token_type_ids': batch_seq_token_type_ids,
        },
        'desc_input_feat': {
            'input_ids': batch_desc_input_ids,
            'attention_mask': batch_desc_attention_mask,
            'token_type_ids': batch_desc_token_type_ids
        },
        'phenotype':
        {
            # 'max_length': max_pheno_length,
            'positive': {
                # 'pheno_idx': [elem['pos_pheno_idx'] for elem in batch_data_raw]
                # 'input_ids': batch_pos_input_ids,  # (batch_size, max_seq_length, max_pheno_length)
                # 'attention_mask': batch_pos_attention_mask,  # (batch_size, max_seq_length, max_pheno_length)
                'pheno_idx': batch_pos_pheno_idx,  # Tensor
                'pheno_desc': [elem['pos_pheno_desc'] for elem in batch_data_raw]
            },
            'negative': {
                # 'pheno_idx': [elem['neg_pheno_idx'] for elem in batch_data_raw]
                # 'input_ids': batch_neg_input_ids,
                # 'attention_mask': batch_neg_attention_mask,
                'pheno_idx': batch_neg_pheno_idx,
                'pheno_desc': [elem['neg_pheno_desc'] for elem in batch_data_raw]
            }
            # 'pos_input_ids': batch_pos_input_ids,  # phenotype input ids (batch_size, max_seq_length, max_pheno_length)
            # 'pos_pheno_idx': batch_pos_pheno_idx,
            # 'neg_input_ids': batch_neg_input_ids,
            # 'neg_pheno_idx': batch_neg_pheno_idx,
        }
    }
    

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

        if padding_side == 'right':
            x_pad[i, :cur_length] = x_cur
        else:
            x_pad[i, -cur_length:] = x_cur
    
    return x_pad
            

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

