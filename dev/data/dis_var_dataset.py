import os
from pathlib import Path
import json
import copy
import numpy as np
import dataclasses
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import tqdm
import random
import networkx as nx
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import dgl

from transformers import BertModel, BertTokenizer
from transformers import PreTrainedTokenizerBase
import pandas as pd

from dev.preprocess.utils import fetch_prot_seq
from dev.utils import get_memory_usage


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


class ProteinVariantSeqDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        seq_dict: dict,
        variant_file: str = None,
        protein_tokenizer: PreTrainedTokenizerBase = None,
        max_protein_seq_length: int = None,
        phenotype_vocab: List = None,
        pos_offset = 1,
        pid_col='UniProt',
        pos_col='Protein_position',
        label_col='label',
        pheno_col='phenotype',
        emb_idx_dict: Dict = None,
        **kwargs
    ):
        super(ProteinVariantSeqDataset, self).__init__()
        self.data_root = Path(data_dir)
        self.seq_dict = seq_dict
        self.variant_data = []  # list of variant dict
        self.protein_ids = []
        self.pos_offset = pos_offset  # 1 for 1-based position
        self.protein_tokenizer = protein_tokenizer
        self.max_protein_seq_length = max_protein_seq_length
        self.pheno_descs = phenotype_vocab
        self.unk_pheno_mask = []
        self.label = []
        self.label_col = label_col
        self.pheno_col = pheno_col
        self.emb_idx_dict = emb_idx_dict
        
        df_var = pd.read_csv(self.data_root / variant_file)
        self._load_variant_data(df_var, pid_col, pos_col)

    
    def _load_variant_data(self, df_var, pid_col='UniProt', pos_col='Protein_position'):
        prots_all = df_var[pid_col].drop_duplicates().tolist()
        self.protein_ids = prots_all

        self.has_phenotype_label = self.pheno_col in df_var.columns

        cur_indice = 0
        for i, uprot in enumerate(prots_all):
            cur_protein = dict()
            if uprot not in self.seq_dict:
                seq = fetch_prot_seq(uprot, seq_only=True)
                self.seq_dict[uprot] = seq
            else:
                seq = self.seq_dict[uprot]
            
            df_prot = df_var[df_var[pid_col] == uprot]
            if len(df_prot) == 0:
                continue
            # var_names = []
            for j, record in df_prot.iterrows():
                # pheno_cur = record[self.pheno_col]
                if record[pos_col] >= self.max_protein_seq_length:  # skip out-of-bound variants for now
                    continue
                if record[pos_col] > len(seq):
                    logging.warning('Invalid variant position {} for protein {} with length {}'.format(record[pos_col], uprot, len(seq)))
                    continue
                var_pos_idx = record[pos_col] - self.pos_offset
                ref_aa_cur = record['REF_AA']
                alt_aa_cur = record['ALT_AA']
                cur_var_name = '{}_{}_{}/{}'.format(uprot, record[pos_col], record['REF_AA'], record['ALT_AA'])
                pheno_cur = record[self.pheno_col]
                if self.emb_idx_dict:
                    try:  # limited to pre-computed disease embeddings
                        pheno_idx = self.emb_idx_dict[pheno_cur]
                    except KeyError:
                        continue
                else:
                    pheno_idx = self.pheno_descs.index(pheno_cur)
                cur_variant = {'id': cur_var_name,
                               'uprot': uprot,
                               'indice': cur_indice,
                               'var_idx': var_pos_idx,
                               'var_pos': record[pos_col],
                            #    'label': record[self.label_col],
                               'ref_aa': self.protein_tokenizer.convert_tokens_to_ids(ref_aa_cur),
                               'alt_aa': self.protein_tokenizer.convert_tokens_to_ids(alt_aa_cur),
                               'pos_pheno_desc': pheno_cur,
                               'pos_pheno_idx': pheno_idx
                               }
                self.variant_data.append(cur_variant)
                

    def __getitem__(self, index):
        # return self.protein_variants[index]
        return self.variant_data[index]

    def __len__(self):
        return len(self.variant_data)

    def get_all_protein_seq(self):
        return self.seq_dict


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
        use_pheno_desc: bool = False,
        pheno_desc_dict: Dict[str, str] = None,
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
        mode='train',  # choose from {'train', 'eval'}
        binary_inference_only=False,  # infer pathogenic vs benign only
        update_var_cache=True,  # if True, current variant (disease) will be visible to other variants (set to False at inference)
        use_struct_neighbor=False,
        max_struct_dist=20,
        af_graph_dir='',
        pdb_graph_dir='',
        use_struct_vocab=False,
        comb_seq_dict=None,  # combined sequence (residue+struct from foldseek)
        access_to_context=False,
        exclude_prots=None,
        use_alphamissense=False,
        afmis_root=None,
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
        self.mode = mode
        self.patho_only = patho_only
        self.binary_inference_only = binary_inference_only
        self.update_var_cache = update_var_cache
        self.has_phenotype_label = False
        self.use_struct_neighbor = use_struct_neighbor
        self.max_struct_dist = max_struct_dist
        if self.use_struct_neighbor:
            self.af_graph_root = Path(af_graph_dir)
            self.pdb_graph_root = Path(pdb_graph_dir)
        self.num_struct_neighbors = []
        self.use_struct_vocab = use_struct_vocab
        self.comb_seq_dict = comb_seq_dict
        self.access_to_context = access_to_context

        self.use_alphamissense = use_alphamissense
        self.afmis_root = afmis_root
        self.afmis_prots = []
        if self.use_alphamissense and not self.afmis_root.exists():
            self.use_alphamissense = False
        if self.use_alphamissense:
            for fpath in self.afmis_root.glob('*json'):
                self.afmis_prots.append(fpath.name.split('_')[0])

        if not variant_file:
            variant_file = split + '.csv'
        
        df_var = pd.read_csv(self.data_root / variant_file).drop_duplicates([pid_col, pos_col, label_col, 'REF_AA', 'ALT_AA']).reset_index(drop=True)
        if exclude_prots:
            df_var = df_var[~df_var[pid_col].isin(exclude_prots)].reset_index(drop=True)
        self._load_variant_data(df_var, pid_col, pos_col)
        if self.use_struct_neighbor:
            self.build_var_context_graph()
    

    def _load_variant_data(self, df_var, pid_col='UniProt', pos_col='Protein_position'):

        # assert self.var_input_file != None
        # df_var = pd.read_csv(self.var_input_file)
        seq_only = not self.use_protein_desc
        prots_all = df_var[pid_col].drop_duplicates().tolist()
        self.protein_ids = prots_all

        self.has_phenotype_label = self.pheno_col in df_var.columns

        cur_indice = 0
        for i, uprot in enumerate(prots_all):
            cur_protein = dict()
            # cur_protein['id'] = uprot
            cache_context_var_idx = []
            new_context_var_idx = []
            if self.prot_var_cache:
                cur_protein = self.prot_var_cache.get(uprot, None)
            if cur_protein:  # protein found in cache
                seq = cur_protein['seq']
                prot_desc = cur_protein['prot_desc']
                cache_context_var_idx = cur_protein['context_var_idx']
                pos_pheno_desc = cur_protein['var_pheno_descs']
            else:
                # Load sequence (and functional description)
                if uprot not in self.seq_dict:
                    prot_info = fetch_prot_seq(uprot, seq_only=seq_only)
                    if seq_only:
                        seq = prot_info
                    else:
                        try:
                            seq = prot_info['sequence']['value']
                            prot_desc_dict = prot_info['proteinDescription']
                            if 'recommendedName' in prot_desc_dict:
                                prot_name = prot_desc_dict['recommendedName']['fullName']['value']
                            elif 'submissionNames' in prot_desc_dict:
                                prot_name = prot_desc_dict['submissionNames'][0]['fullName']['value']
                            else:
                                prot_name = self.text_tokenizer.unk_token

                            if uprot not in self.protein_info_dict:
                                self.protein_info_dict[uprot] = prot_name
                        except KeyError:
                            logging.warning(f'{uprot} not found')
                            continue
                        
                        # cur_protein['desc'] = prot_desc
                    
                    # cur_protein['seq'] = seq
                    self.seq_dict[uprot] = seq
                    
                else:
                    seq = self.seq_dict[uprot]
                    if self.use_protein_desc:
                        prot_desc = self.protein_info_dict.get(uprot, self.text_tokenizer.unk_token)
            
            if self.use_struct_vocab and self.comb_seq_dict:
                try:
                    comb_seq = self.comb_seq_dict[uprot]
                except KeyError:
                    # logging.warning(f'No structure-aware sequence for {uprot}')
                    comb_seq = ''.join([f'{s}#' for s in seq])  # unknown structure
                    self.comb_seq_dict[uprot] = comb_seq
                struct_seq = comb_seq[1::2]
                try:
                    assert len(seq) == len(struct_seq)
                except AssertionError:
                    logging.warning(f'Inconsistent sequence lengths for structure & primary sequence ({uprot})')
                    continue
            
            prot_afmis_dict = dict()
            if self.use_alphamissense:
                if uprot in self.afmis_prots:
                    with open(self.afmis_root / f'{uprot}_sub_all.json') as f:
                        prot_afmis_dict = json.load(f)
                
            if self.use_struct_neighbor:
                pdb_graph_exists = (self.pdb_graph_root / f'{uprot}.graphml.gz').exists()
                af_graph_exists = (self.af_graph_root / f'{uprot}.graphml.gz').exists()
                if pdb_graph_exists:
                    g_pdb = nx.read_graphml(self.pdb_graph_root / f'{uprot}.graphml.gz')
                else:
                    g_pdb = nx.Graph()
                if af_graph_exists:
                    g_af = nx.read_graphml(self.af_graph_root / f'{uprot}.graphml.gz')
                else:
                    g_af = nx.Graph()
                g_prot = nx.compose(g_af, g_pdb)  # structure graph for protein (edge feature: distance)
                node_id_map = {nid: int(nid.split('_')[1]) - 1 for nid in g_prot.nodes()}
                g_prot = nx.relabel_nodes(g_prot, node_id_map)  # node ID: "uprot_resid" --> resid (int, start from 0)
                if g_prot.number_of_nodes() < len(seq):  # add missing nodes
                    missing_res = set(range(len(seq))) - set(g_prot.nodes())
                    g_prot.add_nodes_from(missing_res)
                try:
                    assert len(seq) == g_prot.number_of_nodes()
                except AssertionError:
                    logging.warning(f'Inconsistent lengths for structure & primary sequence ({uprot})')
                    continue
                g_prot_csr = nx.to_scipy_sparse_array(g_prot, weight='distance')
                # g_prot = g_prot.to_directed()

            df_prot = df_var[df_var[pid_col] == uprot]
            if self.patho_only:
                df_prot = df_prot[df_prot[self.label_col == 1]]
            if len(df_prot) == 0:
                continue
            is_var = np.zeros(len(seq))
            pos_pheno_idx = np.full(len(seq), fill_value=self.pheno_descs.index(self.text_tokenizer.unk_token))  # positive phenotypes
            # neg_pheno_idx = np.full(len(seq), fill_value=self.text_tokenizer.pad_token_id)  # negative phenotypes
            # pos_pheno_desc = np.full(len(seq), fill_value=self.text_tokenizer.pad_token)
            pos_pheno_desc = [self.text_tokenizer.unk_token] * len(seq)
            # neg_pheno_desc = [self.text_tokenizer.pad_token] * len(seq)
            # neg_pheno_desc = np.full(len(seq), fill_value=self.text_tokenizer.pad_token)
            prot_var_pos = []
            
            ref_aa = []
            alt_aa = []
            var_names = []
            labels = []
            
            for j, record in df_prot.iterrows():
                # pheno_cur = record[self.pheno_col]
                # if record[pos_col] >= self.max_protein_seq_length:  # skip out-of-bound variants for now
                #     continue
                if record[pos_col] > len(seq):
                    logging.warning('Invalid variant position {} for protein {} with length {}'.format(record[pos_col], uprot, len(seq)))
                    continue
                sample_mask = np.zeros(len(self.pheno_descs), dtype=bool)
                var_pos_idx = record[pos_col] - self.pos_offset
                is_var[var_pos_idx] = 1
                if self.label_col in record:
                    labels.append(record[self.label_col])
                
                pos_pheno_is_known = False
                infer_phenotype = False
                if self.binary_inference_only:
                    # pheno_idx = self.text_tokenizer.unk_token_id
                    pheno_cur = self.text_tokenizer.unk_token
                    pheno_idx = self.pheno_descs.index(pheno_cur)

                else:
                    if self.has_phenotype_label:
                        pheno_cur = record[self.pheno_col]
                    else:  # no phenotype information (presumably in test/inference only)
                        pheno_cur = np.nan
                    
                    if self.mode == 'train':
                        if isinstance(pheno_cur, str) and record[self.label_col] == 1:  # pathogenic AND phenotype information available (in train & eval)
                            try:
                                pheno_idx = self.pheno_descs.index(pheno_cur)
                                sample_mask[pheno_idx] = True
                                infer_phenotype = True
                                pos_pheno_is_known = True
                            except ValueError:
                                # pheno_idx = self.text_tokenizer.unk_token_id
                                # pheno_cur = self.text_tokenizer.unk_token
                                pheno_idx = self.pheno_descs.index(self.text_tokenizer.unk_token)
                            
                            pos_pheno_idx[var_pos_idx] = pheno_idx
                            pos_pheno_desc[var_pos_idx] = pheno_cur
                            # if pos_pheno_desc[var_idx] != self.text_tokenizer.pad_token:
                            #     pos_pheno_update = set(pos_pheno_desc[var_idx].split(';')) - {self.text_tokenizer.pad_token, self.text_tokenizer.unk_token}
                            #     pos_pheno_desc[var_idx] = self.text_tokenizer.sep_token.join([pos_pheno_desc[var_idx]] + pheno_cur) 

                            # if self.split in ['train', 'val']:  # modified: now variants in testing set is not visible to each other
                            #     cache_context_var_idx.append(var_idx)  # context information for phenotype inference
                            new_context_var_idx.append(var_pos_idx)
                        else:
                            # pheno_idx = self.text_tokenizer.unk_token_id
                            # pheno_cur = self.text_tokenizer.unk_token
                            pheno_idx = self.pheno_descs.index(self.text_tokenizer.unk_token)
                    # elif record[self.label_col] == 0 and self.mode == 'eval':
                    else:  # in inference, always infer phenotype but positive label not always available
                        infer_phenotype = True
                        try:
                            pheno_idx = self.pheno_descs.index(pheno_cur)
                            sample_mask[pheno_idx] = True
                            pos_pheno_is_known = True
                        except ValueError:
                            # pheno_idx = self.text_tokenizer.unk_token_id
                            # pheno_cur = self.text_tokenizer.unk_token
                            if not isinstance(pheno_cur, str):  # NA
                                # infer_phenotype = False
                                pheno_cur = self.text_tokenizer.unk_token
                            pheno_idx = self.pheno_descs.index(self.text_tokenizer.unk_token)
                        
                        pos_pheno_idx[var_pos_idx] = pheno_idx
                        pos_pheno_desc[var_pos_idx] = pheno_cur
                        if record[self.label_col] == 1:
                            new_context_var_idx.append(var_pos_idx)
                        # if self.split in ['train', 'val'] and record[self.label_col] == 1:  # modified: now variants in testing set is not visible to each other
                        #     cache_context_var_idx.append(var_idx)  # context information for phenotype inference
                        
                    # else:
                    #     pheno_idx = self.text_tokenizer.unk_token_id
                    #     pheno_cur = self.text_tokenizer.unk_token
                    #     pos_pheno_available = False
                        # neg_sample_idx = self.text_tokenizer.unk_token_id
                        # neg_pheno_cur = self.text_tokenizer.unk_token
                ref_aa_cur = record['REF_AA']
                alt_aa_cur = record['ALT_AA']

                if self.use_struct_vocab:
                    ref_aa_cur = record['REF_AA'] + struct_seq[var_pos_idx]
                    alt_aa_cur = record['ALT_AA'] + '#'  # mask structure for variant
                ref_aa.append(ref_aa_cur)
                alt_aa.append(alt_aa_cur)
                cur_var_name = '{}_{}_{}/{}'.format(uprot, record[pos_col], record['REF_AA'], record['ALT_AA'])
                var_names.append(cur_var_name)
                # var_names.append(''.join([record['REF_AA'], str(record[pos_col]), record['ALT_AA']]))
                prot_var_pos.append(record[pos_col])
                # prot_var_idx_target.append(var_idx)

                cur_variant = {'id': cur_var_name,
                               'uprot': uprot,
                               'indice': cur_indice,
                               'var_idx': var_pos_idx,
                               'var_pos': record[pos_col],
                               'label': record[self.label_col],
                            #    'ref_aa': self.protein_tokenizer.convert_tokens_to_ids(ref_aa_cur),
                            #    'alt_aa': self.protein_tokenizer.convert_tokens_to_ids(alt_aa_cur),
                               'ref_aa': ref_aa_cur,
                               'alt_aa': alt_aa_cur,
                               'pos_pheno_desc': pheno_cur,
                               'pos_pheno_idx': pheno_idx,
                            #    'neg_pheno_desc': neg_pheno_cur,
                            #    'neg_pheno_idx': neg_sample_idx,
                               'infer_phenotype': infer_phenotype,
                               'pos_pheno_is_known': pos_pheno_is_known
                               }
                if self.use_alphamissense:
                    cur_variant['afmis_score'] = prot_afmis_dict.get(cur_var_name, -1)
                    cur_variant['afmis_mask'] = int(cur_variant['afmis_score'] == -1)
                # if self.use_struct_neighbor:
                #     nx.set_node_attributes(g_prot, values={var_pos_idx: {'pheno_idx': pheno_idx}})
                #     var_struct_graph = g_prot.edge_subgraph(g_prot.in_edges(var_pos_idx))
                #     nx.set_node_attributes(var_struct_graph, {var_pos_idx: {'pheno_idx': self.text_tokenizer.mask_token_id}})
                #     cur_variant['var_struct_graph'] = dgl.from_networkx(var_struct_graph, node_attrs=['pheno_idx'], edge_attrs=['distance'])
                self.variant_data.append(cur_variant)
                cur_indice += 1
            # seq_input_ids = self.encode_protein_seq(seq)
            cur_protein = {'seq': seq,
                           'seq_length': len(seq),
                           'prot_desc': prot_desc,
                        #    'context_var_idx': cache_context_var_idx,  
                           'var_pheno_descs': pos_pheno_desc,
                           'var_pheno_idx': pos_pheno_idx
                           }
            # if self.split == 'train':  # modified: variants in val and test not visible to each other for phenotype inference
            if self.access_to_context:
                cur_protein['context_var_idx'] = cache_context_var_idx + new_context_var_idx
            else:
                cur_protein['context_var_idx'] = cache_context_var_idx

            if self.use_struct_vocab:
                cur_protein['comb_seq'] = comb_seq
            
            if self.use_struct_neighbor:
                # nx.set_node_attributes(g_prot, values=dict(zip(range(len(seq)), pos_pheno_idx)), name='pheno_idx')
                # nx.set_node_attributes(g_prot, values=dict(zip(range(len(seq)), pos_pheno_desc)), name='pheno_descs')
                cur_protein['struct_graph'] = g_prot_csr

            self.protein_variants[uprot] = cur_protein
            if self.update_var_cache:
                self.prot_var_cache[uprot] = {'seq': seq,
                                              'seq_length': len(seq),
                                              'prot_desc': prot_desc,
                                              'context_var_idx': cache_context_var_idx + new_context_var_idx,  
                                              'var_pheno_descs': pos_pheno_desc
                                              }
                # self.prot_var_cache.update(self.protein_variants)
    
    def build_var_context_graph(self):
        logging.info('Extract structural context...')
        for i, cur_variant in enumerate(self.variant_data):
            if not cur_variant['infer_phenotype']:
                continue
            uprot = cur_variant['uprot']
            var_idx = cur_variant['var_idx']
            protein_info = self.protein_variants[uprot]
            prot_context_var_idx = protein_info['context_var_idx']
            struct_g = nx.from_scipy_sparse_array(protein_info['struct_graph'], edge_attribute='distance')
            var_graph_edges, context_nodes = extract_context_graph(var_idx, prot_context_var_idx, struct_g, dist_cutoff=self.max_struct_dist)
            cur_variant['var_graph_edges'] = var_graph_edges
            cur_variant['num_struct_neighbors'] = len(context_nodes)
            self.num_struct_neighbors.append(len(context_nodes))
            # struct_g.clear()
            del struct_g
            # self.num_struct_neighbors.append(var_context_graph.number_of_nodes() - 1)
            # cur_variant['var_struct_graph'] = var_context_graph

    def average_struct_neighbors(self):
        return np.mean(self.num_struct_neighbors)


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
    use_prot_desc: bool = False
    label_pad_idx: int = -100
    phenotype_vocab: List = None
    pheno_desc_dict: Dict[str, str] = None
    use_pheno_desc: bool = False  # use phenotype description or not
    half_window_size: int = 64
    truncate_protein: bool = True
    max_protein_length: int = None
    max_pheno_desc_length: int = 512
    mode: str = 'train'
    has_phenotype_label: bool = True
    use_struct_vocab: bool = False
    use_struct_neighbor: bool = False
    use_alphamissense: bool = False
    struct_radius_cutoff: float = 25
    include_unknown: bool = False
    context_agg_opt: str = 'concat'

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
                                           pheno_vocab=self.phenotype_vocab, use_prot_desc=self.use_prot_desc, 
                                           use_pheno_desc=self.use_pheno_desc, pheno_desc_dict=self.pheno_desc_dict, 
                                           truncate_protein=self.truncate_protein, max_protein_length=self.max_protein_length, max_pheno_desc_length=self.max_pheno_desc_length, 
                                           half_window_size=self.half_window_size, mode=self.mode, has_phenotype_label=self.has_phenotype_label,
                                           use_struct_vocab=self.use_struct_vocab, use_struct_neighbor=self.use_struct_neighbor, struct_radius=self.struct_radius_cutoff,
                                           use_alphamissense=self.use_alphamissense, include_unknown=self.include_unknown, context_agg_opt=self.context_agg_opt)

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


@dataclass
class ProteinVariantSeqCollator:
    """
    Seq-only variant data collator
    """
    protein_data: Dict
    protein_tokenizer: PreTrainedTokenizerBase
    protein_seq_dict: Dict

    def __call__(
        self,
        batch_data_raw: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        batch = variant_seq_collate_fn(batch_data_raw, self.protein_tokenizer, self.protein_seq_dict)

        return batch


def variant_seq_collate_fn(
    batch_data_raw: List[Dict], 
    protein_tokenizer: PreTrainedTokenizerBase,
    protein_seq_dict: Dict,
    max_protein_length: int = None
):
    # max_seq_length = max(prot_lengths)
    if not max_protein_length:
        max_protein_length = protein_tokenizer.model_max_length
    else:
        max_protein_length = min(max_protein_length, protein_tokenizer.model_max_length)

    batch_prot_ids = [elem['uprot'] for elem in batch_data_raw]
    prot_unique = list(set(batch_prot_ids))
    # protein sequence
    seq_lst = [protein_seq_dict[pid] for pid in prot_unique]
    prot_lengths = [len(seq) for seq in seq_lst]
    batch_seq_tokenized = protein_tokenizer(seq_lst, padding=True, truncation=True, return_tensors='pt', max_length=max_protein_length)
    batch_seq_input_lst = batch_seq_tokenized['input_ids'].tolist()
    
    var_idx_all = []
    var_seq_input_ids = []
    var_names_all = []
    prot_idx_all = []
    pos_pheno_name_all = []
    pos_pheno_idx_all = []

    for b, elem in enumerate(batch_data_raw):
        uprot = elem['uprot']
        prot_idx_cur = prot_unique.index(uprot)
        var_idx = elem['var_idx']
        ref_seq_input_ids = batch_seq_input_lst[prot_idx_cur]
        var_seq_input_cur = ref_seq_input_ids[:var_idx+1] + [elem['alt_aa']] + ref_seq_input_ids[var_idx+2:]

        var_seq_input_ids.append(var_seq_input_cur)
        var_idx_all.append(var_idx)
        var_names_all.append(elem['id'])
        prot_idx_all.append(prot_idx_cur)
        pos_pheno_name_all.append(elem['pos_pheno_desc'])
        pos_pheno_idx_all.append(elem['pos_pheno_idx'])
    
    variant_dict = {
            'indices': [elem['indice'] for elem in batch_data_raw], 
            'prot_idx': prot_idx_all,
            'var_pos': [elem['var_pos'] for elem in batch_data_raw],
            'var_idx': var_idx_all,
            'var_names': var_names_all,
            'var_seq_input_ids': torch.tensor(var_seq_input_ids),
            'var_seq_attention_mask': batch_seq_tokenized['attention_mask'][prot_idx_all],
            # 'pos_pheno_name': pos_pheno_name_all,
            'pos_pheno_name': pos_pheno_name_all,
            'pos_pheno_idx': torch.tensor(pos_pheno_idx_all),
            # 'pheno_var_names': pheno_var_names,
        }
    
    return variant_dict


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
            half_window_size: int = 64,
            max_num: int = None,
            mask_token: str = '[MASK]',
            unknown_token: str = '[UNK]',
            pheno_desc_dict: Dict[str, str] = None,
            use_pheno_desc: bool = False,
            include_unknown: bool = False
    ):
        start = max(var_idx - half_window_size, 0)
        end = min(var_idx + half_window_size, seq_length - 1)
        # target_idx = var_idx - start

        pheno_in_frame = []
        context_pheno_idx_list = []  # position index for contextual disease variants
        for i, idx in enumerate(sorted(set(context_var_idx + [var_idx]))):
            if idx == var_idx:
                pheno_in_frame.append(mask_token)
                context_pheno_idx_list.append(idx)
            elif idx in range(start, end+1):
                if include_unknown or pos_pheno_descs[idx] != unknown_token:
                    pheno_in_frame.append(pos_pheno_descs[idx])
                    context_pheno_idx_list.append(idx)
            if idx > end:
                break

        target_pheno_loc = pheno_in_frame.index(mask_token)
        if max_num:
            if len(pheno_in_frame) > max_num:
                if target_pheno_loc <= max_num - 1:
                    pheno_in_frame = pheno_in_frame[:max_num]  # right truncation
                    context_pheno_idx_list = context_pheno_idx_list[:max_num]
                else:
                    pheno_in_frame = pheno_in_frame[-max_num:]
                    context_pheno_idx_list = context_pheno_idx_list[-max_num:]
        
        # pheno_in_frame.pop(target_pheno_loc)
        pheno_desc_in_frame = []
        if use_pheno_desc and pheno_desc_dict:
            for name in pheno_in_frame:
                pheno_desc = '{name} {desc}'.format(name=name, desc=pheno_desc_dict.get(name, '')).strip()
                pheno_desc_in_frame.append(pheno_desc)
            
            return pheno_desc_in_frame, context_pheno_idx_list

        return pheno_in_frame, context_pheno_idx_list


def get_optimal_window(var_position, seq_len_raw, model_window):
    half_model_window = model_window // 2
    if seq_len_raw <= model_window:  # full sequence
        return [0, seq_len_raw]
    elif var_position < half_model_window:  # truncate tail
        return [0, model_window]
    elif var_position >= seq_len_raw - half_model_window:  # truncate head
        return [seq_len_raw - model_window, seq_len_raw]
    else:  # truncate both sides (centered at variant)
        return [max(0, var_position - half_model_window), min(seq_len_raw, var_position + half_model_window)]
    

def protein_variant_collate_fn(
    batch_data_raw: List[Dict],
    protein_tokenizer: PreTrainedTokenizerBase,
    text_tokenizer: PreTrainedTokenizerBase,
    protein_data: Dict,
    pheno_vocab: List[str],
    use_pheno_desc: bool = False,
    use_prot_desc: bool = True,
    half_window_size: int = 64,
    truncate_protein: bool = True,
    max_protein_length: int = None,
    max_context_phenos: int = None,
    max_pheno_desc_length: int = None,
    mode: str = 'train',  # {'train', 'eval'}
    pheno_desc_dict: Dict[str, str] = None,
    has_phenotype_label: bool = True,
    use_struct_vocab: bool = False,
    foldseek_vocab: str = "pynwrqhgdlvtmfsaeikc#",
    use_struct_neighbor: bool = False,
    struct_radius: float = 25,
    use_alphamissense: bool = False,
    include_unknown: bool = False,
    context_agg_opt: str = 'concat'  # concat: concatenate in-context phenotypes; count: aggregate by count
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
    if use_struct_vocab:
        seq_lst = [protein_data[pid]['comb_seq'] for pid in prot_unique]
    # max_seq_length = max(prot_lengths)
    # if not max_protein_length:
    #     max_protein_length = protein_tokenizer.model_max_length
    # else:
    #     max_protein_length = min(max_protein_length, protein_tokenizer.model_max_length)

    # batch_seq_tokenized = protein_tokenizer(seq_lst, padding=True, truncation=True, return_tensors='pt', max_length=max_protein_length)

    patho_var_prot_idx = []  # index of protein in the batch for pathogenic variants
    prot_idx_all = []
    var_idx_all = []
    offset_all = []
    ref_aa_idx = []
    alt_aa_idx = []
    pos_pheno_desc_all = []
    pos_pheno_name_all = []
    pos_pheno_idx_all = []
    neg_pheno_desc_all = []
    neg_pheno_idx_all = []
    pos_pheno_known_vec = []
    phenos_in_frame_all = []
    context_pheno_dict = defaultdict(list)  # for context-disease information
    var_names_all = []
    batch_label = []
    pheno_var_names = []
    infer_pheno_vec = []
    # var_seq_input_ids = []
    masked_seq_list = []
    var_seq_list = []
    has_struct_context_all = []
    var_struct_graphs_raw = []
    struct_context_pheno_uniq = set()

    # batch_seq_input_lst = batch_seq_tokenized['input_ids'].tolist()
    # phenos_in_frame_input_ids = [] # List[torch.Tensor] --> length=n_variants
    pheno_var_indice = 0

    for b, elem in enumerate(batch_data_raw):
        # var_idx_lst = elem['var_idx']
        uprot = elem['uprot']
        prot_idx_cur = prot_unique.index(uprot)
        
        protein_info_cur = protein_data[uprot]
        prot_context_var_idx = protein_info_cur['context_var_idx']
        prot_desc = protein_info_cur['prot_desc']
        pos_pheno_name = elem['pos_pheno_desc']
        if use_pheno_desc and pheno_desc_dict is not None:
            pos_pheno_descs = '{name} {desc}'.format(name=pos_pheno_name, desc=pheno_desc_dict.get(pos_pheno_name, '')).strip()
        else:
            pos_pheno_descs = elem['pos_pheno_desc']
        # neg_pheno_descs = elem['neg_pheno_desc']
        if use_struct_vocab:
            ref_aa_idx.append(protein_tokenizer.convert_tokens_to_ids(elem['ref_aa'][0] + foldseek_vocab[0]))  # start index (will sum over all possible struct)
            alt_aa_idx.append(protein_tokenizer.convert_tokens_to_ids(elem['alt_aa'][0] + foldseek_vocab[0]))
        else:
            ref_aa_idx.append(protein_tokenizer.convert_tokens_to_ids(elem['ref_aa']))
            alt_aa_idx.append(protein_tokenizer.convert_tokens_to_ids(elem['alt_aa']))

        var_names_all.append(elem['id'])
        var_idx = elem['var_idx']
        batch_label.append(elem['label'])
        infer_pheno_vec.append(int(elem['infer_phenotype']))
        pos_pheno_known_vec.append(int(elem['pos_pheno_is_known']))

        # ref_seq_input_ids = batch_seq_input_lst[prot_idx_cur]
        ref_seq = seq_lst[prot_idx_cur]  # TODO: NEED TO REVISE FOR STRUCT_VOCAB (SaProt)
        cur_length = prot_lengths[prot_idx_cur]
        masked_seq_tokens = protein_tokenizer.tokenize(ref_seq)
        start = 0
        end = cur_length
        if truncate_protein:
            start, end = get_optimal_window(var_idx+1, cur_length, max_protein_length)
        offset_all.append(start)
        if use_struct_vocab:
            masked_seq_tokens[var_idx] = '#' + elem['ref_aa'][-1]  # mask AA type, keep original struct token
        else:
            masked_seq_tokens[var_idx] = protein_tokenizer.mask_token
        alt_seq_tokens = masked_seq_tokens[:var_idx] + [elem['alt_aa']] + masked_seq_tokens[var_idx+1: ]
        masked_seq_cur = ''.join(masked_seq_tokens[start:end])
        var_seq_cur = ''.join(alt_seq_tokens[start:end])
        # else:
        #     masked_seq_cur = ref_seq[:var_idx+1] + protein_tokenizer.mask_token + ref_seq[var_idx+2: ]
        #     var_seq_cur = ref_seq[:var_idx+1] + elem['alt_aa'] + ref_seq[var_idx+2: ]
        masked_seq_list.append(masked_seq_cur)
        var_seq_list.append(var_seq_cur)
        # var_seq_input_cur = ref_seq_input_ids[:var_idx+1] + [elem['alt_aa']] + ref_seq_input_ids[var_idx+2:]
        # var_seq_input_ids.append(var_seq_input_cur)
        
        var_idx_all.append(var_idx)
        prot_idx_all.append(prot_idx_cur)
        # for var_idx in var_idx_lst:
        has_struct_context = False
        if elem['infer_phenotype']:
            phenos_in_frame_cur, context_pheno_var_idx_cur = fetch_phenotypes_in_frame(var_idx, prot_lengths[prot_idx_cur], prot_context_var_idx, 
                                                                                   protein_info_cur['var_pheno_descs'], half_window_size, max_context_phenos, 
                                                                                   use_pheno_desc=use_pheno_desc, pheno_desc_dict=pheno_desc_dict,
                                                                                   mask_token=text_tokenizer.mask_token, unknown_token=text_tokenizer.unk_token,
                                                                                   include_unknown=include_unknown)
            patho_var_prot_idx.append(prot_idx_cur)
            pheno_var_names.append(elem['id'])
            if has_phenotype_label:
                pos_pheno_desc_all.append(pos_pheno_descs)
                pos_pheno_name_all.append(pos_pheno_name)
                pos_pheno_idx_all.append(elem['pos_pheno_idx'])
            # neg_pheno_label_all.append(neg_pheno_descs)
            # phenos_in_frame_cur_full = [prot_desc] + phenos_in_frame_cur
            # context_info_joined = text_tokenizer.sep_token.join([prot_desc] + phenos_in_frame_cur)  # joined into single string
            if use_struct_neighbor and ('struct_graph' in protein_info_cur):
                # TODO: check empty graph case
                # var_context_graph = extract_context_graph(var_idx, prot_context_var_idx, protein_info_cur['struct_graph'], 
                #                                           use_pheno_desc=use_pheno_desc, pheno_desc_dict=pheno_desc_dict, 
                #                                           mask_token_id=text_tokenizer.mask_token_id, mask_token=text_tokenizer.mask_token, dist_cutoff=struct_radius)
                # var_context_graph = elem['var_struct_graph']
                var_graph_edges = elem['var_graph_edges']
                var_context_graph = nx.DiGraph()
                if elem['num_struct_neighbors']:
                    var_context_graph.add_weighted_edges_from(var_graph_edges, weight='distance')
                    prot_pheno_dict = {'pheno_descs': dict(zip(range(protein_info_cur['seq_length']), protein_info_cur['var_pheno_descs'])),
                                       'pheno_idx': dict(zip(range(protein_info_cur['seq_length']), protein_info_cur['var_pheno_idx']))}
                    var_context_graph = update_node_attrs(var_idx, var_context_graph, prot_pheno_dict, 
                                                          use_pheno_desc=use_pheno_desc, pheno_desc_dict=pheno_desc_dict, 
                                                          mask_token_id=text_tokenizer.mask_token_id, mask_token=text_tokenizer.mask_token)
                    struct_context_pheno_uniq.update(nx.get_node_attributes(var_context_graph, 'pheno_descs').values())
                    var_struct_graphs_raw.append(var_context_graph)
                    has_struct_context = True
                    # var_struct_graphs.append(dgl.from_networkx(var_context_graph, node_attrs=['pheno_idx'], edge_attrs=['distance']))
            has_struct_context_all.append(has_struct_context)

            if context_agg_opt == 'concat':
                context_info_joined = text_tokenizer.sep_token.join(phenos_in_frame_cur)  # remove protein desc from context (treat separately)
                # phenos_in_frame_all.append(context_info_joined)
                context_pheno_dict['phenotype'].append(context_info_joined)
                context_pheno_dict['size'].append(1)
                context_pheno_dict['indice'].append(pheno_var_indice)
                # context_pheno_dict['indice'].extend([pheno_var_indice] * len(phenos_in_frame_cur))
                pheno_var_indice += 1
            else:  # stack
                context_pheno_dict['phenotype'].extend(phenos_in_frame_cur)  # modified
            # context_pheno_dict['counts'].append(1)
            # context_pheno_dict['indice'].extend([pheno_var_indice] * len(phenos_in_frame_cur))
            context_pheno_dict['size'].append(len(context_pheno_var_idx_cur))  # number of contextual disease variants
            context_pheno_dict['position_idx'].extend(context_pheno_var_idx_cur)  # position index of each disease variant
            # phenos_in_frame_all.append([prot_desc] + phenos_in_frame_cur)
            # pheno_var_indice += 1

            if mode == 'train':
                neg_pheno_idx, neg_pheno_name = sample_negative(pheno_vocab, pos_pheno_name)
                if use_pheno_desc and pheno_desc_dict is not None:
                    neg_pheno_desc = '{name} {desc}'.format(name=neg_pheno_name, desc=pheno_desc_dict.get(neg_pheno_name, '')).strip()
                else:
                    neg_pheno_desc = neg_pheno_name
                neg_pheno_desc_all.append(neg_pheno_desc)
                neg_pheno_idx_all.append(neg_pheno_idx)

        # phenos_all.extend([prot_desc] + phenos_in_frame_cur)
            # max_pheno_length = max(max_pheno_length, phenos_input_ids_cur.shape[-1])
            # phenos_in_frame_input_ids.append(phenos_input_ids_cur)
    masked_seq_tokenized = protein_tokenizer(masked_seq_list, padding=True, return_tensors='pt')
    var_seq_tokenized = protein_tokenizer(var_seq_list, padding=True, return_tensors='pt')

    if use_prot_desc:
        desc_lst = [protein_data[pid]['prot_desc'] for pid in prot_unique]
        batch_desc_tokenized = text_tokenizer(desc_lst, padding=True, return_tensors='pt', truncation=True, max_length=max_pheno_desc_length)
        # max_desc_length = batch_desc_tokenized['input_ids'].shape[-1]

    if sum(infer_pheno_vec) == 0:
        variant_dict = {
            'indices': [elem['indice'] for elem in batch_data_raw],  # indice in dataset (in original order without shuffling)
            'var_pos': [elem['var_pos'] for elem in batch_data_raw],
            'var_idx': var_idx_all,
            'var_names': var_names_all,
            'offset_idx': torch.tensor(offset_all),
            'ref_aa': torch.LongTensor(ref_aa_idx),
            'alt_aa': torch.LongTensor(alt_aa_idx),
            'infer_phenotype': False,
            # 'var_seq_input_ids': torch.tensor(var_seq_tokenized['input_ids']),
            # 'n_variants': [len(elem['alt_aa']) for elem in batch_data_raw],
            'pos_pheno_is_known': torch.tensor(pos_pheno_known_vec),
            'prot_idx': prot_idx_all,
            'use_struct': use_struct_neighbor and any(has_struct_context_all),
            'has_struct_context': has_struct_context_all,
            'var_seq_input_feat': {
                'input_ids': var_seq_tokenized['input_ids'],  # batch_size x max_length
                'attention_mask': var_seq_tokenized['attention_mask'],  # 1 if token should be attended to
                'token_type_ids': torch.zeros_like(var_seq_tokenized['input_ids'], dtype=torch.long)
            },
            'masked_seq_input_feat': {
                'input_ids': masked_seq_tokenized['input_ids'],  # batch_size x max_length
                'attention_mask': masked_seq_tokenized['attention_mask'],  # 1 if token should be attended to
                'token_type_ids': torch.zeros_like(masked_seq_tokenized['input_ids'], dtype=torch.long)
            },
        }
    else:
       
        # batch_pheno_tokenized = text_tokenizer(phenos_in_frame_all, padding=True, return_tensors='pt', truncation=True, max_length=max_pheno_desc_length)
        batch_pheno_tokenized = text_tokenizer(context_pheno_dict['phenotype'], padding=True, return_tensors='pt', truncation=True, max_length=max_pheno_desc_length)
        batch_pheno_input_ids = batch_pheno_tokenized['input_ids']
        # batch_pheno_input_ids = torch.stack(pheno_input_ids_padded)  # n_variants, max_phenos_in_frame + 1, max_pheno_length
        # batch_pheno_attenton_mask = (batch_uniq_input_ids != text_tokenizer.pad_token_id).long()
        variant_dict = {
            'indices': [elem['indice'] for elem in batch_data_raw], 
            'var_pos': [elem['var_pos'] for elem in batch_data_raw],
            'var_idx': var_idx_all,
            'var_names': var_names_all,
            # 'var_seq_input_ids': torch.tensor(var_seq_tokenized['input_ids']),
            'offset_idx': torch.tensor(offset_all),
            'pheno_var_names': pheno_var_names,
            'ref_aa': torch.LongTensor(ref_aa_idx),
            'alt_aa': torch.LongTensor(alt_aa_idx),
            'infer_phenotype': True,
            "context_pheno_positions": torch.LongTensor(context_pheno_dict['position_idx']),
            'context_pheno_size': context_pheno_dict['size'],  # list
            # 'n_variants': [len(elem['alt_aa']) for elem in batch_data_raw],
            'patho_var_prot_idx': patho_var_prot_idx,
            'pos_pheno_avail': pos_pheno_known_vec,
            'prot_idx': prot_idx_all,
            'use_struct': use_struct_neighbor and any(has_struct_context_all),
            'has_struct_context': has_struct_context_all,
            'var_seq_input_feat': {
                'input_ids': var_seq_tokenized['input_ids'],  # batch_size x max_length
                'attention_mask': var_seq_tokenized['attention_mask'],  # 1 if token should be attended to
                'token_type_ids': torch.zeros_like(var_seq_tokenized['input_ids'], dtype=torch.long)
            },
            'masked_seq_input_feat': {
                'input_ids': masked_seq_tokenized['input_ids'],  # batch_size x max_length
                'attention_mask': masked_seq_tokenized['attention_mask'],  # 1 if token should be attended to
                'token_type_ids': torch.zeros_like(masked_seq_tokenized['input_ids'], dtype=torch.long)
            },
        }
        if context_agg_opt == 'concat':
            variant_dict['context_pheno_indices'] = torch.LongTensor(context_pheno_dict['indice'])
            variant_dict['context_pheno_input_ids'] = batch_pheno_input_ids
            variant_dict['context_pheno_attention_mask'] = (batch_pheno_input_ids != text_tokenizer.pad_token_id).long()
        else:
            batch_uniq_input_ids, batch_uniq_indices = torch.unique(batch_pheno_input_ids, dim=0, return_inverse=True)
            variant_dict['context_pheno_indices'] = batch_uniq_indices
            variant_dict['context_pheno_input_ids'] = batch_uniq_input_ids
            variant_dict['context_pheno_attention_mask'] = (batch_uniq_input_ids != text_tokenizer.pad_token_id).long()
        # if context_agg_opt == 'count':
        #     variant_dict['context_pheno_counts'] = torch.tensor(context_pheno_dict['counts'])

        if has_phenotype_label:
            pos_pheno_tokenized = text_tokenizer(pos_pheno_desc_all, padding=True, return_tensors='pt', truncation=True, max_length=max_pheno_desc_length)
            variant_dict.update({
                'pos_pheno_name': pos_pheno_name_all,
                'pos_pheno_desc': pos_pheno_desc_all,
                'pos_pheno_idx': torch.tensor(pos_pheno_idx_all),
                'pos_pheno_input_ids': pos_pheno_tokenized['input_ids'],
                'pos_pheno_attention_mask': pos_pheno_tokenized['attention_mask']})
            
        if mode == 'train':
            neg_pheno_tokenized = text_tokenizer(neg_pheno_desc_all, padding=True, return_tensors='pt', truncation=True, max_length=max_pheno_desc_length)
            variant_dict.update({
                'neg_pheno_idx': torch.tensor(neg_pheno_idx_all),
                'neg_pheno_desc': neg_pheno_desc_all,
                'neg_pheno_input_ids': neg_pheno_tokenized['input_ids'],
                'neg_pheno_attention_mask': neg_pheno_tokenized['attention_mask']})
        
        if use_struct_neighbor and any(has_struct_context_all):
            struct_context_pheno_uniq = list(struct_context_pheno_uniq)
            var_struct_graphs = []
            for var_graph in var_struct_graphs_raw:
                node2pheno_idx = dict()
                for node, nfeats in var_graph.nodes(data=True):
                    node2pheno_idx[node] = struct_context_pheno_uniq.index(nfeats['pheno_descs'])
                nx.set_node_attributes(var_graph, node2pheno_idx, name='indice')
                var_struct_graphs.append(dgl.from_networkx(var_graph, node_attrs=['pheno_idx', 'indice', 'mask'], edge_attrs=['distance']))
            struct_pheno_tokenized = text_tokenizer(list(struct_context_pheno_uniq), padding=True, return_tensors='pt', truncation=True, max_length=max_pheno_desc_length)
            struct_pheno_input_ids = struct_pheno_tokenized['input_ids']
            struct_pheno_attention_mask = (struct_pheno_input_ids != text_tokenizer.pad_token_id).long()

            batch_struct_graph = dgl.batch(var_struct_graphs)
            batch_struct_graph.edata['distance'] = batch_struct_graph.edata['distance'].float()
            
            variant_dict.update({
                'var_struct_graph': batch_struct_graph,
                'struct_pheno_input_ids': struct_pheno_input_ids,
                'struct_pheno_attention_mask': struct_pheno_attention_mask
            })
    if use_alphamissense:
        variant_dict.update({
            'afmis_score': torch.tensor([elem['afmis_score'] for elem in batch_data_raw]), 
            'afmis_mask': torch.tensor([elem['afmis_mask'] for elem in batch_data_raw], dtype=bool),  # True if alphamissense NOT available
        })
    del var_struct_graphs_raw
    variant_dict['label'] = torch.tensor(batch_label)
    variant_dict['infer_pheno_vec'] = torch.tensor(infer_pheno_vec)
    
    return {
        # 'id': batch_prot_ids,
        'seq_length': prot_lengths, 
        'window_size': 2 * half_window_size,
        # 'variant_mask': batch_var_mask,  # true for variant (batch_size, max_length)
        # 'pathogenic_mask': batch_patho_mask,  # true for pathogenic (batch_size, max_length)
        'label': batch_label,  # 1: pathogenic, 0: benign, -100: non-variant spots
        # 'seq_input_feat': {  # reference sequence
        #     'input_ids': batch_seq_tokenized['input_ids'],  # batch_size x max_length
        #     'attention_mask': batch_seq_tokenized['attention_mask'],  # 1 if token should be attended to
        #     'token_type_ids': torch.zeros_like(batch_seq_tokenized['input_ids'], dtype=torch.long)
        # },
        'desc_input_feat': {
            'input_ids': batch_desc_tokenized['input_ids'],
            'attention_mask': batch_desc_tokenized['attention_mask'],
            'token_type_ids': torch.zeros_like(batch_desc_tokenized['input_ids'], dtype=torch.long)
        },
        'use_alphamissense': use_alphamissense,
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


def extract_context_graph(var_idx, 
                          prot_context_var_idx, 
                          g_prot, 
                          dist_cutoff=25):
    
    edge_dist_list = []
    context_nodes = []
    if nx.is_empty(g_prot):
        return edge_dist_list, context_nodes
    if var_idx not in g_prot:
        return edge_dist_list, context_nodes
    
    dist_to_target = nx.single_source_dijkstra_path_length(g_prot, var_idx, cutoff=dist_cutoff, weight='distance')
    edge_dist_list.append((var_idx, var_idx, 1e-5))  # add self-loop for context nodes
    for context_idx in prot_context_var_idx:
        if context_idx == var_idx:
            continue
        # if context_idx not in g_prot:
        #     continue
        # dist = nx.dijkstra_path_length(g_prot, var_idx, context_idx, weight='distance')
        # if dist <= dist_cutoff:
        #     edge_dist_list.append((context_idx, var_idx, dist))
        #     edge_dist_list.append((context_idx, context_idx, 1e-5))  # add self-loop for context nodes
        #     context_nodes.append(context_idx)
        if context_idx in dist_to_target:
            edge_dist_list.append((context_idx, var_idx, dist_to_target[context_idx]))
            edge_dist_list.append((context_idx, context_idx, 1e-5))  # add self-loop for context nodes
            context_nodes.append(context_idx)

    return edge_dist_list, context_nodes
    # var_context_graph.add_weighted_edges_from(edge_dist_list, weight='distance')
    
    # return var_context_graph
    # update_dict = dict()
    # # nodes = set(prot_context_var_idx + [var_idx])
    # nodes = var_context_graph.nodes()
    # for nid in nodes:
    #     ndata_dict = g_prot.nodes[nid]
    #     update_dict[nid] = copy.deepcopy(ndata_dict)
    #     update_dict[nid]['mask'] = 0 if nid != var_idx else 1
    #     if use_pheno_desc and pheno_desc_dict is not None:
    #         update_dict[nid]['pheno_descs'] = '{name} {desc}'.format(name=ndata_dict['pheno_descs'], desc=pheno_desc_dict.get(ndata_dict['pheno_descs'], '')).strip()
    # nx.set_node_attributes(var_context_graph, update_dict)
    # nx.set_node_attributes(var_context_graph, values={var_idx: {'pheno_idx': mask_token_id, 'pheno_descs': mask_token}})
    
    # struct_context_pheno_uniq.update(nx.get_node_attributes(var_context_graph, 'pheno_descs').values())
    # return var_context_graph

def update_node_attrs(var_idx, var_graph, prot_pheno_dict, use_pheno_desc=False, pheno_desc_dict=None, mask_token_id=0, mask_token='[MASK]'):
    # update_dict = copy.deepcopy(prot_pheno_dict)
    update_dict = {k: {} for k in prot_pheno_dict.keys()}
    update_dict['mask'] = dict()
    # nodes = set(prot_context_var_idx + [var_idx])
    nodes = var_graph.nodes()
    for nid in nodes:
        # ndata_dict = prot_graph.nodes[nid]
        # ndata_dict = prot_pheno_dict['pheno_desc'][nid]
        # update_dict[nid] = copy.deepcopy(ndata_dict)
        pheno_name = prot_pheno_dict['pheno_descs'][nid]
        update_dict['pheno_descs'][nid] = pheno_name
        update_dict['pheno_idx'][nid] = prot_pheno_dict['pheno_idx'][nid]
        update_dict['mask'][nid] = 0 if nid != var_idx else 1
        if use_pheno_desc and pheno_desc_dict is not None:
            update_dict['pheno_descs'][nid] = '{name} {desc}'.format(name=pheno_name, desc=pheno_desc_dict.get(pheno_name, '')).strip()
    for attr_name, attr_dict in update_dict.items():
        nx.set_node_attributes(var_graph, attr_dict, name=attr_name)
    # nx.set_node_attributes(var_graph, update_dict)
    nx.set_node_attributes(var_graph, values={var_idx: {'pheno_idx': mask_token_id, 'pheno_descs': mask_token}})
    # struct_context_pheno_uniq.update(nx.get_node_attributes(var_context_graph, 'pheno_descs').values())
    return var_graph


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


def merge_struct_graphs(g_pdb, g_af):
    if nx.is_empty(g_pdb):
        return g_af
    if nx.is_empty(g_af):
        return g_pdb
    
    g_prot = nx.Graph()
    # Add nodes
    # for node, data in g_pdb.nodes(data=True):
    #     g_prot.add_node(node, **data)
    # for node, data in g_af.nodes(data=True):
    #     if node in g_prot:
    #         g_prot.nodes[node].update(data)
    #     else:
    #         g_prot.add_node(node, **data)
    g_prot.add_nodes_from(g_pdb.nodes(data=True))
    g_prot.add_edges_from(g_pdb.edges(data=True))

    # for u, v, data in g_pdb.edges(data=True):
    #     g_prot.add_edge(u, v, **data)

    # Add edges from the second graph, merging distances
    for u, v, data in g_af.edges(data=True):
        if not g_prot.has_edge(u, v):
            g_prot.add_edge(u, v, **data)
    
    return g_prot


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
        pheno_desc_dict: Dict[str, str] = None,
        use_desc: bool = False  # use phenotype description or not
        # tokenizer: PreTrainedTokenizerBase = None,
        # max_phenotype_length: int = None,
        # embed_all = True
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
                return ' '.join([pheno_name, desc])
            except KeyError:
                pass
        return pheno_name

    def __len__(self):
        return len(self.phenotypes)

@dataclass
class TextDataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    truncation: bool = True
    max_length: int = 512

    def __call__(self, batch):
        batch_tokenized = self.tokenizer(batch, padding=self.padding, truncation=self.truncation, return_tensors='pt', max_length=self.max_length)
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

