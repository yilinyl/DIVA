import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModel

ESM_MODEL = "facebook/esm2_t12_35M_UR50D"

def init_pretrained_lm(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

    return tokenizer, model


def calc_esm_emb(seq, tokenizer, model, clip=True):
    device = model.device
    with torch.no_grad():
        inputs = tokenizer(seq, return_tensors='pt').to(device)
        outputs = model(**inputs)
        emb = outputs.last_hidden_state.squeeze(0)

        if clip:
            return emb[1:-1].detach().cpu()  # ESM add two additiona dims at the start & end

        return emb.detach().cpu()


def simple_random_sampling(
    num_neg_sample: int,
    vocab: List = None,
    **kwargs
) -> List:
    """
    A simple strategy of negative sampling.

    Args:
        cur_entity: Fixed entity (id) with relation (id) which used to construct negative triplet samples with another entity 
                    sampled from a specific entity set.
        true_triplet: dict containing postive triplets. Key is (head,relation) tail is set of positive triplets
        num_neg_sample: the number of negative sampling.
        vocab: List containing all phenotype terms valid as potential negative sample for the current phenotype term
    """
    
    negative_sample_list = []
    negative_sample_size = 0

    while negative_sample_size < num_neg_sample:
        negative_sample = np.random.choice(vocab, size=num_neg_sample, replace=False)

        negative_sample_list.append(negative_sample)
        negative_sample_size += negative_sample.size
    
    negative_sample = np.concatenate(negative_sample_list)[:num_neg_sample]
    return negative_sample


negative_sampling_strategy = {
    'simple_random': simple_random_sampling,
}