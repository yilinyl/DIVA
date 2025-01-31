import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def select_hard_negatives(query_embs, ref_embs, positive_indices, topk_escape=50, n_negatives=100):
    """
    pred_embs: Tensors of shape (batch_size, embedding_dim)
    vocab_embs: Tensors of shape (vocab_size, embedding_dim)
    positive_indices: Tensors of shape (batch_size, )
    """
    pred_embs_norm = F.normalize(query_embs, dim=-1)
    vocab_embs_norm = F.normalize(ref_embs, dim=-1)
    positive_embs = vocab_embs_norm[positive_indices]  # (batch_size, embedding_dim)
    sim_with_pred = torch.matmul(pred_embs_norm, vocab_embs_norm.t())  # (batch_size, vocab_size)
    sim_with_positive = torch.matmul(positive_embs, vocab_embs_norm.t())
    batch_size = positive_embs.size(0)

    escape_indices = torch.topk(sim_with_positive, k=topk_escape, dim=-1).indices
    row_indices = torch.arange(batch_size).unsqueeze(1)
    mask = torch.zeros_like(sim_with_positive)
    mask[row_indices, escape_indices] = 1
    sim_with_positive[mask.bool()] = -1
    hard_neg_indices = torch.topk(sim_with_pred, k=n_negatives).indices

    return hard_neg_indices


def sample_random_negative(vocab_size, positive_indices, n_neg=1):
    batch_size = positive_indices.size(0)
    pheno_idx_all = np.arange(vocab_size)
    neg_idx_list = []
    for i in range(batch_size):
        # pos_idx = positive_indices.detach()[i].item()
        pos_idx = positive_indices[i].detach().cpu().numpy()
        sample_mask = np.zeros(vocab_size, dtype=bool)
        sample_mask[pos_idx] = True
        
        neg_sample_idx = np.random.choice(pheno_idx_all[~sample_mask], size=n_neg, replace=False)  # scalar
        neg_idx_list.append(neg_sample_idx)
    
    negative_indices = torch.tensor(np.array(neg_idx_list), device=positive_indices.device)

    return negative_indices


def embed_phenotypes(model, device, pheno_loader):
    model.eval()
    all_pheno_embs = []

    with torch.no_grad():
        for idx, batch_pheno in enumerate(pheno_loader):
            # pheno_input_dict = load_input_to_device(batch_pheno, device)
            pheno_input_dict = batch_pheno.to(device)
            pheno_embs = model.get_pheno_emb(pheno_input_dict, proj=True, agg_opt='cls')
            all_pheno_embs.append(pheno_embs.detach().cpu().numpy())
    
        all_pheno_embs = np.concatenate(all_pheno_embs, 0)
    
    return all_pheno_embs