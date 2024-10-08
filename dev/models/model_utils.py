import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_loss(logits, label, device, use_weight_in_loss=False):
    if use_weight_in_loss:
        V = label.size(0)
        label_count = torch.bincount(label.long())
        cluster_sizes = torch.zeros(label_count.size(0)).long().to(device)
        cluster_sizes[torch.arange(label_count.size(0)).long()] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()

        criterion = nn.BCEWithLogitsLoss(weight=weight[label.long()])
    else:
        criterion = nn.BCEWithLogitsLoss()

    loss = criterion(logits, label.float())

    return loss


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