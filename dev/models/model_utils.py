import torch
import torch.nn as nn


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
