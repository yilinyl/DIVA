import torch, dgl
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import GATConv, EGNNConv
import numpy as np


class MultiModel(nn.Module):
    def __init__(self):
        super(MultiModel, self).__init__()
