import numpy as np
import torch
from torch import nn
from itertools import chain


class Loss(nn.Module):
    def __init__(self, scores, targets):
        self.scores = scores
        self.targets = targets

    def calculate_loss(self):
        return nn.CrossEntropyLoss(self.scores, self.targets)


def compute_node_num(all_data):
    # 返回一个chain对象
    all_data_1D = list(chain.from_iterable(all_data))
    n_node = len(np.unique(np.asarray(all_data_1D)))

    return n_node


class Metrics(nn.Module):
    def __init__(self, score, k):
        self.score = score
        self.k = k

    def calculate_mrr_k(self):
        print(self.score)
