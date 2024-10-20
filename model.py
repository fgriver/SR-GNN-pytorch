import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math


class GNN(nn.Module):
    def __init__(self, hidden_size, step=2):
        super(GNN, self).__init__()
        # ggnn卷积次数
        self.step = step

        self.hidden_size = hidden_size
        self.input_size = 2 * self.hidden_size
        self.gate_size = 3 * self.hidden_size  # 同时处理 更新门i 重置门r 新状态n
        # 注意是 Tensor 不是 tensor
        self.wi = nn.Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.bi = nn.Parameter(torch.Tensor(self.gate_size))

        self.wh = nn.Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.bh = nn.Parameter(torch.Tensor(self.gate_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size)

        self.b_iah = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = nn.Parameter(torch.Tensor(self.hidden_size))

    def GRUCell(self, A, hidden):
        """
        :param A:
        :param hidden:
        :return:
        """
        # A = A
        # hidden = hidden
        input_in = torch.matmul(A[:, :, :A.shape[1]].float(), self.linear_edge_in(hidden).float()) + self.b_iah.float()
        input_out = torch.matmul(A[:, :, A.shape[1]:].float(), self.linear_edge_out(hidden).float()) + self.b_oah.float()
        input_emb = torch.cat([input_in, input_out], 2)
        # 处理 inputs-> [b x l x 2*h]
        gi = F.linear(input_emb, self.wi, self.bi)
        # 处理 hidden_emb-> [b x l x h]
        gh = F.linear(hidden, self.wh, self.bh)
        # GRU需要按固定顺序拆分
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)

        update_gate = torch.sigmoid(i_i + h_i)
        reset_gate = torch.sigmoid(i_r + h_r)
        new_gate = torch.tanh(reset_gate * h_n + i_n)

        hidden_state = (1 - update_gate) * hidden + update_gate * new_gate

        return hidden_state

    def forward(self, A, hidden):
        for _ in range(self.step):
            hidden = self.GRUCell(A, hidden)
        return hidden


class SRGNN(nn.Module):
    def __init__(self, opt, n_node):
        super(SRGNN, self).__init__()
        self.hidden_size = opt.hidden_size

        self.n_node = n_node
        self.node_embedding = nn.Embedding(n_node + 1, self.hidden_size, padding_idx=0)
        self.gnn = GNN(self.hidden_size, step=opt.step)

        # q: hidden x 1
        self.q = nn.Linear(self.hidden_size, 1, bias=False)
        self.w_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_3 = nn.Linear(2 * self.hidden_size, self.hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, A, items, mask, alias_input):
        """
        :param A: 邻接矩阵
        :param items: 会话中的项目
        :param mask: 掩码
        :param alias_input: 用于还原原始会话的映射
        :return:
            s_hidden
        """
        items = items
        hidden = self.node_embedding(items)
        # 生成对应的项目嵌入
        hidden = self.gnn(A, hidden)
        hidden = hidden
        get = lambda i: hidden[i][alias_input[i]]
        seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_input)).long()])

        local_emb = seq_hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch x hidden_size

        q1 = self.w_1(local_emb).unsqueeze(1)  # batch x 1 x hidden_size
        q2 = self.w_2(seq_hidden)  # batch x len x hidden
        alpha = self.q(torch.sigmoid(q1 + q2))  # batch x len x 1

        # 在第一维求和 相当于将注意力分数与会话中的项目乘积后
        global_emb = torch.sum(alpha * seq_hidden * mask.view(mask.shape[0], -1, 1).float(), 1)  # batch x len x hidden

        # batch x len x 2 * hidden -> batch x len x hidden 要注意torch.cat的维度
        seq_hidden_final = self.w_3(torch.cat([local_emb, global_emb], 1))

        return seq_hidden_final

    def compute_score(self, seq_hidden_final):
        """
        :param seq_hidden_final: 已处理好的会话嵌入
        :return:
            z: batch x len x n_node: 会话中的每个项目与所有项目集内积得到的匹配得分
        """
        n_items = self.node_embedding.weight[1:]  # n_node x hidden
        z_i = torch.matmul(seq_hidden_final, n_items.transpose(1, 0))  # batch x len x node

        return z_i
