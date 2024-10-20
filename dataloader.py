import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def data_mask(all_session_data, padding_item):
    sessions_len = [len(session) for session in all_session_data]
    max_len = max(sessions_len)
    session_padded_seq = [session_info + padding_item * (max_len - le)
                          for session_info, le in zip(all_session_data, sessions_len)]
    session_padded_mask = [[1] * le + [0] * (max_len - le) for le in sessions_len]

    return session_padded_seq, session_padded_mask, max_len


def compute_max_len(rawdata):
    session_len = [len(session) for session in rawdata]
    max_len = max(session_len)
    return max_len


def compute_max_n_node(rawdata):
    session_n_node = [len(np.unique(session)) for session in rawdata]
    max_n_node = max(session_n_node)
    return max_n_node


class MyDataset(Dataset):
    def __init__(self, rawdata):
        self.inputs = rawdata[0]
        # max_len: 这一段数据集中的最大长度
        self.max_len = compute_max_len(self.inputs)
        self.max_n_node = compute_max_n_node(self.inputs)
        self.targets = rawdata[1]

    def __len__(self):
        return len(self.inputs)

    # 输出 items:会话中的项目 / mask: 原始会话中的掩码 / targets / A:邻接矩阵 / alias_inputs: 用于恢复原序列的别名列表
    # 需要为每个会画图构建图 并保存别名序列
    def __getitem__(self, index):
        """
        :param index: the index of batch; index = batch_size
        :return:
            item [ len]:
            mask [ len]:
            target [1]:
            alias_input [len]:
        """
        # TODO: 修改 input, mask, target
        one_input = self.inputs[index] + (self.max_len - len(self.inputs[index])) * [0]
        mask = len(self.inputs[index]) * [1] + (self.max_len - len(self.inputs[index])) * [0]
        target = self.targets[index] - 1

        return one_input, mask, target
        # node = np.unique(one_input)
        #
        # item = node.tolist() + (self.max_n_node - len(node)) * [0]
        # alias_input = [np.where(node == i)[0][0] for i in one_input]
        #
        # u_A = np.zeros((self.max_n_node, self.max_n_node))
        # for i in np.arange(len(one_input) - 1):
        #     if one_input[i] == 0:
        #         break
        #     u = np.where(node == one_input[i])[0][0]
        #     v = np.where(node == one_input[i + 1])[0][0]
        #     u_A[u][v] = 1
        #
        # # 求入度: 行相加
        # u_sum_in = np.sum(u_A, 0)
        # # 方便下列归一化时除数不为0
        # u_sum_in[np.where(u_sum_in == 0)] = 1
        # u_A_in = np.divide(u_A, u_sum_in)
        #
        # # 同理求出度
        # u_sum_out = np.sum(u_A, 1)
        # u_sum_out[np.where(u_sum_out == 0)] = 1
        # u_A_out = np.divide(u_A.transpose(), u_sum_out)
        #
        # # 拼接出入度矩阵 完善u_A 然后将其添加到A中/ 使用transpose进行转置使得 u_A 的行作为出入度信息而不是列 即.sum(u_A, 1)
        # u_A = np.concatenate([u_A_in, u_A_out]).transpose()
        #
        # return item, mask, target, u_A, alias_input


def collate_fn(batch):
    inputs, masks, targets = zip(*batch)
    inputs, masks, targets = list(inputs), list(masks), list(targets)
    items,alias_inputs, A = [], [], []
    n_node = []
    for session in inputs:
        n_node.append(len(np.unique(session)))
    # 记录项目节点的最大数 -> 用于构建邻接矩阵
    batch_max_n_node = np.max(n_node)
    # 记录会话的最大长度 -> 用于 padding

    for session in inputs:
        node = np.unique(session)
        # items: [b, batch_max_n_node]
        items.append(node.tolist() + [0] * (batch_max_n_node - len(node)))
        #
        alias_inputs.append([np.where(node == i)[0][0] for i in session])

        u_A = np.zeros((batch_max_n_node, batch_max_n_node))
        # 使用 np.arange 进行迭代
        for i in np.arange(len(session) - 1):
            if session[i+1] == 0:
                break
            u = np.where(node == session[i])[0][0]
            v = np.where(node == session[i+1])[0][0]
            u_A[u][v] = 1
        # u_A = u_A
        u_sum_in = np.sum(u_A, 0)
        u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A, u_sum_in)

        u_sum_out = np.sum(u_A, 1)
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A.transpose(), u_sum_out)

        u_A = np.concatenate([u_A_in, u_A_out]).transpose()
        #
        A.append(u_A)
    return items, masks, targets, A, alias_inputs


def get_dataloader(dataset, batch_size):
    dataset2loader = MyDataset(rawdata=dataset)
    return DataLoader(dataset2loader, batch_size=batch_size, shuffle=True)


def split_validation(train_data, valid_rate):
    """
    :param train_data: tuple: train_data[0]->session; train_data[1]->target
    :param valid_rate: the rate of validation in all data set
    :return:
        (train_session_set, train_target_set), (valid_session_set, valid_target_set)
    """

    session_data, target_data = train_data
    num_samples = np.max(session_data)
    sidx = np.arange(len(session_data))
    np.random.shuffle(sidx)
    num_train = int(np.round(num_samples * (1. - valid_rate)))

    train_session_set = [session_data[s] for s in sidx[:num_train]]
    train_target_set = [target_data[s] for s in sidx[:num_train]]

    valid_session_set = [session_data[s] for s in sidx[num_train:]]
    valid_target_set = [target_data[s] for s in sidx[num_train:]]

    return (train_session_set, train_target_set), (valid_session_set, valid_target_set)
