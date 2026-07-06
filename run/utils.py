import torch
from torch.utils.data import Dataset
import random as rd
from sklearn.preprocessing import StandardScaler
import numpy as np
def set_random_seed(seed: int = 42) -> None:
    rd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def convert_to_pyg_data(feature, adj_sparse, train_data, val_data, test_data):
    from torch_geometric.data import Data

    x = feature
    edge_index = adj_sparse.indices()

    train_pos_mask = train_data[:, 2] == 1
    train_neg_mask = train_data[:, 2] == 0
    train_pos_edge_index = train_data[train_pos_mask, :2].t()
    train_neg_edge_index = train_data[train_neg_mask, :2].t()

    def get_labels(data):
        pos_mask = data[:, 2] == 1
        neg_mask = data[:, 2] == 0
        pos_edges = data[pos_mask, :2].t()
        neg_edges = data[neg_mask, :2].t()
        return pos_edges, neg_edges

    val_pos, val_neg = get_labels(val_data)
    test_pos, test_neg = get_labels(test_data)

    train_pyg_data = Data(
        x=x,
        edge_index=edge_index,
        pos_edge_label_index=train_pos_edge_index,
        pos_edge_label=torch.ones(train_pos_edge_index.size(1)),
        neg_edge_label_index=train_neg_edge_index,
    )
    val_pyg_data = Data(
        x=x,
        edge_index=edge_index,
        pos_edge_label_index=val_pos,
        pos_edge_label=torch.ones(val_pos.size(1)),
        neg_edge_label_index=val_neg,
        neg_edge_label=torch.zeros(val_neg.size(1)),
    )
    test_pyg_data = Data(
        x=x,
        edge_index=edge_index,
        pos_edge_label_index=test_pos,
        pos_edge_label=torch.ones(test_pos.size(1)),
        neg_edge_label_index=test_neg,
        neg_edge_label=torch.zeros(test_neg.size(1)),
    )
    return train_pyg_data, val_pyg_data, test_pyg_data


class scRNADataset(Dataset):
    def __init__(self, train_set, num_gene, flag=False):
        super(scRNADataset, self).__init__()
        self.train_set = train_set
        self.num_gene = num_gene
        self.flag = flag

    def __getitem__(self, idx):
        train_data = self.train_set[:, :2]
        train_label = self.train_set[:, -1]

        if self.flag:
            train_len = len(train_label)
            train_tan = np.zeros([train_len, 2])
            train_tan[:, 0] = 1 - train_label
            train_tan[:, 1] = train_label
            train_label = train_tan

        data = train_data[idx].astype(np.int64)
        label = train_label[idx].astype(np.float32)

        return data, label

    def __len__(self):
        return len(self.train_set)
    def Adj_Generate(self, TF_set, direction=False, loop=False):
        N = self.num_gene
        rows = []
        cols = []
        for pos in self.train_set:
            if pos[-1] != 1:
                continue

            tf = int(pos[0])
            target = int(pos[1])

            if not direction:
                rows.extend([tf, target])
                cols.extend([target, tf])
            else:
                rows.append(tf)
                cols.append(target)
                if target in TF_set:
                    rows.append(target)
                    cols.append(tf)

        if loop:
            for i in range(N):
                rows.append(i)
                cols.append(i)

        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.ones(len(rows), dtype=torch.float32)

        adj = torch.sparse_coo_tensor(
            indices,
            values,
            size=(N, N)
        )

        adj = adj.coalesce()

        return adj






class load_data:
    def __init__(self, data, normalize=True):
        self.data = data
        self.normalize = normalize

    def data_normalize(self, data):
        standard = StandardScaler()
        epr = standard.fit_transform(data.T)

        return epr.T

    def exp_data(self):
        data_feature = self.data.values

        if self.normalize:
            data_feature = self.data_normalize(data_feature)

        data_feature = data_feature.astype(np.float32)

        return data_feature

def adj2saprse_tensor(adj):
    return adj.coalesce()






