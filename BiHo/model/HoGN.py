import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import SAGEConv
import numpy as np
import torch.nn as nn


class HoGN(torch.nn.Module):
    def __init__(self,opt):
        super(HoGN, self).__init__()
        self.opt=opt
        self.conv1 = SAGEConv(opt['feature_dim'], 128)
        self.relu = nn.ReLU()
        self.lin1 = torch.nn.Linear(128, 64)
        self.lin2 = torch.nn.Linear(64, 32)
        self.lin3 = torch.nn.Linear(32, 2)
    def get_link_labels(self, pos_edge_index, neg_edges):
        E = pos_edge_index.size(1) + neg_edges.size(1)

        link_labels = torch.zeros(E, dtype=torch.float)

        link_labels[:pos_edge_index.size(1)] = 1.

        return link_labels

    def forward(self,rna_f, protein_f, all_edges): 
        n_fea = torch.cat((rna_f, protein_f), dim = 0)
        edges = []
        for i in range(len(all_edges)):
            if i % 2 == 0:
                edges.append(list(all_edges[i]))
        edges = np.array(edges)
        edges = torch.from_numpy(edges).t().type(torch.long)

        neg_edges = negative_sampling(
            edge_index=edges, num_nodes=self.opt['number_rna'] + self.opt['number_protein'],
            num_neg_samples=edges.size(1),  # 负采样数量根据正样本
            force_undirected=True,
        )
        edge_index = torch.cat([edges, neg_edges], dim=-1)

        logits = (n_fea[edge_index[0]] * n_fea[edge_index[1]])
        logits = self.conv1(logits, edge_index)
        logits = self.relu(logits)
        logits = self.lin1(logits)
        logits = self.relu(logits)
        logits = self.lin2(logits)
        logits = self.relu(logits)
        logits = self.lin3(logits)
        Prob = F.log_softmax(logits, dim=-1)#.max(dim = 1)[1]
        Label = self.get_link_labels(edges, neg_edges)
        return Prob, Label.long()
