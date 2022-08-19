import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.HoGN import HoGN
from model.GNN import GNN

from torch_geometric.nn import GCNConv, TopKPooling, SAGEConv, EdgePooling
class BiHo(nn.Module):
    def __init__(self, opt):
        super(BiHo, self).__init__()
        self.opt=opt
        self.GNN = GNN(opt) # fast mode(GNN), slow mode(GNN2)
        self.HoGN = HoGN(opt)
        self.rna_embedding = nn.Embedding(opt["number_rna"], opt["feature_dim"])
        self.protein_embedding = nn.Embedding(opt["number_protein"], opt["feature_dim"])
        self.protein_index = torch.arange(0, self.opt["number_protein"], 1)
        self.rna_index = torch.arange(0, self.opt["number_rna"], 1)
        if self.opt["cuda"]:
            self.protein_index = self.protein_index.cuda()
            self.rna_index = self.rna_index.cuda()

    def score_predict(self, fea):
        out = self.GNN.score_function1(fea)
        out = F.relu(out)
        out = self.GNN.score_function2(out)
        out = torch.sigmoid(out)
        return out.view(out.size()[0], -1)

    def score(self, fea):
        out = self.GNN.score_function1(fea)
        out = F.relu(out)
        out = self.GNN.score_function2(out)
        out = torch.sigmoid(out)
        return out.view(-1)

    def forward(self, ufea, vfea, UV_adj, VU_adj, adj):
        learn_rna,learn_protein = self.GNN(ufea,vfea,UV_adj,VU_adj,adj)
        return learn_rna,learn_protein
