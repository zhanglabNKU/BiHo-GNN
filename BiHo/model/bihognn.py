import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import util
from model.BiHo import BiHo

class Trainer(object):
    def __init__(self, opt):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError


    def update_lr(self, new_lr):  # here should change
        util.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

class BIHOTrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt
        self.model = BiHo(opt)
        self.criterion = nn.BCELoss()
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = util.get_optimizer(opt['optim'], self.model.parameters(), opt['lr'])
        self.epoch_rec_loss = []
        self.epoch_ho_loss = []



    def unpack(self, batch, cuda):
        if cuda:
            inputs = [Variable(b.cuda()) for b in batch]
            rna_index = inputs[0]
            protein_index = inputs[1]
            negative_protein_index = inputs[2]
        else:
            inputs = [Variable(b) for b in batch]
            rna_index = inputs[0]
            protein_index = inputs[1]
            negative_protein_index = inputs[2]
        return rna_index, protein_index, negative_protein_index

    def update_bipartite(self, static_rna_feature, static_protein_feature, UV_adj, VU_adj, adj):

        rna_feature = self.model.rna_embedding(self.model.rna_index)
        protein_feature = self.model.protein_embedding(self.model.protein_index)

        self.rna_hidden_out, self.protein_hidden_out = self.model(rna_feature, protein_feature, UV_adj, VU_adj, adj)

    def my_index_select(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans

    def reconstruct(self, UV, VU, adj,all_edges, rna_feature, protein_feature, batch):
        self.model.train()
        self.optimizer.zero_grad()

        self.update_bipartite(rna_feature,protein_feature, UV, VU, adj)
        rna_hidden_out = self.rna_hidden_out
        protein_hidden_out = self.protein_hidden_out


        rna_One, protein_One, neg_protein_One = self.unpack(batch, self.opt["cuda"])

        rna_feature_Two = self.my_index_select(rna_hidden_out, rna_One)
        protein_feature_Two = self.my_index_select(protein_hidden_out, protein_One)
        neg_protein_feature_Two = self.my_index_select(protein_hidden_out, neg_protein_One)

        pos_One = self.model.score(torch.cat((rna_feature_Two, protein_feature_Two), dim=1))
        neg_One = self.model.score(torch.cat((rna_feature_Two, neg_protein_feature_Two), dim=1))


        Label = torch.cat((torch.ones_like(pos_One), torch.zeros_like(neg_One)))#.cuda()
        pre = torch.cat((pos_One, neg_One))
        reconstruct_loss = self.criterion(pre, Label)
        Prob, Labels = self.model.HoGN(rna_hidden_out, protein_hidden_out, all_edges)
        #print(Prob[:1].size())
        Ho_loss = F.cross_entropy(Prob, Labels) 

        loss = (1 - self.opt["alpha"]) * reconstruct_loss +self.opt["alpha"] * Ho_loss
        self.epoch_rec_loss.append((1 - self.opt["alpha"]) * reconstruct_loss.item())
        self.epoch_ho_loss.append(self.opt["alpha"] * Ho_loss.item())
        loss.backward()
        self.optimizer.step()
        return loss.item()