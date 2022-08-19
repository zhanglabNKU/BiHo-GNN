"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np




class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, rna_real_dict, protein_real_dict, evaluation):
        self.batch_size = batch_size
        self.opt = opt
        self.eval = evaluation
        self.ma = {}
        with open(filename) as infile:
            data=[]
            for line in infile:
                line=line.strip().split("\t")
                data.append([int(line[0]),int(line[1]),int(line[2])])
                if int(line[0]) not in self.ma.keys():
                    self.ma[int(line[0])] = set()
                self.ma[int(line[0])].add(int(line[1]))
        self.raw_data = data
        self.rna_real_dict = rna_real_dict




        data = self.preprocess(data, opt)

        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
            if batch_size > len(data):
                batch_size = len(data)
                self.batch_size = batch_size
            if len(data)%batch_size != 0:
                data += data[:batch_size]
            data = data[: (len(data)//batch_size) * batch_size]
        self.num_examples = len(data)
        if not evaluation:
            data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        else :
            data = [data]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def preprocess(self, data, opt):
        """ Preprocess the data and convert to ids. """
        processed = []
        self.rna_protein_pair = []
        for mytuple in data:
            processed.append((mytuple[0],mytuple[1],mytuple[2]))
            if len(self.rna_real_dict[mytuple[0]]) > self.opt["min_neighbor"] and len(self.protein_real_dict[mytuple[1]]) > self.opt["min_neighbor"] :
                self.rna_protein_pair.append((mytuple[0], mytuple[1]))
        return processed

    def __len__(self):
        return len(self.data)

    def __getprotein__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        if self.eval :
            batch = list(zip(*batch))
            return torch.LongTensor(batch[0]), torch.LongTensor(batch[1])+torch.tensor(self.opt["number_rna"]), np.array(batch[2])
        else :
            negative_tmp = []
            for i in range(batch_size):
                for j in range(self.opt["negative"]):
                    while 1:
                        rand = random.randint(0,self.opt["number_protein"]-1)
                        if rand not in self.rna_real_dict[batch[i][0]]:
                            negative_tmp.append((batch[i][0],rand))
                            break
            batch = list(zip(*batch))
            negative_tmp = list(zip(*negative_tmp))

        return torch.LongTensor(batch[0]), torch.LongTensor(batch[1]),torch.LongTensor(negative_tmp[1]) # User , protein,  neg_protein -> batch | batch | batch
    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getprotein__(i)

