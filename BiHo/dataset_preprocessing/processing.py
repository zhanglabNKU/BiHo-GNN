import pandas as pd
import numpy as np
import random
import codecs 

def divide_dataset(div,dataset):
    random.shuffle(dataset)
    train_len = int(len(dataset)*(float(div)/10))+1

    train_set = dataset[:train_len]
    test_set = dataset[train_len:]
    adj={}
    protein_id= {}
    rna_id = {}
    for mytuple in train_set:
        if mytuple[0] not in rna_id:
            rna_id[mytuple[0]]=len(rna_id)
            adj[ rna_id[mytuple[0]]]={}
        if mytuple[1] not in protein_id:
            protein_id[mytuple[1]]=len(protein_id)
        adj[rna_id[mytuple[0]]][protein_id[mytuple[1]]] = 1
    for mytuple in test_set:
        if mytuple[0] not in rna_id:
            continue
        if mytuple[1] not in protein_id:
            continue
        adj[rna_id[mytuple[0]]][protein_id[mytuple[1]]] = 1

    print(len(test_set))
    print("rna_number: ", len(rna_id))
    print("protein_number: ", len(protein_id))
    with codecs.open("dataset/NPInter2_55/train.txt", "w", encoding="utf-8") as fw:
        for mytuple in train_set:

            fw.write("{}\t{}\t{}\n".format(rna_id[mytuple[0]], protein_id[mytuple[1]], int(mytuple[2])))

    with codecs.open("dataset/NPInter2_55/test.txt", "w", encoding="utf-8") as fw:
        for mytuple in test_set:
            if mytuple[0] not in rna_id:
                continue
            if mytuple[1] not in protein_id:
                continue
            fw.write("{}\t{}\t{}\n".format(rna_id[mytuple[0]], protein_id[mytuple[1]], int(mytuple[2])))

            neg=random.randint(0,len(protein_id)-1)
            while adj[rna_id[mytuple[0]]].get(neg,"0") == 1:
                neg = random.randint(0, len(protein_id)-1)
            fw.write("{}\t{}\t{}\n".format(rna_id[mytuple[0]], neg, 0))

    with codecs.open("test_T.txt", "w", encoding="utf-8") as fw:
        neg1 = dict(zip(protein_id.values(),protein_id.keys()))
        for mytuple in test_set:
            if mytuple[0] not in rna_id:
                continue
            if mytuple[1] not in protein_id:
                continue
            fw.write("{}\t{}\t{}\n".format(mytuple[0], mytuple[1], int(mytuple[2])))

            neg=random.randint(0,len(protein_id)-1)
            while adj[rna_id[mytuple[0]]].get(neg,"0") == 1:
                neg = random.randint(0, len(protein_id)-1)
            fw.write("{}\t{}\t{}\n".format(mytuple[0], neg1[neg], 0))
data = pd.read_excel('dataset/NPInter2.xlsx')
data = np.array(data)
divide_dataset(5, data)