from sklearn.metrics import roc_curve,roc_auc_score,average_precision_score,precision_recall_curve,precision_score
from sklearn import metrics


from sklearn.linear_model import LogisticRegression
import numpy as np

import os
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
from model.bihognn import BIHOTrainer
from loader import DataLoader
from model.Rnaprotein import Rnaprotein

import matplotlib.pyplot as plt

import pandas as pd


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='NPInter2_55')

parser.add_argument('--feature_dim', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay', type=float, default=0.9)
parser.add_argument('--decay_epoch', type=int, default=5)
parser.add_argument('--leakey', type=float, default=0.1)
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--alpha', type=float, default=0.9)
parser.add_argument('--num_epoch', type=int, default=10)
parser.add_argument('--min_neighbor', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--seed', type=int, default=2040)
parser.add_argument('--negative', type=int, default=1)




def result(X_train,y_train,X_test,y_test):
    lg = LogisticRegression(penalty='l2', C=0.001,max_iter=500)
    lg.fit(X_train,y_train)
    lg_y_pred_est = lg.predict_proba(X_test)[:,1]
    pred = lg.predict(X_test)
    fpr,tpr,thresholds = metrics.roc_curve(y_test,lg_y_pred_est)
    average_precision = average_precision_score(y_test, lg_y_pred_est)
    recall = metrics.recall_score(y_test,pred)
    f1 = metrics.f1_score(y_test,pred)
    pr = precision_score(y_test,pred)
    return metrics.auc(fpr,tpr), average_precision, recall, pr, f1, lg_y_pred_est.tolist(), y_test,pred

def concat_feature(feature, id, cuda):
    x = id[:, 0]
    y = id[:, 1]
    x = torch.LongTensor(x)
    y = torch.LongTensor(y)
    if cuda:
        x = x.cuda()
        y = y.cuda()

    x = torch.index_select(feature, 0, x)  
    y =torch.index_select(feature, 0, y) 
    ans = torch.cat((x, y), dim=1)
    return x, y, ans
def ran(seed=1111):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


args = parser.parse_args()
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)
init_time = time.time()
opt = vars(args)
ran(opt["seed"])
opt['dataset'] = 'dataset/' + opt['dataset']+'/'
G = Rnaprotein(opt,"train.txt")
UV = G.UV
VU = G.VU
adj = G.adj
rna_real_dict = G.rna_real_dict
protein_real_dict = G.protein_real_dict
all_edges = G.all_edges



rna_feature = np.random.randn(opt["number_rna"], opt["feature_dim"])
protein_feature = np.random.uniform(-1, 1, (opt["number_protein"], opt["feature_dim"]))


rna_feature = torch.FloatTensor(rna_feature)
protein_feature = torch.FloatTensor(protein_feature)


if opt["cuda"]:
    rna_feature = rna_feature.cuda()
    protein_feature = protein_feature.cuda()
    UV = UV.cuda()
    VU = VU.cuda()
    adj = adj.cuda()


print("Loading data from {} with batch size {}...".format(opt['dataset'], opt['batch_size']))
train_batch = DataLoader(opt['dataset'] + 'train.txt', opt['batch_size'], opt,
                         rna_real_dict,protein_real_dict, evaluation=False)
dev_batch = DataLoader(opt['dataset'] + 'test.txt', 1000000, opt, rna_real_dict,protein_real_dict, evaluation=True)


trainer = BIHOTrainer(opt)

dev_score_history = []
current_lr = opt['lr']
global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
max_steps = len(train_batch) * opt['num_epoch']




link_dataset = []
link_label = []
for i, batch in enumerate(train_batch):
    rna_index = batch[0].cpu().detach().numpy()
    protein_index = batch[1].cpu().detach().numpy()
    negative_protein_index = batch[2].cpu().detach().numpy()
    for j in range(len(rna_index)):
        link_dataset.append([rna_index[j],protein_index[j] + opt["number_rna"]])
        link_dataset.append([rna_index[j], negative_protein_index[j] + opt["number_rna"]])
        link_label.append(1)
        link_label.append(0)

link_dataset_test = []
link_label_test = []
for i, batch in enumerate(dev_batch):
    rna_index = batch[0].cpu().detach().numpy()
    protein_index = batch[1].cpu().detach().numpy()
    link_label_test = batch[2]
    for j in range(len(rna_index)):
        link_dataset_test.append([rna_index[j],protein_index[j]])
link_dataset = np.array(link_dataset)
link_dataset_test = np.array(link_dataset_test)



print("train set:",len(link_dataset))
print("test set:",len(link_dataset_test))

all_score = []
for epoch in range(1, opt['num_epoch'] + 1):
    train_loss = 0
    start_time = time.time()
    for i, batch in enumerate(train_batch):
        global_step += 1
        loss = trainer.reconstruct(UV, VU, adj, all_edges, rna_feature, protein_feature, batch) 
        train_loss += loss
    duration = time.time() - start_time
    print(format_str.format(datetime.now(), global_step, max_steps, epoch, \
                            opt['num_epoch'], train_loss / len(train_batch), duration, current_lr))
    print("batch_rec_loss: ", sum(trainer.epoch_rec_loss) / len(trainer.epoch_rec_loss))



    print("Evaluating on dev set...")

    trainer.model.eval()
    trainer.update_bipartite(rna_feature, protein_feature, UV, VU, adj)

    rna_hidden_out = trainer.rna_hidden_out
    protein_hidden_out = trainer.protein_hidden_out

    print('rna_hidden_out', rna_hidden_out.size())
    bi_feature = torch.cat((rna_hidden_out, protein_hidden_out),dim=0)


    train_x, train_y, train_feature = concat_feature(bi_feature, link_dataset, opt["cuda"])
    test_x, test_y, test_feature = concat_feature(bi_feature, link_dataset_test, opt["cuda"])
    auc_roc, auc_pr, recall, pr, f1,  predict_label, y_true,pred = result(train_feature.cpu().detach().numpy(), link_label,
                                               test_feature.cpu().detach().numpy(), link_label_test)
    train_loss = train_loss / train_batch.num_examples * opt['batch_size']  # avg loss per batch
    print("epoch {}: train_loss = {:.6f}, auc_roc = {:.6f}, auc_pr = {:.6f}, recall = {:.6f}, pr = {:.6f}, f1 = {:.6f}".format(epoch, \
                                                                                    train_loss, auc_roc, auc_pr, recall, pr, f1))
    dev_score = auc_roc
    
    all_score.append([train_loss, auc_roc, auc_pr, recall, pr, f1])

    if epoch == 1 or dev_score > max(dev_score_history):

        predict,y_true_bi,pred =  predict_label, y_true,pred
    if len(dev_score_history) > opt['decay_epoch'] and dev_score <= dev_score_history[-1] and \
            opt['optim'] in ['sgd', 'adagrad', 'adadelta']:
        current_lr *= opt['lr_decay']
        trainer.update_lr(current_lr)

    dev_score_history += [dev_score]





print("Training ended with {} epochs.".format(epoch))

all_score = np.array(all_score)
arg = np.argmax(all_score[:,1], axis=0)
best_score = all_score[arg][1:]

best_score = pd.DataFrame(best_score)

writer = pd.ExcelWriter('result.xlsx')		# 写入Excel文件
best_score = best_score.stack().unstack(0)
best_score.to_excel(writer, 'page_1',header = ['AUC ', 'AUC_PR ', 'Recall ', 'PR', 'F1'], float_format='%.5f')
writer.save()
if 'NPInter_55' in opt['dataset']:

    fs = 16


    y1 = np.loadtxt('./baselines/LPBNI.txt',delimiter=',', encoding='UTF-8-sig').flatten()
    y2 = np.loadtxt('./baselines/LPIHN.txt',delimiter=',', encoding='UTF-8-sig').flatten()
    y3 = np.loadtxt('./baselines/LPIIBNRA.txt',delimiter=',', encoding='UTF-8-sig').flatten()
    y4 = np.loadtxt('./baselines/RWR.txt',delimiter=',', encoding='UTF-8-sig').flatten()
    y5 = np.loadtxt('./baselines/LPISKF.txt',delimiter=',', encoding='UTF-8-sig').flatten()
    y6 = np.loadtxt('./baselines/LPIGAC.txt',delimiter=',', encoding='UTF-8-sig').flatten()
    y7 = predict
    ytrue = np.loadtxt('./baselines/interaction.txt',delimiter=',', encoding='UTF-8-sig').flatten()


    AUC_ROC_y1 = roc_auc_score(ytrue,y1)
    AUC_ROC_y2 = roc_auc_score(ytrue,y2)
    AUC_ROC_y3 = roc_auc_score(ytrue,y3)
    AUC_ROC_y4 = roc_auc_score(ytrue,y4)
    AUC_ROC_y5 = roc_auc_score(ytrue,y5)
    AUC_ROC_y6 = roc_auc_score(ytrue,y6)
    AUC_ROC_y7 = roc_auc_score(y_true_bi,y7)

    fpr1,tpr1,rocth1 = roc_curve(ytrue,y1)
    fpr2,tpr2,rocth2 = roc_curve(ytrue,y2)
    fpr3,tpr3,rocth3 = roc_curve(ytrue,y3)
    fpr4,tpr4,rocth4 = roc_curve(ytrue,y4)
    fpr5,tpr5,rocth5 = roc_curve(ytrue,y5)
    fpr6,tpr6,rocth6 = roc_curve(ytrue,y6)
    fpr7,tpr7,rocth7 = roc_curve(y_true_bi,y7)

    precision1,recall1,prth1 = precision_recall_curve(ytrue,y1)
    precision2,recall2,prth2 = precision_recall_curve(ytrue,y2)
    precision3,recall3,prth3 = precision_recall_curve(ytrue,y3)
    precision4,recall4,prth4 = precision_recall_curve(ytrue,y4)
    precision5,recall5,prth5 = precision_recall_curve(ytrue,y5)
    precision6,recall6,prth6 = precision_recall_curve(ytrue,y6)
    precision7,recall7,prth7 = precision_recall_curve(y_true_bi,y7)


    PR_y1 = average_precision_score(ytrue,y1)
    PR_y2 = average_precision_score(ytrue,y2)
    PR_y3 = average_precision_score(ytrue,y3)
    PR_y4 = average_precision_score(ytrue,y4)
    PR_y5 = average_precision_score(ytrue,y5)
    PR_y6 = average_precision_score(ytrue,y6)
    PR_y7 = average_precision_score(y_true_bi,y7)


    # fig,ax = plt.subplots(1,2,figsize=(13,6),dpi=100)
    # plt.rcParams.update({"font.size":fs})

    roc_curve = plt.figure()


    plt.plot(fpr7,tpr7,'-',color='orangered',label='BiHo-GNN(%0.3f)' % AUC_ROC_y7)
    plt.plot(fpr6,tpr6,'-.', color='orange',label='LPIGAC(%0.3f)' % AUC_ROC_y6)
    plt.plot(fpr5,tpr5,'-.', color='green',label='LPISKF(%0.3f)' % AUC_ROC_y5)
    plt.plot(fpr4,tpr4,'-.', color='blue',label='RWR(%0.3f)' % AUC_ROC_y4)
    plt.plot(fpr1,tpr1,'-.', color='lightseagreen',label='LPBNI(%0.3f)' % AUC_ROC_y1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    # ax0.tick_params(labelsize=fs)
    plt.savefig('auc.eps',format='eps')
    plt.show()

    pr_curve = plt.figure()

    plt.plot(recall7,precision7,'-', color='orangered',label='BiHo-GNN(%0.3f)' % PR_y7)
    plt.plot(recall6,precision6,'-.', color='orange',label='LPIGAC(%0.3f)' % PR_y6)
    plt.plot(recall5,precision5,'-.', color='green',label='LPISKF(%0.3f)' % PR_y5)
    plt.plot(recall4,precision4,'-.', color='blue',label='RWR(%0.3f)' % PR_y4)

    plt.plot(recall1,precision1, '-.', color='lightseagreen',label='LPBNI(%0.3f)' % PR_y1)
    plt.ylim([-0.01, 1.01])
    plt.xlim([-0.01, 1.01])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curves')
    plt.legend(loc="lower left")


    # ax1.legend(loc='lower left')
    # plt.tight_layout()
    # plt.savefig('auc.pdf')
    # #plt.savefig('auc5cv2.eps')
    plt.savefig('pr.eps',format='eps')

    plt.show()

    # def Specificity(y_true, y_pred):
    #     tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    #     return tn/(tn+fp)

    # def NPV(y_true, y_pred):
    #     tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    #     return tn/(tn+fn)
    # def test_result(y_score):
    #     y_test=ytrue
    #     result = []
    #     y_pred_net = y_score
    #     for i in range(len(y_pred_net)):
    #         if y_pred_net[i] >= 0.5:
    #             result.append(1)
    #         else:
    #             result.append(0)
    #     result = np.array(result)
    #     print(metrics.classification_report(y_test, result))
    #     print(metrics.confusion_matrix(y_test, result))
    #     print('AUROC:%0.3f' %  metrics.roc_auc_score(y_test, y_pred_net))
    #     print('AUPR:%0.3f'% metrics.average_precision_score(y_test, y_pred_net))
    #     # print('MCC:%0.3f' %  metrics.matthews_corrcoef(y_test, result))
    #     # print('Accuracy:%0.3f' % metrics.accuracy_score(y_test, result))
    #     # print('Specificity:%0.3f'% Specificity(y_test, result))
    #     print('Recall:%0.3f' % metrics.recall_score(y_test, result))
    #     # print('NPV:%0.3f' % NPV(y_test, result))
    #     print('Precision:%0.3f' %  metrics.precision_score(y_test, result))
    #     print('F1:%0.3f' % metrics.f1_score(y_test, result))
    #     print('================================================================')
    # y=[y1,y2,y3,y4,y5,y6,y7]
    # for i in range(len(y)):
    #     test_result(y[i])
