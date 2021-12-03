# -*- coding:utf-8 -*-
import os
import time
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
import pickle
import argparse
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
from sklearn.model_selection import KFold
import Graph4DIV
import data_process as DP
import evaluate as EV
from divtype import *
MAXDOC = 50
REL_LEN = 18


def set_seed(seed = 0):
    '''
    some cudnn methods can be random even after fixing the seed
    unless you tell it to be deterministic
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def list_pairwise_loss(score_1, score_2, delta):
    acc = torch.Tensor.sum((score_1-score_2)>0).item()/float(score_1.shape[0])
    loss = -torch.sum(delta * torch.Tensor.log(1e-8+torch.sigmoid(score_1 - score_2)))/float(score_1.shape[0])
    return acc, loss


def build_graph_file(EMB_LEN, EMB_TYPE):
    DP.build_test_graph(EMB_LEN, EMB_TYPE)
    DP.multiprocessing_build_fold_training_graph(EMB_LEN, EMB_TYPE)


def run(BATCH_SIZE, EPOCH, LR, DROPOUT, EMB_LEN, EMB_TYPE, COMMENT):
    tmp_dir = './tmp/'+str(COMMENT)+'_Batch_size_'+str(BATCH_SIZE)+'_EPOCH_'+str(EPOCH)+'_LR_'+str(LR)+'_DROP_'+str(DROPOUT)+'_EMB_'+str(EMB_TYPE)+'/'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    ''' load randomly shuffled queries '''
    all_qids = np.load('./data/gcn_dataset/all_qids.npy')
    test_graph_path = './data/gcn_dataset/'+str(EMB_TYPE)+'_test_graph.data'
    test_graph_dict = pickle.load(open(test_graph_path, 'rb'))
    qd = pickle.load(open('./data/gcn_dataset/div_query.data', 'rb'))
    final_metrics = []
    best_model_list = []
    fold_time = 0
    test_qids_list = []
    for train_ids, test_ids in KFold(5).split(all_qids):
        fold_epoch_list = range(1, EPOCH+1)
        fold_time += 1
        print('Fold = ', fold_time)
        train_ids.sort()
        test_ids.sort()
        train_qids = [str(all_qids[i]) for i in train_ids]
        test_qids = [str(all_qids[i]) for i in test_ids]
        test_qids_list.append(test_qids)

        graph_data_loader = DP.get_fold_loader(fold_time, train_qids, BATCH_SIZE, EMB_TYPE)
        model = Graph4DIV.Graph4Div(node_feature_dim = EMB_LEN, hidden_dim = [EMB_LEN, EMB_LEN], output_dim = EMB_LEN, dropout = DROPOUT)
        if torch.cuda.is_available():
            model = model.cuda()
        opt = torch.optim.Adam(model.parameters(), lr = LR)
        params = list(model.parameters())
        if fold_time == 1:
            print('model = ', model)
            print(len(params))
            for param in params:
                print(param.size())
            n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
            print('* number of parameters: %d' % n_params)

        all_steps = len(graph_data_loader)
        max_metric = 0
        patience = 0
        best_model = ""
        for epoch in range(EPOCH):
            print('Start Training...')
            model.train()
            for step, train_data in enumerate(tqdm(graph_data_loader, desc = 'BATCH', ncols=80)):
                A, feat, rel_feat, degree_feat, pos_mask, neg_mask, w = train_data
                if torch.cuda.is_available():
                    A = A.cuda()
                    feat = feat.cuda()
                    rel_feat = rel_feat.cuda()
                    degree_feat = degree_feat.cuda()
                    pos_mask = pos_mask.cuda()
                    neg_mask = neg_mask.cuda()
                    w = w.cuda()
                score_1, score_2 = model(A, feat, rel_feat, degree_feat, pos_mask, neg_mask, True)
                acc, loss = list_pairwise_loss(score_1, score_2, w)
                opt.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm = 1)
                opt.step()
                if (step + 1) % (all_steps // 10) == 0:
                    model.eval()
                    metrics = []
                    for qid in test_qids:
                        metric = EV.get_metric_nDCG_random(model, test_graph_dict[str(qid)], qd[str(qid)], str(qid))
                        metrics.append(metric)
                    avg_alpha_NDCG = np.mean(metrics)
                    if max_metric < avg_alpha_NDCG:
                        max_metric = avg_alpha_NDCG
                        tqdm.write('max avg_alpha_NDCG updated: {}'.format(max_metric))
                        model_filename = 'model/TOTAL_EPOCH_' + str(EPOCH)+'_FOLD_' + str(fold_time) + '_EPOCH_'+str(epoch) + '_LR_'+str(LR)+'_BATCHSIZE_' + str(BATCH_SIZE) + '_DROPOUT_' + str(DROPOUT) + '_'+str(EMB_TYPE) + '_alpha_NDCG_' + str(max_metric) + '.pickle'
                        torch.save(model.state_dict(), model_filename)
                        tqdm.write('save file at: {}'.format(model_filename))
                        best_model = model_filename
                        patience = 0
                    else: 
                        patience += 1
                    model.train()
                    if epoch > 0 and patience > 2:
                        new_lr = 0.0
                        for param_group in opt.param_groups:
                            param_group['lr'] = param_group['lr'] * 0.5
                            new_lr = param_group['lr'] 
                        patience = 0
                        tqdm.write("decay lr: {}, load model: {}".format(new_lr, best_model))
            model.eval()
            metrics = []
            for qid in test_qids:
                metric= EV.get_metric_nDCG_random(model, test_graph_dict[str(qid)], qd[str(qid)], str(qid))
                metrics.append(metric)
            avg_alpha_NDCG = np.mean(metrics)
            if max_metric < avg_alpha_NDCG:
                max_metric = avg_alpha_NDCG
                tqdm.write('max avg_alpha_NDCG updated: {}'.format(max_metric))
                model_filename = 'model/TOTAL_EPOCH_'+str(EPOCH)+'_FOLD_'+str(fold_time)+'_EPOCH_'+str(epoch)+'_LR_'+str(LR)+'_BATCHSIZE_'+str(BATCH_SIZE)+'_DROPOUT_'+str(DROPOUT)+'_'+str(EMB_TYPE)+'_alpha_NDCG_'+str(max_metric)+'.pickle'
                torch.save(model.state_dict(), model_filename)
                tqdm.write('save file at: {}'.format(model_filename))
                best_model = model_filename
            if epoch == (EPOCH-1):
                final_metrics.append(max_metric)
                best_model_list.append(best_model)
    
    print('final list = {}'.format(final_metrics))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--mode', type = str, default = "load_std_models", help = "run mode")
    parser.add_argument('--batchsize', type = int, default = 16, help = "the batch size")
    parser.add_argument('--epoch', type = int, default = 3, help = "the training epoches")
    parser.add_argument('--lr', type = float, default = 8e-4, help = "Which learning rate to start with. (Default: 1e-3)")
    parser.add_argument('--dropout', type = float, default = 0.5, help = "the dropout rate")
    parser.add_argument('--device', type = str, default = "1", help = "GPU ID")
    parser.add_argument('--comment', type = str, default = "std_test", help = "run comment")
    parser.add_argument('--fold', type = int, default = 1, help = "the training fold for classifier")

    set_seed()
    args = parser.parse_args()
    mode = args.mode
    BATCH_SIZE = args.batchsize
    EPOCH = args.epoch
    LR = args.lr
    DROPOUT = args.dropout
    EMB_LEN = 100
    EMB_TYPE = 'doc2vec'
    DEVICE = args.device
    COMMENT = args.comment
    fold = args.fold
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE)
    if mode == "data_preprocess":
        ''' data preprocess : /gcn_dataset/best_rank/'''
        DP.data_process()
    elif mode == "load_query":
        ''' generate query data files : div_query.data'''
        DP.generate_qd()
    elif mode == "gen_train_data":
        ''' generate Training Datasets : listpair_train.data '''
        D = div_dataset()
        D.get_listpair_train_data()
    elif mode == "get_intent_cover":
        ''' generate document pair coverage file : intent_coverage.csv '''
        DP.get_intent_coverage()
    elif mode == "get_doc_tokens":
        ''' documents tokenization for relation classifier training '''
        DP.document_process()
    elif mode == "divide_clf_dataset":
        ''' divide cover data to 5 folds :  ./data/clf_cover_data/ '''
        DP.divide_five_fold_train_test()
    elif mode == "make_dataset_clf":
        ''' make classifier dataset for corresponding fold '''
        DP.make_data_set("train", fold)
        DP.make_data_set("test", fold)
    elif mode == "build_graph":
        ''' build graph for training and testing '''
        build_graph_file(EMB_LEN, EMB_TYPE)
    elif mode == "train":
        ''' trian Graph4DIV model '''
        run(BATCH_SIZE, EPOCH, LR, DROPOUT, EMB_LEN, EMB_TYPE, COMMENT)
    else:
        print('Incorrect Command!')

