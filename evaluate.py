import os
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import pickle
import Graph4DIV
import torch
from data_process import *

MAXDOC = 50
REL_LEN = 18


def adjust_graph(A, rel_score_list, degree_tensor, selected_doc_id):
    '''
    adjust adjancent matrix A during the testing process, set the selected doc degree = 0
    :param rel_score_list: initial relevance of the document
    :param degree_tensor: degree tensor of each document
    :return: adjacent matrix A, degree tensor
    '''
    ''' connect selected document to the query node '''
    A[0, selected_doc_id+1, 0] = rel_score_list[selected_doc_id]
    A[0, 0, selected_doc_id+1] = rel_score_list[selected_doc_id]
    ''' remove edges between selected document and candidates '''
    A[0, selected_doc_id+1, 1:] = torch.tensor([0.0]*50).float()
    A[0, 1:, selected_doc_id+1] = torch.tensor([0.0]*50).float()
    ''' set the degree of selected document '''
    degree_tensor[0, selected_doc_id] = torch.tensor(0.0)
    return A, degree_tensor


def get_metric_nDCG_random(model, test_tuple, div_q, qid):
    '''
    get the alpha-nDCG for the input query, the input document list are randomly shuffled. 
    :param test_tuple: the features of the test query qid, test_turple = (feature, index, rel_feat, rel_score, A, degree)
    :param div_q: the div_query object of the test query qid
    :param qid: the id for the test query
    :return: the alpha-nDCG for the test query
    '''
    metric = 0
    end = Max_doc_num = len(div_q.best_docs_rank)
    current_docs_rank = []
    if not test_tuple:
        return 0 
    else:
        feature = test_tuple[0]
        index = test_tuple[1]
        rel_feat_tensor = torch.tensor(test_tuple[2]).float()
        rel_score_list = test_tuple[3]
        A = test_tuple[4]
        degree_tensor = test_tuple[5]

        A.requires_grad = False
        degree_tensor.requires_grad = False
        rel_feat_tensor.requires_grad = False
        lt = len(rel_score_list)
        if lt < MAXDOC:
            rel_score_list.extend([0.0]*(MAXDOC-lt))
        rel_score = torch.tensor(rel_score_list).float()
        
        A = A.reshape(1, A.shape[0], A.shape[1])
        feature = feature.reshape(1, feature.shape[0], feature.shape[1])
        rel_feat_tensor = rel_feat_tensor.reshape(1, rel_feat_tensor.shape[0], rel_feat_tensor.shape[1])
        degree_tensor = degree_tensor.reshape(1, degree_tensor.shape[0], degree_tensor.shape[1])
        
        if th.cuda.is_available():
            A = A.cuda()
            feature = feature.cuda()
            rel_feat_tensor = rel_feat_tensor.cuda()
            degree_tensor = degree_tensor.cuda()
        
        while len(current_docs_rank)<Max_doc_num:
            outputs = model(A, feature, rel_feat_tensor, degree_tensor)
            out = outputs.cpu().detach().numpy()
            result = np.argsort(-out[:end])

            for i in range(len(result)):
                if result[i] < Max_doc_num and index[result[i]] not in current_docs_rank:
                    current_docs_rank.append(index[result[i]])
                    adjust_index = result[i]
                    break
            A, degree_tensor = adjust_graph(A, rel_score, degree_tensor, adjust_index)

        if len(current_docs_rank)>0:
            new_docs_rank = [div_q.doc_list[i] for i in current_docs_rank]
            metric = div_q.get_test_alpha_nDCG(new_docs_rank)
    return metric

        
def evaluate_accuracy(y_pred, y_label):
    num = len(y_pred)
    all_acc = 0.0
    count = 0
    for i in range(num):
        pred = (y_pred[i] > 0.5).astype(int)
        label = y_label[i]
        acc = 1 if pred == label else 0
        all_acc += acc
        count += 1
    return all_acc / count
