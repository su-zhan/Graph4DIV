import os
import math
import copy
import time
import gzip
import pickle
import random
import torch
import torch as th
import pandas as pd
import numpy as np
import multiprocessing
from scipy import stats
import xml.dom.minidom
from xml.dom.minidom import parse
from transformers import BertTokenizer
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from divtype import *
from sklearn.model_selection import KFold

MAXDOC = 50
REL_LEN = 18


def load_emb(filename, sep = '\t'):
    '''
    load embedding from file
    :param filename: embedding file name
    :param sep: the char used as separation symbol
    :return: a dict with item name as key and embedding vector as value
    '''
    with open(filename, 'r') as fp:
        result = {}
        for l in fp:
            l = l.strip()
            if l == '':
                continue
            sp = l.split(sep)
            vals = [float(sp[i]) for i in range(1, len(sp))]
            result[sp[0]] = vals
        return result


def get_rel_feat():
    rel_feat = pd.read_csv('./data/gcn_dataset/rel_feat.csv')
    rel_feat_names = list(sorted(set(rel_feat.columns) - {'query', 'doc'}))
    rel_feat[rel_feat_names] = StandardScaler().fit_transform(rel_feat[rel_feat_names])
    rel_feat = dict(zip(map(lambda x: tuple(x), rel_feat[['query', 'doc']].values),
    rel_feat[rel_feat_names].values.tolist()))
    return rel_feat


def get_clf_train_coverage(fold):
    fold_judge_coverage = './data/gcn_dataset/clf_cover_result/fold'+str(fold)+'_train_clf.csv'
    f = open(fold_judge_coverage, 'r')
    cover_dict = {}
    count = 0
    for line in f:
        if count == 0:
            count += 1
            continue
        line = line.strip('\n')
        qid, docx, docy, c = line.split(',')
        cover_dict[(qid, docx, docy)] = int(c)
    return cover_dict


def get_clf_test_coverage():
    cover_dict = {}
    for fold in range(1,6):
        filename = './data/gcn_dataset/clf_cover_result/fold' + str(fold) + '_test_clf.csv'
        f = open(filename, 'r')
        count = 0
        for line in f:
            if count == 0:
                count += 1
                continue
            line = line.strip('\n')
            qid, docx, docy, c = line.split(',')
            if (qid, docx, docy) in cover_dict:
                print('error:', (qid, docx, docy))
            cover_dict[(qid, docx, docy)] = int(c)
    return cover_dict


def split_list(origin_list, n):
    res_list = []
    L = len(origin_list)
    N = int(math.ceil(L / float(n)))
    begin = 0
    end = begin + N
    while begin < L:
        if end < L:
            temp_list = origin_list[begin:end]
            res_list.append(temp_list)
            begin = end
            end += N
        else:
            temp_list = origin_list[begin:]
            res_list.append(temp_list)
            break
    return res_list


def build_training_graph(qid_list, res_dir, qd, rel_feat_dict, train_data, doc_emb, query_emb, cover_feat_dict, EMB_LEN):
    for qid in tqdm(qid_list, desc = 'GenTrainGraph'):
        output_file_path = res_dir+str(qid)+'.pickle'
        graph_dict = {}
        graph_dict[qid] = []
        if qid not in train_data:
            pickle.dump(graph_dict, open(output_file_path, 'wb'), True)
            continue
        for item in train_data[qid]:
            label = list([float(i) for i in item[0]])
            if len(label) == 0:
                print('No training data for query #{}!'.format(qid))
                continue
            temp_q = qd[str(qid)]
            query = temp_q.query
            doc_list = temp_q.best_docs_rank#to check
            doc_rel_score_list = temp_q.best_docs_rank_rel_score
            lt = len(doc_rel_score_list)
            if lt < MAXDOC:
                doc_rel_score_list.extend([0.0]*(MAXDOC-lt))
            X_q = query_emb[query]
            node_feat = []
            node_feat.append(X_q)
            rel_feat_list = []
            ''' Build Adjacent Matrix for the Query '''
            A = np.zeros((MAXDOC+1, MAXDOC+1), dtype = float)
            for i in range(len(doc_list)):
                for j in range(i+1, len(doc_list)):
                    docx_id = doc_list[i]
                    docy_id = doc_list[j]
                    ''' selected document will not be connected to the candidate documents '''
                    if label[i] < 0 or label[j] < 0:
                        continue
                    try:
                        c = cover_feat_dict[(qid, str(docx_id), str(docy_id))]
                        A[i+1][j+1] = c
                        A[j+1][i+1] = c
                    except:
                        try:
                            c = cover_feat_dict[(qid, str(docy_id), str(docx_id))]
                            A[i+1][j+1] = c
                            A[j+1][i+1] = c
                        except:
                            print('No coverage data! qid = {}, i = {}, j = {}, docx = {}, docy = {}'.format(qid, i, j, docx_id, docy_id))
            ''' get degree feature of each document '''
            sub_A = A[1:, 1:]
            degree_list = []
            for i in range(sub_A.shape[0]):
                degree = np.sum(sub_A[i, :])
                degree_list.append(degree)
            degree_tensor = torch.tensor(degree_list).float().reshape(sub_A.shape[0], 1)

            ''' get node features '''
            for i in range(MAXDOC):
                if i < len(doc_list):
                    doc_label = 1 if label[i] < 0 else 0
                    if doc_label == 1:
                        ''' for the selected document '''
                        A[0][i+1] = doc_rel_score_list[i]
                        A[i+1][0] = doc_rel_score_list[i]
                        X_doc = doc_emb[doc_list[i]]
                        rel_feat = rel_feat_dict[(query, doc_list[i])]
                    else:
                        ''' for the candidate document '''
                        X_doc = doc_emb[doc_list[i]]
                        rel_feat = rel_feat_dict[(query, doc_list[i])]
                else:
                    ''' for the padding document '''
                    X_doc = [0]*EMB_LEN
                    rel_feat = [0]*REL_LEN
                node_feat.append(X_doc)
                rel_feat_list.append(rel_feat)
            A = torch.tensor(A).float()
            ''' Graph_dict[qid]=[(feature_tensor,weight,relevant_features,pos_mask,neg_mask,Adj_Matrix,degree_tensor)]'''
            feature_tensor = th.tensor(node_feat).float()
            weight = item[3]
            pos_mask = item[1].clone().detach()
            neg_mask = item[2].clone().detach()
            graph_dict[qid].append((feature_tensor, weight, rel_feat_list, pos_mask, neg_mask, A, degree_tensor))
        pickle.dump(graph_dict, open(output_file_path, 'wb'), True)


def multiprocessing_build_fold_training_graph(EMB_LEN, EMB_TYPE, worker_num = 4):
    ''' generate 5 fold training graph '''
    res_dir = './data/' + str(EMB_TYPE) + '_fold_training_graph/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    rel_feat_dict = get_rel_feat()
    all_qids = np.load('./data/gcn_dataset/all_qids.npy')
    qd = pickle.load(open('./data/gcn_dataset/div_query.data', 'rb'))
    train_data = pickle.load(open('./data/gcn_dataset/listpair_train.data', 'rb'))
    if EMB_TYPE == 'doc2vec':
        doc_emb = load_emb('./data/gcn_dataset/doc2vec_doc.emb')
        query_emb = load_emb('./data/gcn_dataset/doc2vec_query.emb')
    fold = 0
    for train_ids, test_ids in KFold(5).split(all_qids):
        fold += 1
        cover_feat_dict = get_clf_train_coverage(fold)
        if not cover_feat_dict:
            print('Error! Intent coverage fold {} file not found'.format(fold))
            continue
        fold_dir = res_dir + 'fold_' + str(fold) + '/'
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
        train_ids.sort()
        train_qids = [str(all_qids[i]) for i in train_ids]
        task_list = split_list(train_qids, worker_num)
        jobs = []
        for task in task_list:
            p = multiprocessing.Process(target = build_training_graph, args = (task, fold_dir, qd, rel_feat_dict, train_data, doc_emb, query_emb, cover_feat_dict, EMB_LEN))
            jobs.append(p)
            p.start()
    print('training graph generation done!')


def build_test_graph(EMB_LEN, EMB_TYPE):
    ''' Build the test Graph for testing '''
    graph_dict = {}
    qid_list = []
    rel_feat_dict = get_rel_feat()
    cover_feat_dict = get_clf_test_coverage()
    qd = pickle.load(open('./data/gcn_dataset/div_query.data', 'rb'))
    output_path = './data/gcn_dataset/' + str(EMB_TYPE) + '_test_graph.data'
    if EMB_TYPE == "doc2vec":
        doc_emb = load_emb('./data/gcn_dataset/doc2vec_doc.emb')
        query_emb = load_emb('./data/gcn_dataset/doc2vec_query.emb')
    ''' remove query #95 and #100 '''
    all_qids = range(1, 201)
    del_index = [94, 99]
    all_qids = np.delete(all_qids, del_index)
    qids = [str(i) for i in all_qids]

    for qid in tqdm(qids, desc = "Gen Test Graph"):
        temp_q = qd[str(qid)]
        ''' use the top 50 relevant document for diversity ranking '''
        doc_list = temp_q.doc_list[:MAXDOC]
        rel_score_list = temp_q.doc_score_list[:MAXDOC]
        real_num = len(doc_list)
        label = list([0] * real_num)
        index = [i for i in range(real_num)]
        ''' The document list for testing is randomly shuffled'''
        c = list(zip(doc_list, rel_score_list, index))
        random.shuffle(c)
        doc_list[:], rel_score_list[:], index[:] = zip(*c)
        ''' padding '''
        if len(label) == 0:
            continue
        elif len(label)<MAXDOC:
            label.extend([0] * (MAXDOC-len(label)))
        ''' get the node features '''
        query = temp_q.query
        X_q = query_emb[query]
        node_feat = []
        node_feat.append(X_q)
        rel_feat_list = []
        for i in range(MAXDOC):
            if i < real_num:
                X_doc = doc_emb[doc_list[i]]
                rel_feat = rel_feat_dict[(query, doc_list[i])]
                rel_feat_list.append(rel_feat)
            else:
                X_doc = [0]*EMB_LEN
                rel_feat = [0]*REL_LEN
                rel_feat_list.append(rel_feat)
            node_feat.append(X_doc)

        ''' Build Adjcent Matrix for the shuffled document list '''
        A = np.zeros((MAXDOC+1, MAXDOC+1), dtype = float)
        for i in range(len(doc_list)):
            for j in range(i+1, len(doc_list)):
                docx_id = doc_list[i]
                docy_id = doc_list[j]
                try:
                    c = cover_feat_dict[(str(qid), str(docx_id), str(docy_id))]
                    A[i+1][j+1] = c
                    A[j+1][i+1] = c
                except:
                    try:
                        c = cover_feat_dict[(str(qid), str(docy_id), str(docx_id))]
                        A[i+1][j+1] = c
                        A[j+1][i+1] = c
                    except:
                        print('ERROR: qid = {}, i = {}, j = {}, docx = {}, docy = {}'.format(qid, i, j, docx_id, docy_id))
        ''' get the degree features '''
        sub_A = A[1:, 1:]
        degree_list = []
        for i in range(sub_A.shape[0]):
            degree = np.sum(sub_A[i, :])
            degree_list.append(degree)
        degree_tensor = torch.tensor(degree_list).float().reshape(sub_A.shape[0], 1)

        node_feat = torch.tensor(node_feat).float()
        A = torch.tensor(A).float()
        ''' Graph_dict[qid] = (node_features, relevance_feature_list, relevance_score_list, Adj_Matrix, degree_tensor) '''
        graph_dict[qid] = (node_feat, index, rel_feat_list, rel_score_list, A, degree_tensor)
    pickle.dump(graph_dict, open(output_path, 'wb'), True)


def get_query_dict():
    dq_dict = {}
    topics_list = []
    for year in ['2009','2010','2011','2012']:
        filename = './data/clueweb_data/wt_topics/wt' + year + '.topics.xml'
        DOMTree = xml.dom.minidom.parse(filename)
        collection = DOMTree.documentElement
        topics = collection.getElementsByTagName("topic")
        topics_list.extend(topics)
    ''' load subtopics for each query '''
    for topic in topics_list:
        if topic.hasAttribute("number"):
            qid = topic.getAttribute("number")
        query = topic.getElementsByTagName('query')[0].childNodes[0].data
        subtopics = topic.getElementsByTagName('subtopic')
        subtopic_id_list = []
        subtopic_list = []
        for subtopic in subtopics:
            if subtopic.hasAttribute('number'):
                subtopic_id = subtopic.getAttribute('number')
                subtopic_id_list.append(subtopic_id)
            sub_query = subtopic.childNodes[0].data
            subtopic_list.append(sub_query)
        dq = div_query(qid, query, subtopic_id_list, subtopic_list)
        dq_dict[str(qid)] = dq
    return dq_dict


def get_docs_dict():
    ''' 
    get the relevance score of the documents 
    docs_dict[qid] = [doc_id, ...]
    docs_rel_score_dict[qid] = [score, ...]
    '''
    docs_dict = {}
    docs_rel_score_dict = {}
    for year in ['2009','2010','2011','2012']:
        filename = './data/clueweb_data/wt' + year + '.txt'
        f = open(filename)
        for line in f:
            qid, _, docid, _, score, _ = line.split(' ')
            if str(qid) not in docs_dict:
                docs_dict[str(qid)] = []
                docs_rel_score_dict[str(qid)] = []
            docs_dict[str(qid)].append(str(docid))
            docs_rel_score_dict[str(qid)].append(float(score))
    ''' Normalize the relevance score of the documents '''
    for qid in docs_rel_score_dict:
        temp_score_list = copy.deepcopy(docs_rel_score_dict[qid])
        for i in range(len(temp_score_list)):
            temp_score_list[i] = docs_rel_score_dict[qid][0]/docs_rel_score_dict[qid][i]
        docs_rel_score_dict[qid] = temp_score_list
    return docs_dict, docs_rel_score_dict


def get_doc_judge(qd, dd, ds):
    ''' 
    load document list and relevance socre list for the corresponding query 
    qd : query dictionary
    dd : document dictionary
    ds : document relevance score dictionary
    '''
    for key in qd:
        qd[key].add_docs(dd[key])
        qd[key].add_docs_rel_score(ds[key])
    for year in ['2009','2010','2011','2012']:
        filename = './data/clueweb_data/wt_judge/' + year + '.diversity.qrels'
        f = open(filename, 'r')
        for line in f:
            qid, subtopic, docid, judge = line.split(' ')
            judge = int(judge)
            if judge > 0:
                if str(docid) in qd[str(qid)].subtopic_df.index.values:
                    qd[str(qid)].subtopic_df[str(subtopic)][str(docid)] = 1
    return qd


def data_process_worker(task):
    for item in task:
        qid = item[0]
        dq = item[1]
        ''' get the best ranking for the top 50 relevant documents '''
        dq.get_best_rank(MAXDOC)
        pickle.dump(dq, open('./data/gcn_dataset/best_rank/'+str(qid)+'.data', 'wb'), True)


def calculate_best_rank(qd):
    data_dir = './data/gcn_dataset/best_rank/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    q_list = []
    for key in qd:
        x = copy.deepcopy(qd[key])
        q_list.append((str(key), x))
    jobs = []
    task_list = split_list(q_list, 8)
    for task in task_list:
        p = multiprocessing.Process(target = data_process_worker, args = (task, ))
        jobs.append(p)
        p.start()


def get_stand_best_metric(qd):
    ''' load best alpha-nDCG from DSSA '''
    std_dict = pickle.load(open('./data/gcn_dataset/stand_metrics.data', 'rb'))
    for qid in std_dict:
        m = std_dict[qid]
        target_q = qd[str(qid)]
        target_q.set_std_metric(m)


def data_process():
    ''' get subtopics for each query '''
    qd = get_query_dict()
    ''' get documents dictionary '''
    dd, ds = get_docs_dict()
    ''' get diversity judge for documents '''
    qd = get_doc_judge(qd, dd, ds)
    ''' get the stand best alpha-nDCG from DSSA '''
    get_stand_best_metric(qd)
    ''' get the best ranking for top n relevant documents and save as files'''
    calculate_best_rank(qd)


def generate_qd():
    ''' generate diversity_query file from data_dir '''
    data_dir = './data/gcn_dataset/best_rank/'
    files = os.listdir(data_dir)
    files.sort(key = lambda x:int(x[:-5]))
    query_dict = {}
    for f in files:
        file_path = os.path.join(data_dir, f)
        temp_q = pickle.load(open(file_path, 'rb'))
        query_dict[str(f[:-5])] = temp_q
    pickle.dump(query_dict, open('./data/gcn_dataset/div_query.data', 'wb'), True)
    return query_dict


def get_fold_loader(fold, train_ids, BATCH_SIZE, EMB_TYPE):
    graph_list = []
    t = time.time()
    starttime = time.time()
    print('Begin loading fold {} training data'.format(fold))
    print('trian_ids = ', train_ids)
    data_dir = './data/'+str(EMB_TYPE)+'_fold_training_graph/fold_'+str(fold)+'/'
    for qid in tqdm(train_ids, desc = 'loading training data', ncols=80):
        file_path = os.path.join(data_dir, str(qid)+'.pickle')
        temp_dict = pickle.load(open(file_path, 'rb'))
        graph_list.extend(temp_dict[str(qid)])
    graph_dataset = GraphDataset(graph_list)
    loader = DataLoader(
        dataset = graph_dataset, 
        batch_size = BATCH_SIZE, 
        shuffle = True, 
        num_workers = 4, 
        pin_memory = True, 
    )
    print('Training data loaded!')
    endtime = time.time()
    print('Total time  = ', round(endtime - starttime, 2), 'secs')
    return loader


def get_intent_coverage():
    ''' get the intent coverage of all document pairs '''
    qd = pickle.load(open('./data/gcn_dataset/div_query.data', 'rb'))
    output_file = './data/gcn_dataset/intent_coverage.csv'
    fout = open(output_file, 'w')
    fout.write('qid,docx,docy,label\n')
    for qid in qd:
        dq = qd[qid]
        doc_list = dq.doc_list
        subtopic_df = np.array(dq.subtopic_df)
        real_num = min(len(doc_list), MAXDOC)
        for i in range(real_num):
            for j in range(i+1, real_num):
                doc_x_id = doc_list[i]
                doc_y_id = doc_list[j]
                coverage = 0
                for k in range(subtopic_df.shape[1]):
                    if subtopic_df[i][k] == 1 and subtopic_df[j][k] == 1:
                        coverage = 1
                        break
                fout.write(str(qid)+','+str(doc_x_id)+','+str(doc_y_id)+','+str(coverage)+'\n')
    fout.close()


def document_process():
    ''' tokenize the document for relation classifier training '''
    input_file_path = './data/gcn_dataset/full_content.pkl.gz'
    output_file_path = './data/gcn_dataset/token_content.pkl.gz'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    with gzip.open(input_file_path, 'rb') as f:
        doc_dict = pickle.load(f)
    token_dict = {}

    qd = pickle.load(open('./data/gcn_dataset/div_query.data', 'rb'))
    all_docs_id_list = []
    for qid in qd:
        dq = qd[qid]
        doc_list = dq.doc_list
        real_num = min(len(doc_list), MAXDOC)
        all_docs_id_list.extend(doc_list[:real_num])
    all_docs_id_list = list(set(all_docs_id_list))

    for key in tqdm(doc_dict, desc = "Docs process"):
        if key in all_docs_id_list:
            content = doc_dict[key]
            if type(content) == list:
                content = '. '.join(content)
            content = content.lower()
            sent = tokenizer.tokenize(content)
            token_dict[key] = sent[:511]
    with gzip.open(output_file_path, 'wb') as f:
        pickle.dump(token_dict, f)


def gen_data_file(data_list, file_full, file_equal):
    ''' Generate full dataset '''
    full_output = open(file_full, 'w')
    full_output.write('qid,docx,docy,label\n')
    for t in tqdm(data_list, desc = "Gen full data"):
        full_output.write(str(t[0])+','+str(t[1])+','+str(t[2])+','+str(t[3])+'\n')
    full_output.close()
    ''' Generate proportionate dataset '''
    equal_output = open(file_equal, 'w')
    equal_output.write('qid,docx,docy,label\n')
    pos_data = []
    neg_data = []
    for t in tqdm(data_list, desc = "Split data"):
        if t[3] == '1':
            pos_data.append(t)
        elif t[3] == '0':
            neg_data.append(t)
        else:
            print('error', t)
    Lp = len(pos_data)
    Ln = len(neg_data)
    real_num = min(Lp, Ln)
    for i in tqdm(range(real_num), desc = "Gen pos"):
        t = pos_data[i]
        equal_output.write(str(t[0])+','+str(t[1])+','+str(t[2])+','+str(t[3])+'\n')
    ''' Randomly sample the negative data '''
    output_neg_index = list(set([random.randint(0, Ln - 1) for _ in range(int(1.2 * real_num))]))[:real_num]
    output_neg_index.sort()
    for i in tqdm(output_neg_index, desc = 'Gen neg'):
        u = neg_data[i]
        equal_output.write(str(u[0])+','+str(u[1])+','+str(u[2])+','+str(u[3])+'\n')
    equal_output.close()


def divide_five_fold_train_test():
    ''' Divide the coverage data into 5 folds train_dataset and test_dataset '''
    all_qids = np.load('./data/gcn_dataset/all_qids.npy')
    f = open('./data/gcn_dataset/intent_coverage.csv', 'r')
    all_data = []
    for line in tqdm(f, desc = "reading"):
        line = line.strip('\n')
        qid, docx_id, docy_id, label = line.split(',')
        all_data.append((qid, docx_id, docy_id, label))
    data_dir = './data/clf_cover_data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    fold = 0
    for train_ids, test_ids in KFold(5).split(all_qids):
        fold += 1
        res_dir = data_dir + 'fold' + str(fold) + '/'
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        train_ids.sort()
        test_ids.sort()
        train_qids = [str(all_qids[i]) for i in train_ids]
        test_qids = [str(all_qids[i]) for i in test_ids]
        train_data_list = []
        test_data_list = []
        train_full_file = res_dir + 'train_full.csv'
        train_equal_file = res_dir + 'train_equal.csv'
        test_full_file = res_dir + 'test_full.csv'
        test_equal_file = res_dir + 'test_equal.csv'
        for t in tqdm(all_data, desc = "Split Train_Test"):
            if t[0] in train_qids:
                train_data_list.append(t)
            elif t[0] in test_qids:
                test_data_list.append(t)
        gen_data_file(train_data_list, train_full_file, train_equal_file)
        gen_data_file(test_data_list, test_full_file, test_equal_file)


def make_data_set(data_type, fold):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    pad_token_segment_id = 0
    with gzip.open("./data/gcn_dataset/token_content.pkl.gz", 'rb') as f:
        docs= pickle.load(f)
    max_length = 512
    input_ids1, attention_mask1, token_type_ids1, input_ids2, attention_mask2, token_type_ids2, y_label = [], [], [], [], [], [], []
    with open("./data/clf_cover_data/fold" + str(fold) + '/' + data_type + "_equal.csv", "r") as fr:
        for idx, line in enumerate(fr):
            if idx == 0:
                continue
            else:
                if idx % 10000 == 0:
                    print(idx)
                line = line.strip().split(",")
                doc1 = docs[line[1]]
                doc2 = docs[line[2]]
                label = int(line[3])

                doc1_ids = tokenizer.convert_tokens_to_ids(doc1)
                doc1_encoding = tokenizer.encode_plus(doc1, max_length=max_length, add_special_tokens=True, truncation=True)
                doc1_input_ids, doc1_token_type_ids = doc1_encoding["input_ids"], doc1_encoding["token_type_ids"]
                doc1_attention_mask = [1] * len(doc1_input_ids)
                padding_length = max_length - len(doc1_input_ids)
                doc1_input_ids = doc1_input_ids + ([pad_token] * padding_length)
                doc1_attention_mask = doc1_attention_mask + ([0] * padding_length)
                doc1_token_type_ids = doc1_token_type_ids + ([pad_token_segment_id] * padding_length)
                assert len(doc1_input_ids) == max_length, "Error with input length {} vs {}".format(len(doc1_input_ids), max_length)
                assert len(doc1_attention_mask) == max_length, "Error with input length {} vs {}".format(len(doc1_attention_mask), max_length)
                assert len(doc1_token_type_ids) == max_length, "Error with input length {} vs {}".format(len(doc1_token_type_ids), max_length)
                input_ids1.append(doc1_input_ids)
                attention_mask1.append(doc1_attention_mask) 
                token_type_ids1.append(doc1_token_type_ids)

                doc2_ids = tokenizer.convert_tokens_to_ids(doc2)
                doc2_encoding = tokenizer.encode_plus(doc2, max_length=max_length, add_special_tokens=True, truncation=True)
                doc2_input_ids, doc2_token_type_ids = doc2_encoding["input_ids"], doc2_encoding["token_type_ids"]
                doc2_attention_mask = [1] * len(doc2_input_ids)
                padding_length = max_length - len(doc2_input_ids)
                doc2_input_ids = doc2_input_ids + ([pad_token] * padding_length)
                doc2_attention_mask = doc2_attention_mask + ([0] * padding_length)
                doc2_token_type_ids = doc2_token_type_ids + ([pad_token_segment_id] * padding_length)
                assert len(doc2_input_ids) == max_length, "Error with input length {} vs {}".format(len(doc2_input_ids), max_length)
                assert len(doc2_attention_mask) == max_length, "Error with input length {} vs {}".format(len(doc2_attention_mask), max_length)
                assert len(doc2_token_type_ids) == max_length, "Error with input length {} vs {}".format(len(doc2_token_type_ids), max_length)
                input_ids2.append(doc2_input_ids)
                attention_mask2.append(doc2_attention_mask) 
                token_type_ids2.append(doc2_token_type_ids)
                y_label.append(label)
    torch.save((np.array(input_ids1), np.array(attention_mask1), np.array(token_type_ids1), np.array(input_ids2), np.array(attention_mask2), np.array(token_type_ids2), np.array(y_label)), "./data/clf_cover_data/fold" + str(fold) + "/cached_equal_" + data_type)
