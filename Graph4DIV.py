import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F
from modules import GraphConvolutionNetwork


class Graph4DIV(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, output_dim, activation = "relu", nodenum = 50, batch_norm = True, normalization = False, dropout = 0.1):
        super(Graph4DIV, self).__init__()
        self.gcn_layers_dim = [node_feature_dim] + hidden_dim
        self.gcn_layers_num = len(self.gcn_layers_dim)
        self.input_dim = node_feature_dim
        self.output_dim = output_dim
        self.batch_norm = batch_norm
        self.node_num = nodenum
        self.normalization = normalization
        self.GNN = GraphConvolutionNetwork(node_feature_dim, hidden_dim, output_dim, activation, batch_norm, normalization)
        self.fc1 = nn.Linear(3 * self.output_dim + 1, 50)
        self.fc2 = nn.Linear(50, 1)
        self.nfc1 = nn.Linear(18, 18)
        self.nfc2 = nn.Linear(18, 8)
        self.nfc3 = nn.Linear(8, 1)
        init.xavier_normal_(self.nfc1.weight)
        init.xavier_normal_(self.nfc2.weight)
        init.xavier_normal_(self.nfc3.weight)
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, A, X, Ri, Di, pos_mask = None, neg_mask = None, train_flag = False):
        Tg, Zi = self.GNN(A, X)
        Tg = Tg.reshape(Tg.shape[0], 1, Tg.shape[1])
        Tg = Tg.repeat(1, Zi.shape[1] - 1, 1)
        Zq = Zi[:, 0, :].unsqueeze(dim = 1).repeat(1, Zi.shape[1] - 1, 1)
        Zi = Zi[:, 1:, :]
        Zi = torch.cat([Zi, Tg, Zq, Di], dim = 2)

        sd = self.dropout(sd)
        sd = F.relu(self.fc1(sd))
        sd = self.fc2(sd)
        sd = sd.view(sd.shape[0] * self.node_num, 1)

        Ri = Ri.reshape(Ri.shape[0] * self.node_num, 18)
        sr = F.relu(self.nfc1(Ri))
        sr = F.relu(self.nfc2(sr))
        sr = self.nfc3(sr)

        sd = sd.view(sd.shape[0] * sd.shape[1])
        sr = sr.view(sr.shape[0] * sr.shape[1])
        score = 0.5 * sr + 0.5 * sd
        if train_flag:
            pos_mask = pos_mask.view(pos_mask.shape[0] * pos_mask.shape[1])
            score_1 = torch.masked_select(score, pos_mask)
            neg_mask = neg_mask.view(neg_mask.shape[0] * neg_mask.shape[1])
            score_2 = torch.masked_select(score, neg_mask)
            return score_1, score_2
        else:
            return score
