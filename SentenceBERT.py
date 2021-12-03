import random
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.optim as optim
from transformers import BertModel


class SentenceBERT(nn.Module):
    def __init__(self, args, logger=None):
        self.args = args
        super(SentenceBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier1 = nn.Linear(768 * 3, 768)
        self.classifier2 = nn.Linear(768, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, input_ids1, attention_mask1, token_type_ids1, input_ids2, attention_mask2, token_type_ids2):
        doc1_bert_inputs = {'input_ids': input_ids1, 'attention_mask': attention_mask1, 'token_type_ids': token_type_ids1}
        doc1_rep = self.dropout(self.bert(**doc1_bert_inputs)[1])
        doc2_bert_inputs = {'input_ids': input_ids2, 'attention_mask': attention_mask2, 'token_type_ids': token_type_ids2}
        doc2_rep = self.dropout(self.bert(**doc2_bert_inputs)[1])
        logits = self.classifier1(torch.cat([doc1_rep, doc2_rep, torch.abs(doc1_rep - doc2_rep)], dim = -1))
        logits = self.classifier2(self.relu(logits))
        output = torch.sigmoid(logits)
        return output.squeeze(-1)