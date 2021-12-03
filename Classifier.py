import os
import time
import argparse
import pickle
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import logging
import torch.nn.utils as utils
from SentenceBERT import SentenceBERT
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from evaluate import evaluate_accuracy
import data_process as DP

# Required parameters
parser = argparse.ArgumentParser()
parser.add_argument("--is_training", default = False, type = bool, help = "Training model or evaluating model?")
parser.add_argument("--batch_size", default = 4, type = int, help = "The batch size.")
parser.add_argument("--test_batch_size", default = 512, type = int, help = "The batch size.")
parser.add_argument("--learning_rate", default = 3e-5, type = float, help = "The initial learning rate for Adam.")
parser.add_argument("--epochs", default = 3, type = int, help = "Total number of training epochs to perform.")
parser.add_argument("--fold", default = 1, type = int, help = "Fold number.")
parser.add_argument("--is_finetuning", default = True, type = bool, help = "Training model or evaluating model?")
parser.add_argument("--save_path", default = "./clf_model/", type = str, help = "The path to save model.")
parser.add_argument("--bert_save_path", default = "./clf_model/finetuned_bert/", type = str, help = "The path to save model.")
parser.add_argument("--score_file_path", default = "score_file.txt", type = str, help = "The path to save model.")
parser.add_argument("--log_path", default = "./log/", type = str, help = "The path to save log.")
args = parser.parse_args()

data_path = "./data/clf_cover_data/fold" + str(args.fold) + "/"
result_path = "./data/clf_output/fold" + str(args.fold) + "/"
args.save_path += "fold" + str(args.fold) + "/" + SentenceBERT.__name__ + "_equal"
args.score_file_path = result_path + SentenceBERT.__name__ + ".test_full." + args.score_file_path
device = torch.device("cuda:0")
print(args)


def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def train_model():
    X_train_input_ids1, X_train_attention_mask1, X_train_token_type_ids1, X_train_input_ids2, X_train_attention_mask2, X_train_token_type_ids2, y_train = torch.load(data_path + "cached_equal_train")
    X_dev_input_ids1, X_dev_attention_mask1, X_dev_token_type_ids1, X_dev_input_ids2, X_dev_attention_mask2, X_dev_token_type_ids2, y_dev = torch.load(data_path + "cached_equal_test")
    model = SentenceBERT(args=args)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print('* number of parameters: %d' % n_params)
    fit(model, X_train_input_ids1, X_train_attention_mask1, X_train_token_type_ids1, X_train_input_ids2, X_train_attention_mask2, X_train_token_type_ids2, y_train, X_dev_input_ids1, X_dev_attention_mask1, X_dev_token_type_ids1, X_dev_input_ids2, X_dev_attention_mask2, X_dev_token_type_ids2, y_dev)


def train_step(model, data, loss_func):
    with torch.no_grad():
        batch_i1, batch_a1, batch_t1, batch_i2, batch_a2, batch_t2, batch_y = (item.cuda(device=device) for item in data)
    # output: [batch, num_sentences]
    output = model.forward(batch_i1, batch_a1, batch_t1, batch_i2, batch_a2, batch_t2)
    loss = loss_func(output, batch_y)
    return loss


def fit(model, X_train_input_ids1, X_train_attention_mask1, X_train_token_type_ids1, X_train_input_ids2, X_train_attention_mask2, X_train_token_type_ids2, y_train, X_dev_input_ids1, X_dev_attention_mask1, X_dev_token_type_ids1, X_dev_input_ids2, X_dev_attention_mask2, X_dev_token_type_ids2, y_dev):
    X_train_input_ids1, X_train_attention_mask1, X_train_token_type_ids1, X_train_input_ids2, X_train_attention_mask2, X_train_token_type_ids2 = torch.LongTensor(X_train_input_ids1), torch.LongTensor(X_train_attention_mask1), torch.LongTensor(X_train_token_type_ids1), torch.LongTensor(X_train_input_ids2), torch.LongTensor(X_train_attention_mask2), torch.LongTensor(X_train_token_type_ids2)
    y_train = torch.FloatTensor(y_train)
    dataset = TensorDataset(X_train_input_ids1, X_train_attention_mask1, X_train_token_type_ids1, X_train_input_ids2, X_train_attention_mask2, X_train_token_type_ids2, y_train)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    t_total = int(len(X_train_input_ids1) * args.epochs // args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * 0.2), num_training_steps=t_total)
    loss_func = nn.BCELoss()
    best_result = -1e6
    one_epoch_step = len(y_train) // args.batch_size

    for epoch in range(args.epochs):
        print("\nEpoch ", epoch + 1, "/", args.epochs)
        avg_loss = 0
        all_len = len(dataloader)
        model.train()
        for i, data in enumerate(tqdm(dataloader)):
            loss = train_step(model, data, loss_func)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            for param_group in optimizer.param_groups:
                args.learning_rate = param_group['lr']

            if i > 0 and i % (one_epoch_step // 5) == 0:
                best_result = evaluate(model, X_dev_input_ids1, X_dev_attention_mask1, X_dev_token_type_ids1, X_dev_input_ids2, X_dev_attention_mask2, X_dev_token_type_ids2, y_dev, best_result)
                model.train()

            utils.clip_grad_norm_(model.parameters(), 1.0)
            avg_loss += loss.item()
        cnt = len(y_train) // args.batch_size + 1
        tqdm.write("Average loss:{:.6f} ".format(avg_loss / cnt))
        evaluate(model, X_dev_input_ids1, X_dev_attention_mask1, X_dev_token_type_ids1, X_dev_input_ids2, X_dev_attention_mask2, X_dev_token_type_ids2, y_dev, best_result)


def evaluate(model, X_dev_input_ids1, X_dev_attention_mask1, X_dev_token_type_ids1, X_dev_input_ids2, X_dev_attention_mask2, X_dev_token_type_ids2, y_dev,  best_result, is_test=False):
    y_pred = predict(model, X_dev_input_ids1, X_dev_attention_mask1, X_dev_token_type_ids1, X_dev_input_ids2, X_dev_attention_mask2, X_dev_token_type_ids2, y_dev, is_test)
    accuracy = evaluate_accuracy(y_pred, y_dev)
    if not is_test and accuracy > best_result:
        best_result = accuracy
        tqdm.write("Best Result: Acc %.4f" % (accuracy))
        torch.save(model.state_dict(), args.save_path + ".pt")
        with open(args.score_file_path, 'w') as output:
            for r in y_pred:
                output.write(str(r) + '\n')
    if is_test:
        tqdm.write("Best Result: Acc %.4f" % (accuracy))
        with open(args.score_file_path, 'w') as output:
            for r in y_pred:
                output.write(str(r) + '\n')
    return best_result


def predict(model, X_dev_input_ids1, X_dev_attention_mask1, X_dev_token_type_ids1, X_dev_input_ids2, X_dev_attention_mask2, X_dev_token_type_ids2, y_dev, is_test=False):
    model.eval()
    y_pred = []
    X_dev_input_ids1, X_dev_attention_mask1, X_dev_token_type_ids1, X_dev_input_ids2, X_dev_attention_mask2, X_dev_token_type_ids2 = torch.LongTensor(X_dev_input_ids1), torch.LongTensor(X_dev_attention_mask1), torch.LongTensor(X_dev_token_type_ids1), torch.LongTensor(X_dev_input_ids2), torch.LongTensor(X_dev_attention_mask2), torch.LongTensor(X_dev_token_type_ids2)
    dataset = TensorDataset(X_dev_input_ids1, X_dev_attention_mask1, X_dev_token_type_ids1, X_dev_input_ids2, X_dev_attention_mask2, X_dev_token_type_ids2)
    if is_test:
        dataloader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False)
    else: 
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            batch_i1, batch_a1, batch_t1, batch_i2, batch_a2, batch_t2 = (item.cuda() for item in data)
            output = model.forward(batch_i1, batch_a1, batch_t1, batch_i2, batch_a2, batch_t2)
            y_pred.append(output.data.cpu().numpy())
    y_pred = np.concatenate(y_pred, axis=0)
    return y_pred


def load_model(model, path):
    model.load_state_dict(state_dict=torch.load(path))


def test_model():
    X_dev_input_ids1, X_dev_attention_mask1, X_dev_token_type_ids1, X_dev_input_ids2, X_dev_attention_mask2, X_dev_token_type_ids2, y_dev = torch.load(data_path + "cached_full_test")
    model = SentenceBERT(args=args)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    load_model(model, args.save_path + ".pt")
    evaluate(model, X_dev_input_ids1, X_dev_attention_mask1, X_dev_token_type_ids1, X_dev_input_ids2, X_dev_attention_mask2, X_dev_token_type_ids2, y_dev, best_result=-1e9, is_test=True)


if __name__ == '__main__':
    start = time.time()
    set_seed()
    # make dataset
    # DP.make_data_set("train", args.fold)
    # DP.make_data_set("test", args.fold)
    if args.is_training:
        train_model()
    else:
        test_model()
    end = time.time()
    print("use time: ", (end - start) / 60, " min")
