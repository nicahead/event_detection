# -*- coding: utf-8 -*-
# @Time      :   2020/8/17 10:40
# @Author    :   nicahead@gmail.com
# @File      :   train.py
# @Desc      :
from sklearn.metrics import classification_report

import config
import pickle
import torch
import numpy as np
from time import *
from tqdm import tqdm
import copy

from eval import evaluate, eval_graph
from model import SiameseNetwork
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import os

from data_helper import BatchManager

random.seed(2020)
np.random.seed(2020)
torch.manual_seed(2020)
torch.cuda.manual_seed_all(2020)


def prepare_data():
    # 加载预训练词典
    with open(config.VOCAB_PATH, 'rb') as f:
        datalist = pickle.load(f)
    word2idx = datalist[0]
    config.vocab_size = len(word2idx)
    vector_list = datalist[1]
    embed_matrix = torch.from_numpy(np.array(vector_list))  #
    return embed_matrix


# 一个epoch的训练
def train(model, train_manager, loss_func, optimizer, device):
    total_loss = []
    model.train()
    with tqdm(total=train_manager.len_data, desc='train batch') as pbar:
        for (text, event, target) in train_manager.iter_batch(shuffle=True):
            text = torch.LongTensor(text).to(device)
            event = torch.LongTensor(event).to(device)
            target = torch.LongTensor(target).to(device)

            output = model(text, event)
            loss = loss_func(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
            pbar.update(1)
        ave_loss = sum(total_loss) / len(total_loss)
    return ave_loss


def dev(model, dev_manager, loss_func, device):
    total_loss = []
    y_true = []
    y_pred = []
    model.eval()
    with tqdm(total=dev_manager.len_data, desc='dev batch') as pbar:
        for (text, event, target) in dev_manager.iter_batch(shuffle=True):
            text = torch.LongTensor(text).to(device)
            event = torch.LongTensor(event).to(device)
            target = torch.LongTensor(target).to(device)
            y_true.extend(target.tolist())
            output = model(text, event)
            y_pred.extend(torch.max(output, 1)[1].tolist())
            loss = loss_func(output, target)
            total_loss.append(loss.item())
            pbar.update(1)
    ave_loss = sum(total_loss) / len(total_loss)
    # 打印二分类效果
    target_names = ['0', '1']
    print(classification_report(y_true, y_pred, target_names=target_names))
    return ave_loss


if __name__ == '__main__':
    # -------数据准备---------
    # 初始化数据，从本地读取
    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    train_data = train_df.values.tolist()
    train_manager = BatchManager(batch_size=config.BATCH_SIZE, data=train_data)

    dev_df = pd.read_csv(config.DEV_DATA_PATH)
    dev_data = dev_df.values.tolist()
    dev_manager = BatchManager(batch_size=config.BATCH_SIZE, data=dev_data)

    embed_matrix = prepare_data()
    model = SiameseNetwork(embed_matrix)

    if config.DEVICE.type == 'cuda':
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    # 设置权重，解决正负样本不均衡的问题
    # weight = torch.from_numpy(np.array([0.1, 1.0])).float().to(config.DEVICE)
    # loss_func = torch.nn.CrossEntropyLoss(weight=weight)
    loss_func = torch.nn.CrossEntropyLoss()

    train_losses = []
    dev_losses = []
    acc_ls = []
    rec_ls = []
    f1_ls = []

    begin_time = time()
    if not os.path.exists('models'):
        os.mkdir('models')
    os.mkdir('models/temp')
    best_acc = 0
    best_f1 = 0
    best_acc_model_name = None
    best_acc_model = None
    best_f1_model = None
    best_f1_model_name = None
    for epoch in range(config.EPOCH):
        print('=================================== epoch:{} ==================================='.format(epoch))
        train_loss = train(model, train_manager, loss_func, optimizer, config.DEVICE)
        dev_loss = dev(model, dev_manager, loss_func, config.DEVICE)
        acc, pre, rec, f1 = evaluate('dev', model, type='model')
        print('正确率：%.3f 准确率：%.3f 召回率：%.3f F1得分：%.3f' % (acc, pre, rec, f1))
        print('train loss：{}  dev loss：{}\n'.format(train_loss, dev_loss))
        train_losses.append(train_loss)  # 每个epoch的平均误差
        dev_losses.append(dev_loss)
        acc_ls.append(acc)
        rec_ls.append(rec)
        f1_ls.append(f1)
        # 保存模型参数
        if acc > best_acc:
            best_acc = acc
            best_acc_model_name = 'models/temp/epoch{}-loss{}-acc{}-f{}'.format(epoch, dev_loss, acc, f1)
            best_acc_model = copy.deepcopy(model)
        if f1 > best_f1:
            best_f1 = f1
            best_f1_model_name = 'models/temp/epoch{}-loss{}-acc{}-f{}'.format(epoch, dev_loss, acc, f1)
            best_f1_model = copy.deepcopy(model)
    torch.save(best_f1_model, best_f1_model_name)
    torch.save(best_acc_model, best_acc_model_name)
    end_time = time()
    run_time = end_time - begin_time
    print('用时：', run_time)
    # 保存实验结果
    result = {'train_losses': train_losses, 'dev_losses': dev_losses, 'acc_ls': acc_ls, 'rec_ls': rec_ls,
              'f1_ls': f1_ls}
    with open('models/temp/result.pkl', 'wb') as f:
        pickle.dump(result, f)
    eval_graph('models/temp/result.pkl')

    # 使用保存的两个模型在测试集上计算指标
    acc_1, pre1, rec_1, f1_1 = evaluate('test', best_acc_model, type='model')
    print('\nbest_acc_model:')
    print('正确率：%.3f 准确率：%.3f 召回率：%.3f F1得分：%.3f' % (acc_1, pre1, rec_1, f1_1))
    acc_2, pre2, rec_2, f1_2 = evaluate('test', best_f1_model, type='model')
    print('\nbest_f1_model:')
    print('正确率：%.3f 准确率：%.3f 召回率：%.3f F1得分：%.3f' % (acc_2, pre2, rec_2, f1_2))
