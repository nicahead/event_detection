# -*- coding: utf-8 -*-
# @Time      :   2020/8/17 10:40
# @Author    :   nicahead@gmail.com
# @File      :   train.py
# @Desc      :
from sklearn.metrics import classification_report
import pickle
import torch
import numpy as np
from time import *
from tqdm import tqdm
import copy
import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd
import os

from PGD import FGM
from config import config
from data_helper import BatchManager
from data_prepare import get_event_dict
from eval import eval_graph, evaluate, evaluate_results
from loss import focal_loss
from model import SiameseNetwork

import sys


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


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
    y_true = []
    y_pred = []
    model.train()
    fgm = FGM(model)
    with tqdm(total=train_manager.len_data, desc='train batch') as pbar:
        for (text, event, target) in train_manager.iter_batch(shuffle=True):
            optimizer.zero_grad()
            text = torch.LongTensor(text).to(device)
            event = torch.LongTensor(event).to(device)
            target = torch.LongTensor(target).to(device)
            y_true.extend(target.tolist())
            fgm.attack()  # 在embedding上添加对抗扰动
            output = model(text, event)
            y_pred.extend(torch.max(output, 1)[1].tolist())
            loss = loss_func(output, target)

            loss.backward()
            fgm.restore()  # 恢复embedding参数
            optimizer.step()
            total_loss.append(loss.item())
            pbar.update(1)
        ave_loss = sum(total_loss) / len(total_loss)
    # 打印二分类效果
    target_names = ['0', '1']
    print('train binary classification:')
    print(classification_report(y_true, y_pred, target_names=target_names))
    return ave_loss


def dev(model, dev_manager, loss_func, device):
    total_loss = []
    y_true = []
    y_pred = []
    res = []  # 多标签分类结果
    model.eval()
    with tqdm(total=dev_manager.len_data, desc='dev batch') as pbar:
        for (text, event, target) in dev_manager.iter_batch(shuffle=True):
            text = torch.LongTensor(text).to(device)
            event = torch.LongTensor(event).to(device)
            target = torch.LongTensor(target).to(device)
            y_true.extend(target.tolist())  # 二分类label

            output = model(text, event)
            pred = torch.max(output, 1)[1].tolist()
            y_pred.extend(pred)  # 二分类预测值

            loss = loss_func(output, target)
            total_loss.append(loss.item())
            # 评估多标签分类结果
            pred = [i for i in range(len(pred)) if pred[i] == 1]  # 句子的事件类型，预测值
            id2label, label2id = get_event_dict()
            if len(pred) == 0:
                pred = [label2id['NA']]
            label = [i for i in range(len(target)) if target[i] == 1]  # 句子的事件类型，真实值
            if len(label) == 0:
                label = [label2id['NA']]
            res.append((pred, label))
            pbar.update(1)
    ave_loss = sum(total_loss) / len(total_loss)
    # 打印二分类效果
    target_names = ['0', '1']
    print('dev binary classification:')
    print(classification_report(y_true, y_pred, target_names=target_names))
    acc, precision, recall, f1 = evaluate_results(res, label2id['NA'])
    return ave_loss, acc, precision, recall, f1


if __name__ == '__main__':
    if not os.path.exists('models'):
        os.mkdir('models')
    if os.path.exists('models/temp'):
        import shutil

        shutil.rmtree('models/temp')
        os.mkdir('models/temp')
    else:
        os.mkdir('models/temp')
    sys.stdout = Logger('models/temp/output', sys.stdout)
    sys.stderr = Logger('models/temp/output', sys.stderr)

    # -------数据准备---------
    # 初始化数据，从本地读取
    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    train_data = train_df.values.tolist()
    train_manager = BatchManager(batch_size=config.BATCH_SIZE, data=train_data, mode='train')

    dev_df = pd.read_csv(config.DEV_DATA_PATH)
    dev_data = dev_df.values.tolist()
    dev_manager = BatchManager(batch_size=config.N_EVENT_CLASS, data=dev_data, mode='dev')

    embed_matrix = prepare_data()
    model = SiameseNetwork(embed_matrix)

    if config.DEVICE.type == 'cuda':
        model = model.cuda()
    # optimizer = optim.Adam(model.parameters(), lr=config.LR)
    # optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    # 动态调整学习率
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=config.LR_DECAY_RATE, patience=3,
    #                                            verbose=True, threshold=0.01, threshold_mode='abs', cooldown=0,
    #                                            min_lr=0.000001, eps=1e-08)
    # optimizer = optim.SGD(model.parameters(), lr=config.LR, momentum=0.9, weight_decay=config.WEIGHT_DECAY)
    # optimizer = optim.RMSprop(model.parameters(), lr=config.LR, alpha=0.9, weight_decay=config.WEIGHT_DECAY)

    # 设置权重，正负样本不均衡
    # weight = torch.from_numpy(np.array([0.1, 0.5])).float().to(config.DEVICE)
    # loss_func = torch.nn.CrossEntropyLoss(weight=weight)
    # loss_func = torch.nn.CrossEntropyLoss()
    loss_func = focal_loss(alpha=[1, 35], gamma=2, num_classes=2)  # 设置正样本权重

    train_losses = []
    dev_losses = []
    acc_ls = []
    rec_ls = []
    f1_ls = []

    begin_time = time()

    best_acc = 0
    best_f1 = 0
    best_acc_model_name = None
    best_acc_model = None
    best_f1_model = None
    # 训练模型，直到 epoch == config.EPOCH 或者触发 early_stopping 结束训练
    count = 0  # 记录f1值没有增加的epoch个数
    for epoch in range(config.EPOCH):
        print('=================================== epoch:{} ==================================='.format(epoch))
        train_loss = train(model, train_manager, loss_func, optimizer, config.DEVICE)
        dev_loss, acc, pre, rec, f1 = dev(model, dev_manager, loss_func, config.DEVICE)
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
            count = 0
        else:
            count += 1
        # scheduler.step(f1)
        # 若满足 early stopping 要求
        # if count == config.N_EARLY_STOP:
        #   print("Early stopping")
        #   break
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

    print(config.__dict__.items())
    print(model)
