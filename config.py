# -*- coding: utf-8 -*-
# @Time      :   2020/8/14 15:44
# @Author    :   nicahead@gmail.com
# @File      :   config.py
# @Desc      :   配置文件

import torch
import os

PROJECT_ROOT = os.path.dirname(__file__)
VOCAB_PATH = os.path.join(PROJECT_ROOT, 'data/vocab.pkl')  # 词表（预训练参数）
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, 'data/train.csv')  # 训练数据
DEV_DATA_PATH = os.path.join(PROJECT_ROOT, 'data/dev.csv')  # 验证数据
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, 'data/test.csv')  # 测试数据
TRAIN_CORPUS_PATH = os.path.join(PROJECT_ROOT, 'data/corpus_train.csv')
DEV_CORPUS_PATH = os.path.join(PROJECT_ROOT, 'data/corpus_dev.csv')
TEST_CORPUS_PATH = os.path.join(PROJECT_ROOT, 'data/corpus_test.csv')

# model config
EMBED_DIM = 300  # 词嵌入向量维度
HIDDEN_DIM = 300
N_LAYERS = 2
BI = True
DROPOUT_RATE = 0.5
FIX_EMDED = False
ATTENTION = True

# train config
EPOCH = 30
BATCH_SIZE = 200
LR = 0.0001  # learning rate
LR_DECAY_RATE = 0.8
WEIGHT_DECAY = 1e-5
N_EVENT_CLASS = 65  # 事件本体中的事件类个数

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
