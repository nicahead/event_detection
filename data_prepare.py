# -*- coding: utf-8 -*-
# @Time      :   2020/8/24 18:51
# @Author    :   nicahead@gmail.com
# @File      :   data_prepare.py
# @Desc      :   百度语料预处理

import json
import collections
import random

import string

import config
import jieba
import pandas as pd
import numpy as np
import pickle
import re

from utils import is_number

random.seed(2020)
np.random.seed(2020)


def event_count():
    """
    统计原语料中各事件类型的数量
    :return:
    """
    # 读取json文件
    with open('DuEE/data.json', 'r', encoding='UTF-8') as f:
        data = f.readlines()
    data = [json.loads(item) for item in data]  # json对象列表

    all_event_classes = []
    all_events = []
    count = 0
    count_event_num = [0, 0, 0, 0]  # 句子中数目为0、1、2、大于等于3
    # 统计各事件类型的数量
    for item in data:
        # 事件大类
        all_event_classes += [event['event_type'].split('-')[0] for event in item['event_list']]
        # 小类
        all_events += [event['event_type'] for event in item['event_list']]
        count += len(item['event_list'])
        if len(item['event_list']) == 0:
            count_event_num[0] += 1
        elif len(item['event_list']) == 1:
            count_event_num[1] += 1
        elif len(item['event_list']) == 2:
            count_event_num[2] += 1
        elif len(item['event_list']) > 2:
            count_event_num[3] += 1
    res1 = collections.Counter(all_event_classes)
    print('事件大类：', len(res1.items()))
    print(res1)
    res2 = collections.Counter(all_events)
    print('事件小类：', len(res2.items()))
    print(res2)
    print('句子数量：', len(data))
    print('事件数量：', count)
    print('句子中没有事件:{}，句子中有1个事件:{}，句子中有2个事件:{}，句子中有大于3个事件:{}'.format(count_event_num[0], count_event_num[1],
                                                                    count_event_num[2], count_event_num[3]))


def get_event_dict():
    """
    解析ontology.json，得到中事件类标签映射
    :return: [id2label,label2id,event_express]
    """
    id2label = []  # 事件类列表，即id到label的映射
    # 读取json文件
    with open('ontology.json', 'r', encoding='UTF-8') as f:
        data = f.readlines()
    for event in data:
        obj = json.loads(event)
        id2label.append(obj['parent'] + '-' + obj['name'])
    id2label.append('NA')  # 没有事件类型
    label2id = {id2label[i]: i for i in range(len(id2label))}  # 事件类名到id的映射
    return id2label, label2id


def replace_digit(event_content):
    """
    将句子中的数字替换为<NUM>
    :param event_content:
    :return:
    """
    words = event_content.split(' ')
    res = ''
    for word in words:
        if is_number(word):
            word = '<NUM>'
        res += word + ' '
    return res


def stopwordslist():
    """
    加载停用词表
    """
    stopwords = [line.strip() for line in open('cn_stopwords.txt', encoding='UTF-8').readlines()]
    return stopwords


def del_stopwords(words):
    """
    去停用词
    """
    stopwords = stopwordslist()
    res = []
    for word in words:
        if word not in stopwords and word != '\n':
            res.append(word)
    return res


def sentence_handle(sentence):
    """
    对句子进行处理：分词
    :param sentence:
    :return:
    """
    # 去标点符号
    rec = re.sub('[%s]' % re.escape(string.punctuation), '', sentence)
    # 精确分词
    words = jieba.lcut(rec)
    newline = del_stopwords(words)
    str_out = ' '.join(newline).replace('，', '').replace('。', '').replace('?', '').replace('!', '') \
        .replace('“', '').replace('”', '').replace('：', '').replace('‘', '').replace('’', '').replace('-', '') \
        .replace('（', '').replace('）', '').replace('《', '').replace('》', '').replace('；', '').replace('.', '') \
        .replace('、', '').replace('...', '').replace(',', '').replace('？', '').replace('！', '')
    str_out = replace_digit(str_out)
    str_out = ' '.join(str_out.split())  # 删除多余空格，确保两个词之间只有一个空格
    return str_out


def get_corpus_dataset(name="train"):
    """
    根据语料，构造text/label的数据集
    :param events:
    :return:
    """
    id2label, label2id = get_event_dict()
    res = []
    # 读取json文件
    with open('data/' + name + '.json', encoding='UTF-8') as f:
        data = f.readlines()
    for item in data:
        item = json.loads(json.dumps(eval(item)))
        info = {}
        info['label'] = ''
        for event in item['event_list']:
            # 只将事件本体中定义过事件类型的作为数据集，且如果一个句子中包含重复的事件，只保留一个
            if event['event_type'] in id2label and event['event_type'] not in info['label']:
                # if event['event_type'] not in info['label']:
                info['text'] = item['text'].replace('\n', '')
                info['label'] += event['event_type'] + ' '
        if 'text' in info.keys():
            info['label'] = info['label'].rstrip()  # 去掉结尾空格
            res.append(info)
    df = pd.DataFrame(res)
    save_path = 'data/corpus_' + name + '.csv'
    df.to_csv(save_path, index=None)


def get_ontology_dataset():
    """
    根据事件本体，构造text/label的数据集
    :param events:
    :return:
    """
    res = []
    # 读取json文件
    with open('ontology.json', 'r', encoding='UTF-8') as f:
        data = f.readlines()
    for event in data:
        info = {}
        obj = json.loads(event)
        # 事件类的表示，由事件要素组合而成
        info['text'] = ' '.join(
            obj['object'] + obj['action'] + obj['time'] + obj['environment'] + obj['language'])
        info['label'] = obj['parent'] + '-' + obj['name']
        res.append(info)
    df = pd.DataFrame(res)
    save_path = 'data/ontology.csv'
    df.to_csv(save_path, index=None)


def get_dataset(name='train'):
    """
    获取相似度三元组数据集，保存为train.csv
    :return:
    """
    df1 = pd.read_csv('data/corpus_' + name + '.csv')
    df2 = pd.read_csv('data/ontology.csv')
    res = []
    for row in zip(df1['text'], df1['label']):
        for event_row in zip(df2['text'], df2['label']):
            triple = {}
            triple['text'] = sentence_handle(row[0])
            triple['event'] = event_row[0]
            # 正例
            if event_row[1] in row[1]:
                triple['label'] = 1
            # 负例
            else:
                triple['label'] = 0
            res.append(triple)
    df = pd.DataFrame(res)
    save_path = 'data/' + name + '.csv'
    df.to_csv(save_path, index=None)


def read_vectors(path, topn):  # read top n word vectors, i.e. top is 10000
    """
    读取txt文件，得到词和向量
    :param path:
    :param topn:
    :return:
    """
    lines_num, dim = 0, 0
    vectors = {}
    iw = []
    wi = {}
    with open(path, encoding='utf-8', errors='ignore') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                dim = int(line.rstrip().split()[1])
                continue
            lines_num += 1
            tokens = line.rstrip().split(' ')
            vectors[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
            iw.append(tokens[0])
            if topn != 0 and lines_num >= topn:
                break
    for i, w in enumerate(iw):
        wi[w] = i
    return vectors, iw, wi, dim


def get_pretraining_dict():
    """
    读取大的预训练词典，根据本语料中的词和字生成缩小版词典
    :return:
    """
    vocab = []
    vectors, iw, wi, dim = read_vectors('data/merge_sgns_bigram_char300.txt', 0)  # 所有字的词典
    wi['<PAD>'] = len(iw)
    wi['<NUM>'] = len(iw) + 1
    wi['<UNK>'] = len(iw) + 2
    vectors['<PAD>'] = np.zeros((300,))
    vectors['<NUM>'] = np.random.uniform(-0.1, 0.1, 300)
    vectors['<UNK>'] = np.zeros((300,))

    words = []
    # 根据语料得到缩小版的预训练参数
    df1 = pd.read_csv('data/corpus_train.csv')
    for index, value in df1['text'].items():
        word = sentence_handle(value).split(' ')
        words += word
        words += [char for char in word]

    df2 = pd.read_csv('data/corpus_dev.csv')
    for index, value in df2['text'].items():
        word = sentence_handle(value).split(' ')
        words += word
        words += [char for char in word]

    df3 = pd.read_csv('data/corpus_test.csv')
    for index, value in df3['text'].items():
        word = sentence_handle(value).split(' ')
        words += word
        words += [char for char in word]

    df4 = pd.read_csv('data/ontology.csv')
    for index, value in df4['text'].items():
        word = value.split(' ')
        words += word
        words += [char for char in word]

    # words = list(set(words))  # 语料中所有的词
    words.append('<PAD>')
    words.append('<NUM>')
    words.append('<UNK>')
    # 字
    # chars = [list(word) if word not in ['<PAD>', '<NUM>', '<UNK>'] else '的' for word in words]
    # from itertools import chain
    # chars = list(chain(*chars))
    # chars = list(set(chars))  # 语料中所有的字
    # chars.append('<PAD>')
    # chars.append('<UNK>')
    # chars.append('<NUM>')
    # tokens = list(set(chars + words))
    tokens = list(set(words))
    # 取出token对应的信息
    # word2idx和vector matrix
    word2idx = {}
    vec_matrix = []
    index = 0
    for token in tokens:
        word2idx[token] = index
        try:
            embedding = vectors[token]
        except Exception:
            embedding = np.array([vectors.get(word, vectors['<UNK>']) for word in token])
            embedding = embedding.mean(axis=0)
        vec_matrix.append(embedding)
        index += 1
    vocab.append(word2idx)
    vocab.append(vec_matrix)
    # 写入字典
    with open('data/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)


def DuEE_process():
    """
    将data.json（train.json和dev.json合并得到）根据比例划分训练集、验证集、测试集，划分时按照每一类事件划分即分层抽样
    :return:
    """
    with open('DuEE/data.json', 'r', encoding='UTF-8') as f:
        data = f.readlines()
    data = [json.loads(item) for item in data]  # json对象列表
    id2label, label2id = get_event_dict()
    new_data = [[] for i in range(len(id2label))]
    random.shuffle(data)
    # 原数据分类
    for item in data:
        temp = []  # 记录
        for event in item['event_list']:
            # 是事件本体里定义的事件类型，且没有重复
            if event['event_type'] in id2label and event['event_type'] not in temp:
                new_data[label2id[event['event_type']]].append(item)
                temp.append(event['event_type'])
        # 如果是多个事件，取第一个，避免训练集和测试集验证集重合
        # new_data[label2id[item['event_list'][0]['event_type']]].append(item)
    # 切分数据，按照8：1：1的划分比例
    train_data = []
    dev_data = []
    test_data = []
    # 每种事件类型，都安装8：1：1划分，可使各类别更加均匀
    for event_list in new_data:
        train_data_sub = event_list[:int(len(event_list) * 0.8)]
        # 增强train_data
        # if len(event_list) < 50:
        #     # 三倍
        #     # train_data_sub = train_data_sub * 3
        #     train_data_sub = enhance(train_data_sub, 3)
        # elif len(event_list) >= 50 and len(event_list) < 100:
        #     # 两倍
        #     train_data_sub = enhance(train_data_sub, 2)
        #     # train_data_sub = train_data_sub * 2
        train_data.extend(train_data_sub)
        dev_data.extend(event_list[int(len(event_list) * 0.8):int(len(event_list) * 0.9)])
        test_data.extend(event_list[int(len(event_list) * 0.9):])
    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)
    train_id = [item['id'] for item in train_data]
    with open('data/train.json', 'w', encoding='UTF-8') as f:
        for line in train_data:
            f.write(str(line).replace('\n', '') + '\n')
        f.close()
    with open('data/dev.json', 'w', encoding='UTF-8') as f:
        for line in dev_data:
            if line['id'] not in train_id:
                f.write(str(line).replace('\n', '') + '\n')
        f.close()
    with open('data/test.json', 'w', encoding='UTF-8') as f:
        for line in test_data:
            if line['id'] not in train_id:
                f.write(str(line).replace('\n', '') + '\n')
        f.close()


if __name__ == '__main__':
    # event_count()

    DuEE_process()  # 将原始语料的train和dev文件合并，然后分层抽样得到新的train.json、dev.json、test.json
    get_ontology_dataset()  # 提取事件本体中的句子和label，得到ontology.csv

    # 提取语料（train.json）中的句子和label，得到corpus_train.csv、、、
    get_corpus_dataset("train")
    get_corpus_dataset("dev")  # 提取语料中的句子和label
    get_corpus_dataset("test")  # 提取语料中的句子和label

    get_pretraining_dict()  # 根据以上三个文件中的句子得到缩小版的词典
    get_dataset("train")  # 获得三元组数据集
    get_dataset("dev")  # 获得三元组数据集
    get_dataset("test")  # 获得三元组数据集
