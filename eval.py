# -*- coding: utf-8 -*-
# @Time      :   2020/9/11 8:18
# @Author    :   nicahead@gmail.com
# @File      :   eval.py
# @Desc      :
import pandas as pd
import torch
from matplotlib import pyplot as plt
import pickle

from config import config
from data_helper import BatchManager
from data_prepare import sentence_handle, get_event_dict


def evaluate_results(result, neg_id):
    """
    根据预测结果和真实值，计算准确率、精确率、召回率、F1值
    :param result: [预测的事件类型，实际的事件类型] eg:[([1,2],[1,3]),([2],[2])....]
    :param neg_id: 不存在的事件类型 NA
    :return:
    """
    total_p, total_g, right, total, total_right = 0, 0, 0, 0, 0
    # 遍历预测结果
    for _p, g in result:
        total += len(_p)  # 预测的样本数 TP+TN+FP+FN
        if g[0] != neg_id:
            total_g += len(g)  # gold事件总数 TP+FN
        for p in _p:
            if p != neg_id:
                total_p += 1  # 预测的正样本事件总数 TP+FP
            if p in g:
                total_right += 1  # 正确预测的事件数，包括负样本NA TP+TN
            if p != neg_id and p in g:
                right += 1  # 正确预测的事件数，仅正样本 TP
    if total_p == 0:
        total_p = 1
    acc = 1.0 * total_right / total
    precision = 1.0 * right / total_p  # 查准率 TP/(TP+FP)
    recall = 1.0 * right / total_g  # 查全率 TP/(TP+FN)
    f1 = 2 * precision * recall / (precision + recall + 0.000001)
    out = '预测的事件总数（包括负样本NA）：%d\n预测的事件总数（仅正样本）: %d\n正确预测的事件数（包括负样本NA）：%d\n正确预测的事件数（仅正样本）: %d\n真实事件总数: %d' % (
        total, total_p, total_right, right, total_g)
    # out += '----------------------------\n'
    # out += '准确率：%.3f 精确率：%.3f 召回率：%.3f F1得分：%.3f\n' % (acc, pre, rec, f1)
    print(out)
    return acc, precision, recall, f1


def evaluate(name, model, type='model'):
    """
    评估模型，对于每个句子，将其与所有事件类得到组合作为一个batch传入模型，得到属于每个类的相似度
    :param name: dev/test
    :param model:
    :param type: model-传入模型，path-传入路径
    :param device:
    :return:
    """
    df = pd.read_csv('data/corpus_{}.csv'.format(name))
    if type == 'path':
        model = torch.load(model)
        if config.DEVICE.type == 'cuda':
            model = model.cuda()
    id2label, label2id = get_event_dict()
    res = []
    index = 1
    for text, label in zip(df['text'], df['label']):
        pred = predict_sentence(text, model)
        pred = pred.cpu().numpy().tolist()
        pred = [i for i in range(len(pred)) if pred[i] == 1]  # 句子的事件类型，预测值
        if len(pred) == 0:
            pred = [label2id['NA']]
        label = label.split(' ')
        label = [label2id[item] for item in label]  # 句子的事件类型，真实值
        if len(label) == 0:
            label = [label2id['NA']]
        res.append((pred, label))
        # 打印预测结果
        if name == 'test':
            gold_ans = ','.join([id2label[x] for x in label])
            pred_ans = ','.join([id2label[x] for x in pred])
            print('Sample %d: [sentence=%s] \n\t [label=%s], [pred=%s]\n' % (index, text, gold_ans, pred_ans))
        index += 1
    # out = evaluate_results_binary(res, label2id['NA'])
    # print(res)
    return evaluate_results(res, label2id['NA'])
    # print(out)


def predict_sentence(sentence, model):
    """
    对一个句子进行预测，得到其类别
    :param sentence:
    :return:和每个类别的相似程度，1为相似，0为不相似
    """
    df = pd.read_csv('data/ontology.csv')
    text = sentence_handle(sentence)
    res = []
    for event_text in df['text']:
        triple = [text, event_text, '']
        res.append(triple)
    batchManager = BatchManager(config.N_EVENT_CLASS, res, mode='test')
    (text, event, target) = batchManager.iter_batch(shuffle=True).__next__()
    text = torch.LongTensor(text).to(config.DEVICE)
    event = torch.LongTensor(event).to(config.DEVICE)
    model.eval()
    output = model(text, event)
    predictions = torch.max(output, 1)[1]  # 预测值0/1
    return predictions


def eval_graph(path):
    with open(path, 'rb') as f:
        data_dict = pickle.load(f)
    plt.rcParams['figure.figsize'] = (10.0, 5.0)
    # loss图
    plt.subplot(121)
    plt.plot(data_dict['train_losses'], label='train_loss')
    plt.plot(data_dict['dev_losses'], label='dev_loss')
    plt.legend(loc='upper right')
    plt.title('CrossEntropyLoss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    # 评测指标图
    plt.subplot(122)
    plt.plot(data_dict['acc_ls'], label='accuracy')
    plt.plot(data_dict['rec_ls'], label='recall')
    plt.plot(data_dict['f1_ls'], label='F1 Score')
    plt.legend(loc='lower right')
    plt.title('Evaluation')
    plt.xlabel("epoch")
    plt.ylabel("rate")
    plt.savefig('models/temp/loss.jpg')
    plt.show()


def compare_graph(path1, name1, path2, name2):
    with open(path1, 'rb') as f1:
        data_dict1 = pickle.load(f1)
    with open(path2, 'rb') as f2:
        data_dict2 = pickle.load(f2)
    plt.rcParams['figure.figsize'] = (15.0, 5.0)
    # 准确率
    plt.subplot(131)
    plt.plot(data_dict1['acc_ls'], label=name1)
    plt.plot(data_dict2['acc_ls'], label=name2)
    plt.legend(loc='lower right')
    plt.title('accuracy')
    plt.xlabel("epoch")
    plt.ylabel("rate")
    # 召回率
    plt.subplot(132)
    plt.plot(data_dict1['rec_ls'], label=name1)
    plt.plot(data_dict2['rec_ls'], label=name2)
    plt.legend(loc='lower right')
    plt.title('recall')
    plt.xlabel("epoch")
    plt.ylabel("rate")
    # 评测指标图
    plt.subplot(133)
    plt.plot(data_dict1['f1_ls'], label=name1)
    plt.plot(data_dict2['f1_ls'], label=name2)
    plt.legend(loc='lower right')
    plt.title('f1 score')
    plt.xlabel("epoch")
    plt.ylabel("rate")
    plt.show()


def atttention():
    import seaborn as sns
    fr = open('./pkl/attention_matrix.pkl', 'rb')
    tokens, attention = pickle.load(fr)
    plt.figure(figsize=(30, 20))
    sns.heatmap(attention, vamx=100, vmin=0)
    plt.savefig('./log/attention_matrix.png')

    # 获取数据
    # import heapq
    # check_file = './log/check_attention_keywords.txt'
    # clean(check_file)
    # fw = open(check_file, 'a', encoding='utf8')
    # for t, a in zip(tokens, attention):
    #     temp = []
    #     max_num_index_list = map(list(a).index, heapq.nlargest(5, list(a)))
    #     for index in max_num_index_list:
    #         word = t[index]
    #     print(word)
    #     temp.append(word)
    #     fw.write(str(temp) + '\n')

if __name__ == '__main__':
    # compare_graph('models/v6/result.pkl', 'no weighted', 'models/v4/result.pkl', 'weighted')

    acc_1, pre_1, rec_1, f1_1 = evaluate('test',
                                             'models/old/v5-1/epoch22-loss0.003079123445143054-acc0.8977727013135351-f0.9160834161047348',
                                             type='path')
    print('accuracy：%.3f precision：%.3f recall：%.3f F1：%.3f' % (acc_1, pre_1, rec_1, f1_1))
    pass
