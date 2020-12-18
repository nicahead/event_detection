# -*- coding: utf-8 -*-
# @Time      :   2020/8/31 16:06
# @Author    :   nicahead@gmail.com
# @File      :   data_helper.py
# @Desc      :   数据batch准备
import math
import pickle
import random

from config import config


class BatchManager(object):
    def __init__(self, batch_size, data=None):
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    # 构造batch数据
    def sort_and_pad(self, data, batch_size):
        with open(config.VOCAB_PATH, 'rb') as f:
            datalist = pickle.load(f)
        word2idx = datalist[0]
        num_batch = int(math.ceil(len(data) / batch_size))  # 总共有多少个batch
        sorted_data = sorted(data, key=lambda x: max(len(x[0].split(' ')), len(x[1].split(' '))))  # 按照句子中词的个数排序
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i * int(batch_size): (i + 1) * int(batch_size)], word2idx))
        return batch_data

    @staticmethod
    def pad_data(data, word2idx):
        text_list = []
        event_list = []
        targets = []
        max_length = max(
            [len(item[0].split(' ')) for item in data] + [len(item[1].split(' ')) for item in data])  # len(data[-1][0])
        for line in data:
            text, event, target = line

            text = text.split(' ')
            # if len(text) > config.SEQ_LENGTH:
            #     new_text = text[:config.SEQ_LENGTH]
            # else:
            #     text_padding = ['<PAD>'] * (config.SEQ_LENGTH - len(text))
            #     new_text = text + text_padding
            text_padding = ['<PAD>'] * (max_length - len(text))  # 不满max_length填充0
            new_text = text + text_padding
            new_text = [word2idx[word] for word in new_text]

            event = event.split(' ')
            # if len(event) > config.SEQ_LENGTH:
            #     new_event = event[:config.SEQ_LENGTH]
            # else:
            #     event_padding = ['<PAD>'] * (config.SEQ_LENGTH - len(event))
            #     new_event = event + event_padding
            event_padding = ['<PAD>'] * (max_length - len(event))  # 不满max_length填充0
            new_event = event + event_padding
            new_event = [word2idx[word] for word in new_event]

            text_list.append(new_text)
            event_list.append(new_event)
            targets.append(target)
        return [text_list, event_list, targets]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]
