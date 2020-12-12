import math
import torch
import torch.nn as nn
import config
import torch.nn.functional as F


class TextCNNEncoder(nn.Module):
    """
    使用TextCNN对输入进行编码
    """

    def __init__(self, output_size):
        super(TextCNNEncoder, self).__init__()
        input_size = config.PROJ_DIM if config.PROJ else config.EMBED_DIM
        D = input_size  # 嵌入维度
        # C = output_size  # 输出维度
        Ci = 1  # 通道数
        Co = int(output_size / 3)
        Ks = [3, 4, 5]  # 卷积核的size为(K,input_size)
        self.convs1 = nn.ModuleList(
            [nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D), padding=(1, 0)) for K in Ks])
        self.dropout = nn.Dropout(0.8)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # x = self.embed(x)  # (N, W, D)
        # input (128,14,300) (batch_size,seq_len,embed_dim)
        x = x.unsqueeze(1)  # (128,1,30,300)
        x = [F.relu(conv(x)).squeeze(3) for conv in
             self.convs1]  # [(seq_len, Co, W), ...]*len(Ks)  [(128,200,30),(128,200,29),(128,200,28)]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in
             x]  # [(seq_len, Co), ...]*len(Ks)  [(128,200),(128,200),(128,200)]
        x = torch.cat(x, 1)  # (128,600) (seq_len,output_size)
        out = self.dropout(x)  # (N, len(Ks)*Co)
        return out


# 双向LSTM模型+池化
class TextRCNNEncoder(nn.Module):
    def __init__(self):
        super(TextRCNNEncoder, self).__init__()
        self.input_size = config.PROJ_DIM if config.PROJ else config.EMBED_DIM
        self.rnn = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=self.input_size,
            hidden_size=config.HIDDEN_DIM,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            dropout=0.5,
            bidirectional=True
        )
        # self.maxpool = nn.MaxPool1d()
        self.fc = nn.Linear(config.HIDDEN_DIM * 2 + self.input_size, config.HIDDEN_DIM * 2)

    def forward(self, x):
        # x:  (batch_size,seq_len,input_size)  (128,54,300)
        out, _ = self.rnn(x, None)  # (batch_size,seq_len,hidden_dim*2)  (128,39,600)
        out = torch.cat((x, out), 2)  # (128,54,900)
        out = F.relu(out)
        out = out.permute(0, 2, 1)  # (128,900,54)
        out = F.max_pool1d(out, out.size(2))  # 对最后一个维度做maxpooling，卷积核大小为(1,out.size(2)) 得到(128,900,1)
        out = out.squeeze()  # (128,900)
        out = self.fc(out)
        return out


class LSTMEncoder(nn.Module):

    def __init__(self):
        super(LSTMEncoder, self).__init__()
        input_size = config.PROJ_DIM if config.PROJ else config.EMBED_DIM
        dropout = 0 if config.N_LAYERS == 1 else config.DROPOUT_RATE
        # BiLSTM
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=config.HIDDEN_DIM,
                           num_layers=config.N_LAYERS, batch_first=True, dropout=dropout,
                           bidirectional=config.BI)

        # GRU
        # self.rnn = nn.GRU(input_size=input_size, hidden_size=config.HIDDEN_DIM,
        #                   num_layers=config.N_LAYERS, batch_first=True, dropout=dropout,
        #                   bidirectional=config.BI)

        self.dropout = nn.Dropout(0.5)
        if config.BI:
            self.w_omega = nn.Parameter(torch.Tensor(
                config.HIDDEN_DIM * 2, config.HIDDEN_DIM * 2))
            self.u_omega = nn.Parameter(torch.Tensor(config.HIDDEN_DIM * 2, 1))
        else:
            self.w_omega = nn.Parameter(torch.Tensor(
                config.HIDDEN_DIM, config.HIDDEN_DIM))
            self.u_omega = nn.Parameter(torch.Tensor(config.HIDDEN_DIM, 1))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, inputs):
        # inputs:[batch_size,seq_length,embed_dim]
        outputs, hn = self.rnn(inputs)  # outpus:[batch_size,seq_length,HIDDEN_DIM*2]
        if config.ATTENTION:
            # 加入注意力机制
            # 1. 将query和每个key进行相似度计算得到权重
            u = torch.tanh(torch.matmul(outputs, self.w_omega))  # (batch_size, seq_len, 2 * HIDDEN_DIM)
            att = torch.matmul(u, self.u_omega)  # (batch_size, seq_len, 1)
            # 2. 使用softmax函数对这些权重进行归一化
            alpha = F.softmax(att, dim=1)  # (batch_size, seq_len, 1)
            # 3. 将权重和相应的键值value进行加权求和得到最后的attention
            scored_out = outputs * alpha  # (batch_size, seq_len, 2 * HIDDEN_DIM)
            output = torch.sum(scored_out, dim=1)  # (batch_size,2 * HIDDEN_DIM)
        else:
            output = outputs[:, -1, :]
        return output


class SiameseNetwork(nn.Module):

    def __init__(self, embed_weight):
        super(SiameseNetwork, self).__init__()
        self.vocab_size = embed_weight.shape[0]
        self.embed = nn.Embedding(self.vocab_size, config.EMBED_DIM)
        self.embed.weight.data.copy_(embed_weight)
        self.projection = nn.Linear(config.EMBED_DIM, config.PROJ_DIM)
        # 使用bilstm编码
        self.encoder = LSTMEncoder()

        # 使用TextCNN编码
        # self.encoder = TextCNNEncoder(config.HIDDEN_DIM * 2)

        # 使用TextRCNN编码
        # self.encoder = TextRCNNEncoder()

        self.dropout = nn.Dropout(p=config.DROPOUT_RATE)
        self.relu = nn.ReLU()
        seq_in_size = 2 * config.HIDDEN_DIM
        if config.BI:
            seq_in_size *= 2
        self.out = nn.Sequential(
            nn.Linear(seq_in_size, 512),
            self.relu,
            self.dropout,
            nn.Linear(512, 2))

    def forward_once(self, input):
        embeded = self.embed(input)
        if config.FIX_EMDED:
            embeded = embeded.detach()
        if config.PROJ:
            embeded = self.relu(self.projection(embeded))
        encoded = self.encoder(embeded)
        return encoded

    def forward(self, input1, input2):
        premise = self.forward_once(input1)  # [batch_size,hidden_dim*2]
        hypothesis = self.forward_once(input2)
        scores = self.out(torch.cat([premise, hypothesis], 1))  # [batch_size,2]
        return scores
