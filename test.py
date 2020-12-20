import torch

from eval import evaluate
from model import SiameseNetwork
from train import prepare_data

if __name__ == '__main__':
    embed_matrix = prepare_data()
    model = SiameseNetwork(embed_matrix)
    model.load_state_dict(torch.load('checkpoint.pt'))
    model.eval()
    acc, pre, rec, f1 = evaluate('test', model, type='model')
    print('正确率：%.3f 准确率：%.3f 召回率：%.3f F1得分：%.3f' % (acc, pre, rec, f1))