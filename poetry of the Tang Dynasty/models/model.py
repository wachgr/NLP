from torch.utils.data import Dataset,DataLoader
import numpy as np
from torch import nn
import string
    #定义模型
"""
vocab_size表示输入单词的格式
embedding_dim表示将一个单词映射到embedding_dim维度空间
hidden_dim表示lstm输出隐藏层的维度
"""
class Net(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim):
        super(Net,self).__init__()
        self.hidden_dim=hidden_dim
        #embedding层,每个单词将被映射到的连续空间的维度
        self.embeddings=nn.Embedding(vocab_size,embedding_dim)
        #两层lstm
        self.lstm = nn.LSTM(embedding_dim,self.hidden_dim,num_layers=2,batch_first=False)
        self.linear1 = nn.Linear(self.hidden_dim,vocab_size)
    def forward(self,input,hidden=None):
        #获取输入数据的时间步和批次数：
        seq_len,batch_size = input.size()
        #如果没有传入上一个时间的隐藏值，初始化一个，注意是两层
        if hidden is None:
            h_0 = input.data.new(2,batch_size,self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(2,batch_size,self.hidden_dim).fill_(0).float()
        else:
            h_0,c_0 = hidden
        #对于 input 中的每个元素（即每个单词索引），查找对应的嵌入向量。
        embeds = self.embeddings(input)
        print(embeds.shape)
        #embeds 的形状将是 (seq_len, batch_size, embedding_dim),seq_len其实就是data中每一行的数字个数
        output,hidden = self.lstm(embeds,(h_0,c_0))
        output = self.linear1(output.view(seq_len*batch_size,-1))#所有时间步的数据“扁平化”到一个二维张量中，其中第一维是序列元素的总数（即时间步和批量的乘积），第二维是每个序列元素的特征数量。
        return output,hidden
