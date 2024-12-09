###单层lstm
import torch
from torch import nn
sequence_length = 3
batch_size = 2
input_size = 4
input = torch.randn(sequence_length,batch_size,input_size)
print(input)
lstmModel = nn.LSTM(input_size,hidden_size=3,num_layers=1)#num_layers=1隐藏层个数为1
output,(h,c) = lstmModel(input)
#因为有三个时间步，每个时间步有一个隐藏层，每个隐藏层都要有两条数据，隐藏层的维度是3，最终是（3，2，3）
print("LSTM隐藏层输出维度：",output.shape)
print("LSTM隐藏层最后一个时间步输出的维度",h.shape)
print("LSTM隐藏层最后一个时间步细胞状态",c.shape)
###双层lstm
import torch
sequence_length = 3
batch_size = 2
input_size = 4
hidden_dim = 3
vocab_size = 100
input = torch.randn(sequence_length,batch_size,input_size)
print(input.shape)
lstmModel = nn.LSTM(input_size,hidden_size=3,num_layers=2)#num_layers=1隐藏层个数为2
output,(h,c) = lstmModel(input)
#因为有三个时间步，每个时间步有一个隐藏层，每个隐藏层都要有两条数据，隐藏层的维度是3，最终是（3，2，3）
print("LSTM隐藏层输出维度：",output.shape)
print("LSTM隐藏层最后一个时间步输出的维度",h.shape)
print("LSTM隐藏层最后一个时间步细胞状态",c.shape)
output = output.view(sequence_length*batch_size,-1)
print(output.shape)
linear1 = nn.Linear(hidden_dim,vocab_size)
output = linear1(output)
print(output.shape)
