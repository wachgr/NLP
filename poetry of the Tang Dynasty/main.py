from logging import critical

import fire
import torch.nn as nn
import torch
from click.core import batch
from data.dataset import PoetryDataset
from models.model import Net
from torch.utils.data import DataLoader
num_epochs = 5
data_root = "./data/tang.npz"
batch_size = 10


def train(**kwargs):
    datasets = PoetryDataset(data_root)
    data, ix2word, word2ix = datasets.getData()
    lenData = len(data)
    print(lenData)
    print(type(data))
    data = torch.from_numpy(data)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=1)
    model = Net(len(word2ix), 128, 256)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    iteri = 0
    filename = "example.txt"
    totalIter = lenData * num_epochs / batch_size
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            # 将一个张量转换为长整型，然后转置其维度，确保数据在内存中连续存储，并将数据移动到GPU上以便加速计算。
            data = data.long().transpose(0, 1).contiguous().cuda()
            # print(type(data))
            optimizer.zero_grad()
            input, target = (data[:-1, :]), (data[1:, :])
            print(target.shape)
            print(target.view(-1).shape)

            outputs, _ = model(input)
            print(outputs.shape)
            loss = criterion(outputs, target.view(-1))
            loss.backward()
            optimizer.step()
            iteri += 1
            if (iteri % 500 == 0):
                print(str(iteri + 1) + "/" + str(totalIter) + "epoch")

            if (1 + i) % 1000 == 0:  # 每575个batch可视化一次
                with open(filename, 'a') as file:
                    file.write(str(i) + ':' + generate(model, '床前明月光', ix2word, word2ix) + "\n")
        torch.save(model.state_dict(), 'checkpoints/model_poet_2.pth')


def generate(model,start_words,ix2word,word2ix):
    txt = []
    for word in start_words:
        txt.append(word)
    input = torch.Tensor([word2ix['<START>']]).view(1,1).long()
    input= input.cuda()
    hidden = None
    num = len(txt)
    for i in range(48):
        output,hidden = model(input,hidden)
        if i < num:
            w = txt[i]
            input = (input.data.new([word2ix[w]])).view(1,1)
        else:
            top_index = output.data[0].topk(1)[1][0]
            w = ix2word[top_index.item()]
            txt.append(w)
            input = (input.data.new([top_index])).view(1,1)
        if w=='<EOP>':
            break
    return ''.join(txt)

def test():
    datasets = PoetryDataset(data_root)
    data, ix2word, word2ix = datasets.getData()
    modle = Net(len(word2ix), 128, 256)  # 模型定义：vocab_size, embedding_dim, hidden_dim —— 8293 * 128 * 256
    if torch.cuda.is_available() == True:
        modle.cuda()
        modle.load_state_dict(torch.load('./checkpoints/model_poet_2.pth'))
        modle.eval()
        name = input("请输入您的开头：")
        txt = generate(modle, name, ix2word, word2ix)
        print(txt)



if __name__=="__main__":
    fire.Fire()




