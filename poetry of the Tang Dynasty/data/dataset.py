from torch.utils.data import Dataset,DataLoader
import numpy as np
from torch import nn
import string
class PoetryDataset(Dataset):
    def __init__(self,root):
        self.data = np.load(root,allow_pickle=True)
    def __len__(self):
        return len(self.data["data"])
    def __getitem__(self,index):
        return self.data["data"][index]
    def getData(self):
        return self.data["data"],self.data["ix2word"].item(),self.data["word2ix"].item()
if __name__=="__main__":
    datas=PoetryDataset("./tang.npz").data
    print(datas["data"].shape)
    #使用item将numpy数组转换成字典类型，ix2word是每一个数字对应一个字
    ix2word = datas['ix2word'].item()
    print(ix2word)
    word2ix = datas['word2ix'].item()
    print(word2ix)
    #将一首故事转化成索引
    sentence="床前明月光，疑似地上霜"
    print(word2ix['明'])
    print([word2ix[i] for i in sentence])
    #将第一首古诗打印出来
    print(datas["data"][0])
    print([ix2word[i] for i in datas["data"][0]])
