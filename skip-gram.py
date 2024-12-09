#构建词表
from collections import defaultdict

from pygments.lexer import words
import nltk

class Vocab:
    def __init__(self,tokens=None):
        self.idx_to_token = list()
        self.token_to_idx = dict()

        if tokens is not None:
            if "<unk>" not in tokens:
                tokens.append("<unk>")
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token]=len(self.idx_to_token)-1
            self.unk=self.token_to_idx["<unk>"]



    """build函数用于个句子添加一些标记，比如<BOS>表示句子开头，<unk>表示未知词或低频词，
    其中min_frep表示最低词频，词频低于min_frep的单词默认为低频词，不参与词表的映射构建，减少了词表的大小
    和后期训练的计算开销"""
    @classmethod
    def build(cls,text,min_frep=1,reserved_tokens=None):
        token_freps=defaultdict(int)
        for sentence in text:
            for token in sentence:
                token_freps[token] += 1
        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        uniq_tokens += [token for token, freq in token_freps.items() if freq >=min_frep and token !="<unk>"]
        return cls(uniq_tokens)

    def __len__(self):
        return len(self.idx_to_token)
    def __getitem__(self,token):
        return self.token_to_idx.get(token,self.unk)
    def convert_token_to_idx(self,tokens):
        #查找一系列输入词的索引
        return [self.token_to_idx[token] for token in tokens]
    def convert_idx_to_token(self,indics):
        #查找一系列索引值对应的输入
        return [self.idx_to_token[index] for index in indics]

BOS_TOKEN = "<BOS>" #句首标记
EOS_TOKEN = "<EOS>" #句尾标记
PAD_TOKEN = "<PAD>" #填充标记

def load_reuters():
    from nltk.corpus import reuters
    text=reuters.sents()
    #将所有词都小写处理
    text=[[token.lower() for token in sentence] for sentence in text]
    vocab=Vocab.build(text,reserved_tokens=[BOS_TOKEN,EOS_TOKEN,PAD_TOKEN])
    corpus=[vocab.convert_token_to_idx(sentence) for sentence in text]
    return corpus,vocab

#每个单词出现的频率
def get_unigram_distribution(corpus,vocab_size):
    token_count=torch.tensor([0]*vocab_size)
    total_count=0
    for sentence in corpus:
        total_count += len(sentence)
        for token in sentence:
            token_count[token] += 1
    unigram_dist = torch.div(token_count.float(),total_count)
    return unigram_dist

from torch.utils.data import Dataset,DataLoader
import torch
from torch import nn,optim
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

#skip-gram模型数据库的构建
class SkipGramDataset(Dataset):
    def __init__(self,corpus,vocab,context_size,n_negatives,ns_dist):
        self.data=[]
        self.bos=vocab[BOS_TOKEN]
        self.eos=vocab[EOS_TOKEN]
        self.pad=vocab[PAD_TOKEN]
        for sentence in tqdm(corpus,desc='Dataset Construction'):
            sentence=[self.bos]+sentence+[self.eos]
            for i in range(1,len(sentence)-1):
                w=sentence[i]
                #确定上下文左右边界，不够的地方用pad填充
                left_index=max(0,i-context_size)
                right_index=min(len(sentence),i+context_size)
                context=sentence[left_index:i]+sentence[i+1:right_index+1]
                context+=[self.pad] * (context_size * 2 - len(context))
                self.data.append((w,context))
        #正样本与负样本的比例
        self.n_negatives=n_negatives
        #负样本的分布
        self.ns_dist=ns_dist

    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        return self.data[index]
    def collate_fn(self,batch_datas):
        words=torch.tensor([batch[0] for batch in batch_datas],dtype=torch.long)
        contexts=torch.tensor([batch[1] for batch in batch_datas],dtype=torch.long)
        batch_size,context_size=contexts.shape
        neg_contexts=[]
        for i in range(batch_size):
            ns_dist=self.ns_dist.index_fill(0,contexts[i],.0)
            neg_contexts.append(torch.multinomial(ns_dist,context_size*self.n_negatives,replacement=True))
        neg_contexts=torch.stack(neg_contexts,dim=0)
        return words,contexts,neg_contexts

class SkipGramModule(nn.Module):
    def __init__(self,vocab_size,embedding_dim):
        super().__init__()
        self.w_embedding = nn.Embedding(vocab_size,embedding_dim)
        self.c_embedding = nn.Embedding(vocab_size,embedding_dim)
    def forward_w(self,words):
        w_embeds = self.w_embedding(words)
        return w_embeds
    def forward_c(self,contexts):
        c_embeds = self.c_embedding(contexts)

embedding_size=128
batch_size=32
num_epoch=10
context_size=3
n_negatives=5
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


corpus,vocab=load_reuters()
unigram_dist=get_unigram_distribution(corpus,len(vocab))
negative_sample_dist=unigram_dist**0.75
negative_sample_dist /= negative_sample_dist.sum()

dataset = SkipGramDataset(corpus,vocab,context_size=context_size,n_negatives=n_negatives,ns_dist=negative_sample_dist)
dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,collate_fn=dataset.collate_fn)

model=SkipGramModule(len(vocab),embedding_size).to(device)
optimizer=optim.Adam(model.parameters(),lr=0.001)

model.train()
for epoch in range(num_epoch):
    total_loss=0
    for batch in tqdm(dataloader,desc=f"Training Epoch {epoch}"):
        words,contexts,neg_contexts=[x.to(device) for x in batch]
        #去关键词、上下文、负样本的向量
        word_embeds=model.w_embedding(words).unsqueeze(dim=2)
        context_embeds=model.c_embedding(contexts)
        neg_context_embeds=model.c_embedding(neg_contexts)

        context_loss = F.logsigmoid(torch.matmul(context_embeds,word_embeds).squeeze(dim=2))
        context_loss = context_loss.mean(dim=1)
        neg_context_loss=F.logsigmoid(torch.matmul(neg_context_embeds,word_embeds).squeeze(dim=2).neg())
        neg_context_loss=neg_context_loss.mean(dim=1)
        loss=-(context_loss+neg_context_loss).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"loss:{total_loss:.2f}")







