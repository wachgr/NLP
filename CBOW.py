#定义一个句子列表
from matplotlib.pyplot import figure
from transformers.models.cvt.convert_cvt_original_pytorch_checkpoint_to_pytorch import embeddings
import nltk
# nltk.download()
sentences = ["Kage is Teacher", " Mazong is Boss", "Niuzong is Boss",
             "Xiao is student", "xue is  student"]
words = ' '.join(sentences).split()
#建立一个词汇表
word_list = list(set(words))
#生成一个字典，每个词映射一个唯一的索引
word_to_idx = {word:idx for idx,word in enumerate(word_list)}
#每个索引映射一个词语
idx_to_word = {idx:word for idx,word in enumerate(word_list)}
voc_size=len(word_list)
print("词汇表：",word_list)
print("词汇到索引的字典：",word_to_idx)
print("索引到词汇的字典：",idx_to_word)
print("词汇表的大小：",voc_size)

#生成CBOW训练数据
def create_cbow_dataset(sentences,window_size=2):
    data=[]
    for sentence in sentences:
        sentence = sentence.split()
        for idx,word in enumerate(sentence):
            #获取上下词汇，将当前单词前后个window_size个单词作为周围词
            context_words=sentence[max(idx-window_size,0):idx]\
                +sentence[idx+1:min(idx+window_size+1,len(sentence))]
            #将当前单词与上下文词汇组成一组训练数据
            data.append((word,context_words))
    return data

#使用函数COBW训练数据
cbow_data = create_cbow_dataset(sentences)
print("CBOW 数据样例（未编码）：",cbow_data)


#定义One-hot编码
import torch
def one_hot_encoding(word, word_to_idx ):
    #创建一个长度与词汇表相同的全0张量
    tensor = torch.zeros(len(word_to_idx))
    tensor[word_to_idx[word]] =1
    return tensor
word_example = "is"

print("one_hot编码前的单词:" ,word_example)
print("one_hot编码后的单词：",one_hot_encoding(word_example ,word_to_idx))


#定义CBOW模型
import torch.nn as nn
class CBOW(nn.Module):
    def __init__(self,voc_size,embedding_size):
        super(CBOW,self).__init__()
        #从词汇表大小到嵌入大小的线性层（权重矩阵）
        self.input_to_hidden = nn.Linear(voc_size, embedding_size, bias=False)
        self.hidden_to_output = nn.Linear(embedding_size,voc_size,bias=False)

    def forward(self,X):#X[num_context_words:voc_size]
        embeddings = self.input_to_hidden(X)
        hidden_layer = torch.mean(embeddings,dim=0)
        output_layeer = self.hidden_to_output(hidden_layer.unsqueeze(0))
        return output_layeer

embedding_size = 2
cbow_model = CBOW(voc_size,embedding_size)
print("CBOW模型：",cbow_model)


#训练cbow模型
learning_rate = 0.001
epochs = 1000
#定义交叉熵损失函数
criterion = nn.CrossEntropyLoss()
import torch.optim as optim
optimizer = optim.SGD(cbow_model.parameters(),lr=learning_rate)
#开始循环训练
loss_values = []
for epoch in range(epochs):
    loss_sum = 0
    for target,context_words in cbow_data:
        #将上下文词转换为One-Hot向量堆叠
        X=torch.stack([one_hot_encoding(word,word_to_idx) for word in context_words]).float()
        #将目标词转换为索引
        y_true = torch.tensor([word_to_idx[target]],dtype=torch.long)
        print("y_true: ",y_true)
        y_pred = cbow_model(X)
        print("y_pred:", y_pred)
        loss = criterion(y_pred,y_true)
        loss_sum +=loss.item()
        #清空梯度
        optimizer.zero_grad()
        #反向传播
        loss.backward()
        #更新参数
        optimizer.step()
    if (epoch+1) % 100==0:
        print(f"Epoch:{epoch+1},loss:{loss_sum/len(cbow_data)}")
        loss_values.append(loss_sum/len(cbow_data))


#绘制训练损失曲线
import matplotlib.pyplot as plt
#绘制二维词向量图
plt.rcParams["font.family"] =["SimHei"]
plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False
plt.plot(range(1,epochs//100+1),loss_values)
plt.title('训练损失曲线')
plt.xlabel("轮次")
plt.ylabel("损失")
plt.show()

#输入cbow习得的词嵌入
print("CBOW 词嵌入：")
for word, idx in word_to_idx.items():
    print(f"{word}:{cbow_model.input_to_hidden.weight[:,idx].detach().numpy()}")

fig, ax=plt.subplots()
for word, idx in word_to_idx.items():
    #获取每个单词的嵌入向量
    vec = cbow_model.input_to_hidden.weight[:,idx].detach().numpy()
    ax.scatter(vec[0],vec[1])
    ax.annotate(word,(vec[0],vec[1]),fontsize=12)

plt.title("二维词嵌入")
plt.xlabel("向量维度1")
plt.ylabel("向量维度2")
plt.show()



