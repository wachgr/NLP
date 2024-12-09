from nltk import log_likelihood
from nltk.chat.eliza import pairs
from utils import process_tweet, lookup
import pdb
from nltk.corpus import stopwords, twitter_samples
import numpy as np
import pandas as pd
import nltk
import string
from nltk.tokenize import TweetTokenizer
from os import getcwd
# get the sets of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# 训练集与测试集
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

# 标签
train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))

def count_tweets(result,tweets,ys):
    yslist = np.squeeze(ys).tolist()
    for tweet,y in zip(tweets,yslist):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in result:
                result[pair] += 1
            else:
                result[pair]=1
    return result

# result = {}
# tweets = ['i am happy', 'i am tricked', 'i am sad', 'i am tired', 'i am tired']
# ys = [1, 0, 0, 0, 0]
# result=count_tweets(result, tweets, ys)
# print(result)
# stopwords_english = stopwords.words('english')
# print(stopwords_english)

def train_naive_bayes(freqs,train_x,train_y):
    loglikelihood = {}
    logprior = 0
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)
    N_pos=N_neg=0
    for pair in freqs.keys():
        if pair[1] >0:
            N_pos += freqs[pair]
        else:
            N_neg += freqs[pair]
    # 计算积极句子的数量和消极句子的数量
    D = len(train_y)
    D_pos = np.sum(train_y)
    D_neg = D-D_pos

    logprior = np.log(D_pos)-np.log(D_neg)

    for word in vocab:
        freq_pos = lookup(freqs, word, 1)
        freq_neg = lookup(freqs, word, 0)
        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)
        loglikelihood[word] = np.log(p_w_pos / p_w_neg)
    return logprior, loglikelihood

def naive_bayes_predict(tweet,logprior,loglikelihood):
    word_1 = process_tweet(tweet)
    p=0
    p+=logprior
    for word in word_1:
        if word in loglikelihood:
            p += loglikelihood[word]
    return p

freqs = count_tweets({}, train_x, train_y)
my_tweet = 'She smiled.'
logprior,loglikelihood = train_naive_bayes(freqs,train_x,train_y)
p = naive_bayes_predict(my_tweet, logprior, loglikelihood)
print('The expected output is', p)


def test_naive_bayes(test_x, test_y, logprior, loglikelihood):
    y_hats = []
    for tweet in test_x:
        if naive_bayes_predict(tweet, logprior, loglikelihood) > 0:
            y_hat_i = 1
        else:
            y_hat_i = 0
        y_hats.append(y_hat_i)

    error = np.sum(np.abs(test_y - y_hats))/len(test_x)
    accuracy = 1 - error
    return accuracy

print("Navive Bayes accuracy = %0.4f" %(test_naive_bayes(test_x, test_y, logprior, loglikelihood)))