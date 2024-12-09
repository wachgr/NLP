from baselll import *
import nltk
from fontTools.ttx import process
from nltk.corpus import twitter_samples    # sample Twitter dataset from NLTK
import matplotlib.pyplot as plt
import random
import re
import string
import numpy as np
from os import getcwd
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

process = baselll()
# tweets = all_positive_tweets + all_negative_tweets
# labels = np.append(np.ones((len(all_positive_tweets))), np.zeros((len(all_negative_tweets))))
# freqs = process.build_freqs(tweets, labels)
#
# tweet = all_positive_tweets[2277]
# print('\033[92m',tweet)
# print('\033[94m')
# tweets_stem = process.process_tweet(tweet)
# print('preprocessed tweet:',tweets_stem)
#
# data = []
# for word in tweets_stem:
#     pos = 0
#     neg = 0
#     if (word, 1) in freqs:
#         pos = freqs[(word, 1)]
#     if (word, 0) in freqs:
#         neg = freqs[(word, 0)]
#     data.append([word, pos, neg])
# print(data)
#
# fig,ax = plt.subplots(figsize=(8,8))
# x = np.log([x[1]+1 for x in data])
# y = np.log([x[2]+1 for x in data])
# ax.scatter(x,y)
# plt.xlabel('Log Positive count')
# plt.ylabel('Log Negative count')
# for i in range(0,len(data)):
#     ax.annotate(data[i][0],(x[i],y[i]),fontsize=12)
#
# ax.plot([0,9],[0,9],color='red')
# plt.show()


#划分训练集和测试集
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg
#标签
train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)
print("train_y.shape = " + str(train_y.shape))
print("test_y.shape = " + str(test_y.shape))

#得到频率字典包括正频率和负频率
freqs = process.build_freqs(train_x, train_y)


def sigmoid(z):
    h = 1 / (1 + np.exp(-z))
    return h

def gradientDescent(x, y, theta, alpha, num_iters):
    '''
    Input:
        x: matrix of features which is (m,n+1)
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations you want to train your model for
    Output:
        J: the final cost
        theta: your final weight vector
    Hint: you might want to print the cost to make sure that it is going down.
    '''
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    # get 'm', the number of rows in matrix x
    m = x.shape[0]

    for i in range(0, num_iters):
        z = np.dot(x, theta)
        h = sigmoid(z)
        J = -(np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h))) / m
        theta = theta - alpha * (np.dot(x.T, (h - y))) / m

    J = float(J)
    return J, theta

#求每句话的X
def extract_features(tweet, freqs):
    word_1=process.process_tweet(tweet)
    x = np.zeros((1, 3))
    x[0, 0] = 1
    for word in word_1:
        # increment the word count for the positive label 1
        x[0, 1] += freqs.get((word, 1.0), 0)
        # increment the word count for the negative label 0
        x[0, 2] += freqs.get((word, 0.0), 0)

    assert (x.shape == (1, 3))
    return x

X = np.zeros((len(train_x),3))
for i in range(len(train_x)):
    X[i,:] = extract_features(train_x[i],freqs)

Y = train_y
J,theta = gradientDescent(X,Y,np.zeros((3,1)),1e-9,1500)
print(f"The cost after training is {J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")


def predict(tweet,freps,theda):
    x=extract_features(tweet,freqs)
    y_pred = sigmoid(np.dot(x,theda))
    return y_pred

for tweet in ['I am happy', 'I am bad', 'this movie should have been great.', 'great', 'great great', 'great great great', 'great great great great']:
    print( '%s -> %f' % (tweet, predict(tweet, freqs, theta)))


def test_logistic_regression(test_x, test_y, freqs, theta):
    y_hat = []
    for tweet in test_x:
        y_pred = predict(tweet, freqs, theta)
        if y_pred > 0.5:
            # append 1.0 to the list
            y_hat.append(1.0)
        else:
            # append 0 to the list
            y_hat.append(0)
    accuracy = np.sum((np.array(y_hat) == test_y.flatten()) != 0) / len(y_hat)

    return accuracy

tmp_accuracy = test_logistic_regression(test_x, test_y, freqs, theta)
print(f"Logistic regression model's accuracy = {tmp_accuracy:.4f}")


my_tweet = 'This is a ridiculously bright movie. The plot was terrible and I was sad until the ending!'
print(process.process_tweet(my_tweet))
y_hat = predict(my_tweet, freqs, theta)
print(y_hat)
if y_hat > 0.5:
    print('Positive sentiment')
else:
    print('Negative sentiment')