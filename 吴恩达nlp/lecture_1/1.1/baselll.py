import nltk
from nltk.corpus import twitter_samples    # sample Twitter dataset from NLTK
import matplotlib.pyplot as plt
import random
import re
import string
import numpy as np

from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer   # module for tokenizing strings
"""
about the dataset : 
The sample dataset from NLTK is separated into positive and negative tweets. 
It contains 5000 positive tweets and 5000 negative tweets exactly. 
"""
# # nltk.download('twitter_samples')
# all_positive_tweets = twitter_samples.strings('positive_tweets.json')
# all_negative_tweets = twitter_samples.strings('negative_tweets.json')
# # print(all_negative_tweets)
# print("Number of positive tweets:",len(all_positive_tweets))
# print("Number of positive tweets:",len(all_negative_tweets))
# print("\n The type of all_positive_tweets is:", type(all_positive_tweets))
# print("The type of a tweet entry is:", type(all_positive_tweets))
# print("The first list:",all_positive_tweets[0])
#
# # #绘制一个饼图,直观了解数据的分布
# # fig = plt.figure(figsize=(5,5))
# # labels = 'Positive','Negative'
# # sizes = [len(all_positive_tweets),len(all_negative_tweets)]
# # plt.pie(sizes,labels=labels,autopct='%1.1f%%',shadow=True,startangle=90)
# # plt.axis('equal')
# # plt.show()
#
# #looking at raw texts
# print('\033[92m' + all_positive_tweets[random.randint(0,5000)])#绿色
# print('\033[91m' + all_negative_tweets[random.randint(0,5000)])#红色
#
#
#
# # Tokenizing the string 标记字符串
# # Lowercasing
# # Removing stop words and punctuation 除去终止词和标点符号
# # Stemming 词干(大小写呀，时态呀什么的)
#
#
# # download the stopwords from NLTK
# # nltk.download('stopwords')
#
#
# import re
# import string
#
# from nltk.corpus import stopwords          # module for stop words that come with NLTK
# from nltk.stem import PorterStemmer        # module for stemming
# from nltk.tokenize import TweetTokenizer   # module for tokenizing strings
#
# tweet = all_positive_tweets[2277]
# print(tweet)
# print('\033[92m' + tweet)
# print('\033[94m')
# # remove old style retweet text "RT"
# tweet2 = re.sub(r'^RT[\s]+', '', tweet)
# # remove hyperlinks，url
# tweet2 = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet2)
# # remove hashtags
# # only removing the hash # sign from the word
# tweet2 = re.sub(r'#', '', tweet2)
# print(tweet2)
#
# tokenizer = TweetTokenizer(preserve_case=False,strip_handles=True,reduce_len=True)
# tweet_tokens = tokenizer.tokenize(tweet2)
# print('Tokenized string:', tweet_tokens)
# #结果 Tokenized string: ['my', 'beautiful', 'sunflowers', 'on', 'a', 'sunny', 'friday', 'morning', 'off', ':)', 'sunflowers', 'favourites', 'happy', 'friday', 'off', '…']
#
# stopwords_english = stopwords.words('english')
# print('stop word：\n',stopwords_english)
# print('Punctuation:' ,string.punctuation)
#
# tweets_clean=[]
# for word in tweet_tokens:
#     if (word not in stopwords_english and word not in string.punctuation):
#         tweets_clean.append(word)
#
# print('removed stop words and punctuation:' ,tweets_clean)
#
# stemmer = PorterStemmer()
# tweets_stem=[]
# for word in tweets_clean:
#     stem_word=stemmer.stem(word)
#     tweets_stem.append(stem_word)
# print('stemmed words:', tweets_stem)
#
#
# from utils import process_tweet # Import the process_tweet function
# # choose the same tweet
# tweet = all_positive_tweets[2277]
# print()
# print('\033[92m')
# print(tweet)
# print('\033[94m')
# # call the imported function
# tweets_stem = process_tweet(tweet); # Preprocess a given tweet
# print('preprocessed tweet:')
# print(tweets_stem) # Print the




class baselll:
    def __init__(self):
        pass
    def process_tweet(self,tweet):
        #除去RT
        tweet2 = re.sub(r'^RT[\s]+', '', tweet)
        # remove hyperlinks，url
        tweet2 = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet2)
        # remove hashtags
        # only removing the hash # sign from the word
        tweet2 = re.sub(r'#', '', tweet2)
        #字符串便变数组
        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
        tweet_tokens = tokenizer.tokenize(tweet2)
        # 结果 Tokenized string: ['my', 'beautiful', 'sunflowers', 'on', 'a', 'sunny', 'friday', 'morning', 'off', ':)', 'sunflowers', 'favourites', 'happy', 'friday', 'off', '…']
        #1、除去终止词和标点符号
        stopwords_english = stopwords.words('english')
        tweets_clean = []
        for word in tweet_tokens:
            if (word not in stopwords_english and word not in string.punctuation):
                tweets_clean.append(word)
        #2、提取词干(大小写呀，时态呀什么的)
        stemmer = PorterStemmer()
        tweets_stem = []
        for word in tweets_clean:
            stem_word = stemmer.stem(word)
            tweets_stem.append(stem_word)

        return tweets_stem
    def build_freqs(self,tweets,ys):
        yslist = np.squeeze(ys).tolist()
        freqs = {}
        for y, tweet in zip(yslist,tweets):
            for word in self.process_tweet(tweet):
                pair = (word, y)
                if pair in freqs:
                    freqs[pair] += 1
                else:
                    freqs[pair] = 1
        return freqs





