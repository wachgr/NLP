import math
import random
import numpy as np
import pandas as pd
import nltk
from pycparser.ply.yacc import token
from scipy.cluster.hierarchy import to_mlab_linkage


###part1
nltk.data.path.append('.')
#切分段落为句子数组
def split_to_sentences(data):
    sentences = data.split('\n')
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if len(s)>0]
    return sentences
x = """
I have a pen.\nI have an apple. \nAh\nApple pen.\n
"""
print(x)
split_to_sentences(x)
#将句子转化成单词数组
def tokenize_sentences(sentences):
    tokenized_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        tokenize = nltk.word_tokenize(sentence)
        tokenized_sentences.append(tokenize)
    return tokenized_sentences


#将上面给两个函数结合
def get_tokenized_data(data):
    sentences = split_to_sentences(data)
    tokenized_sentences = tokenize_sentences(sentences)
    return tokenized_sentences
x = "Sky is blue.\nLeaves are green\nRoses are red."
tokenized_sentences = get_tokenized_data(x)
print(get_tokenized_data(x))

def count_words(tokenized_sentences):
    word_counts={}
    for sentence in tokenized_sentences:
        for token in sentence:
            if token not in word_counts:
                word_counts[token] = 1
            else:
                word_counts[token] += 1
    return word_counts

# count_words=count_words(tokenized_sentences)
# print(count_words)
#除去超过count_threshold值的单词
def get_words_with_nplus_frequency(tokenized_sentences,count_threshold):
    closed_vocab = []
    word_counts = count_words(tokenized_sentences)
    for word,cnt in word_counts.items():
        if cnt >= count_threshold:
            closed_vocab.append(word)
    return closed_vocab
tmp_closed_words = get_words_with_nplus_frequency(tokenized_sentences,count_threshold=2)
print("Closed vocabulary:",tmp_closed_words)
def replace_oov_words_by_unk(tokenized_sentences,vocabulary,unknown_token="<unk>"):
    vocabulary=set(vocabulary)
    replaced_tokenized_sentences = []
    for sentence in tokenized_sentences:
        replaced_sentence = []
        for token in sentence:
            if token in vocabulary:
                replaced_sentence.append(token)
            else:
                replaced_sentence.append(unknown_token)
        replaced_tokenized_sentences.append(replaced_sentence)
    return replaced_tokenized_sentences
tokenized_sentences = [["dogs", "run"], ["cats", "sleep"]]
vocabulary = ["dogs", "sleep"]
tmp_replaced_tokenized_sentences = replace_oov_words_by_unk(tokenized_sentences, vocabulary)
print(f"Original sentence:")
print(tokenized_sentences)
print(f"tokenized_sentences with less frequent words converted to '<unk>':")
print(tmp_replaced_tokenized_sentences)

def preprocess_data(train_data,test_data,count_threshold):
    vocabulary = get_words_with_nplus_frequency(train_data,count_threshold)
    train_data_replaced = replace_oov_words_by_unk(train_data,vocabulary,unknown_token="<unk>")
    test_data_replaced = replace_oov_words_by_unk(test_data, vocabulary, unknown_token="<unk>")
    return train_data_replaced, test_data_replaced, vocabulary
tmp_train = [['sky', 'is', 'blue', '.'],
     ['leaves', 'are', 'green']]
tmp_test = [['roses', 'are', 'red', '.']]
tmp_train_repl, tmp_test_repl, tmp_vocab = preprocess_data(tmp_train,
                                                           tmp_test,
                                                           count_threshold = 1)

print("tmp_train_repl")
print(tmp_train_repl)
print()
print("tmp_test_repl")
print(tmp_test_repl)
print()
print("tmp_vocab")
print(tmp_vocab)

###part2
#按照n划分句子
def count_n_grams(data,n,start_token='<s>',end_token='<e>'):
    n_grams = {}
    for sentence in data:
        sentence=[start_token]*(n)+sentence+[end_token]
        sentence = tuple(sentence)
        for i in range(len(sentence)-n+1):
            n_gram = sentence[i:i+n]
            if n_gram in n_grams:
                n_grams[n_gram] += 1
            else:
                n_grams[n_gram] = 1
    return n_grams

sentences = [['i', 'like', 'a', 'cat'],
             ['this', 'dog', 'is', 'like', 'a', 'cat']]
print("Uni-gram:")
print(count_n_grams(sentences, 1))
print("Bi-gram:")
print(count_n_grams(sentences, 2))

# Uni-gram:
# {('<s>',): 2, ('i',): 1, ('like',): 2, ('a',): 2, ('cat',): 2, ('<e>',): 2, ('this',): 1, ('dog',): 1, ('is',): 1}
# Bi-gram:
# {('<s>', '<s>'): 2, ('<s>', 'i'): 1, ('i', 'like'): 1, ('like', 'a'): 2, ('a', 'cat'): 2, ('cat', '<e>'): 2, ('<s>', 'this'): 1, ('this', 'dog'): 1, ('dog', 'is'): 1, ('is', 'like'): 1}
#计算P(Wn|Wn-1)
def estimate_probability(word,previous_n_gram,n_gram_counts,n_plus1_gram_counts,vocabulary_size,k=1.0):
    previous_n_gram = tuple(previous_n_gram)
    previous_n_gram_count = n_gram_counts.get(previous_n_gram,0)
    #k*vocabulary_size是进行光滑处理
    denominator = previous_n_gram_count + k*vocabulary_size
    n_plus1_gram = previous_n_gram + tuple([word])
    n_plus1_gram_count = n_plus1_gram_counts.get(n_plus1_gram,0)
    #k是进行光滑处理
    numerator = n_plus1_gram_count + k
    probability = numerator / denominator
    return probability
sentences = [['i', 'like', 'a', 'cat'],
             ['this', 'dog', 'is', 'like', 'a', 'cat']]
unique_words = list(set(sentences[0] + sentences[1]))
unigram_counts = count_n_grams(sentences, 1)
bigram_counts = count_n_grams(sentences, 2)

tmp_prob = estimate_probability("cat", "a", unigram_counts, bigram_counts, len(unique_words), k=1)

print(f"The estimated probability of word 'cat' given the previous n-gram 'a' is: {tmp_prob:.4f}")
def estimate_probabilities(previous_n_gram,n_gram_counts,n_plus1_gram_counts,vocabulary,k=1.0):
    previous_n_gram = tuple(previous_n_gram)
    vocabulary = vocabulary + ["<e>","<unk>"]
    vocabulary_size = len(vocabulary)
    probabilities = {}
    for word in vocabulary:
        probability = estimate_probability(word, previous_n_gram,
                                           n_gram_counts, n_plus1_gram_counts,
                                           vocabulary_size, k=k)
        probabilities[word] = probability
    return probabilities
sentences = [['i', 'like', 'a', 'cat'],
             ['this', 'dog', 'is', 'like', 'a', 'cat']]
unique_words = list(set(sentences[0] + sentences[1]))
unigram_counts = count_n_grams(sentences, 1)
bigram_counts = count_n_grams(sentences, 2)
print(estimate_probabilities("a", unigram_counts, bigram_counts, unique_words, k=1))
def make_count_matrix(n_plus1_gram_counts,vocabulary):
    vocabulary = vocabulary + ["<e>","<unk>"]
    n_grams = []
    for n_plus1_gram in n_plus1_gram_counts.keys():
        n_gram = n_plus1_gram[0:-1]
        n_grams.append(n_gram)
    n_grams = list(set(n_grams))
    row_index = {n_gram: i for i, n_gram in enumerate(n_grams)}
    col_index = {word: j for j, word in enumerate(vocabulary)}
    nrow = len(n_grams)
    ncol = len(vocabulary)
    count_matrix = np.zeros((nrow,ncol))
    for n_plus1_gram,count in n_plus1_gram_counts.items():
        n_gram = n_plus1_gram[0:-1]
        word = n_plus1_gram[-1]
        if word not in vocabulary:
            continue
        i = row_index[n_gram]
        j = col_index[word]
        count_matrix[i,j] = count
    count_matrix = pd.DataFrame(count_matrix,index=n_grams, columns=vocabulary)
    return count_matrix
sentences = [['i', 'like', 'a', 'cat'],
                 ['this', 'dog', 'is', 'like', 'a', 'cat']]
unique_words = list(set(sentences[0] + sentences[1]))
bigram_counts = count_n_grams(sentences, 2)

print('bigram counts')
print(make_count_matrix(bigram_counts, unique_words))

def make_probability_matrix(n_plus1_gram_counts, vocabulary, k):
    count_matrix = make_count_matrix(n_plus1_gram_counts, unique_words)
    count_matrix += k
    prob_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)
    return prob_matrix
print("bigram probabilities")
print(make_probability_matrix(bigram_counts, unique_words, k=1))


print("trigram probabilities")
trigram_counts = count_n_grams(sentences, 3)
print(make_probability_matrix(trigram_counts, unique_words, k=1))

###part3
#PP(W)困惑度
def calculate_perplexity(sentence,n_gram_counts,n_plus1_gram_counts,vocabulary_size,k=1.0):
    n = len(list(n_gram_counts.keys())[0])
    sentence = ["<s>"] * n + sentence + ["<e>"]
    sentence = tuple(sentence)
    N = len(sentence)
    product_pi = 1.0
    for t in range(n,N):
        n_gram = sentence[t-n:t]
        word = sentence[t]
        probability = estimate_probability(word, n_gram,
                         n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0)
        product_pi *= pow(probability, -1)
    perplexity = pow(product_pi, 1 / N)
    return perplexity

sentences = [['i', 'like', 'a', 'cat'],
                 ['this', 'dog', 'is', 'like', 'a', 'cat']]
unique_words = list(set(sentences[0] + sentences[1]))

unigram_counts = count_n_grams(sentences, 1)
bigram_counts = count_n_grams(sentences, 2)


perplexity_train1 = calculate_perplexity(sentences[0],
                                         unigram_counts, bigram_counts,
                                         len(unique_words), k=1.0)
print(f"Perplexity for first train sample: {perplexity_train1:.4f}")

test_sentence = ['i', 'like', 'a', 'dog']
perplexity_test = calculate_perplexity(test_sentence,
                                       unigram_counts, bigram_counts,
                                       len(unique_words), k=1.0)
print(f"Perplexity for test sample: {perplexity_test:.4f}")



###part4 Build an auto-complete system
def suggest_a_word(previous_tokens,n_gram_counts,n_plus1_gram_counts,vocabulary,k=1.0, start_with=None):
    n = len(list(n_gram_counts.keys())[0])
    previous_n_gram = previous_tokens[-n:]
    probabilities = estimate_probabilities(previous_n_gram,
                                           n_gram_counts, n_plus1_gram_counts,
                                           vocabulary, k=k)
    suggestion = None
    max_prob = 0
    for word, prob in probabilities.items():
        if start_with:
            if not word.startswith(start_with):
                continue
        if prob > max_prob:
            suggestion = word
            max_prob = prob
    return suggestion, max_prob

sentences = [['i', 'like', 'a', 'cat'],
             ['this', 'dog', 'is', 'like', 'a', 'cat']]
unique_words = list(set(sentences[0] + sentences[1]))

unigram_counts = count_n_grams(sentences, 1)
bigram_counts = count_n_grams(sentences, 2)
previous_tokens = ["i", "like"]
tmp_suggest1 = suggest_a_word(previous_tokens, unigram_counts, bigram_counts, unique_words, k=1.0)
print(f"The previous words are 'i like',\n\tand the suggested word is `{tmp_suggest1[0]}` with a probability of {tmp_suggest1[1]:.4f}")


def get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0, start_with=None):
    model_counts = len(n_gram_counts_list)
    suggestions = []
    for i in range(model_counts - 1):
        n_gram_counts = n_gram_counts_list[i]
        n_plus1_gram_counts = n_gram_counts_list[i + 1]

        suggestion = suggest_a_word(previous_tokens, n_gram_counts,
                                    n_plus1_gram_counts, vocabulary,
                                    k=k, start_with=start_with)
        suggestions.append(suggestion)
    return suggestions

sentences = [['i', 'like', 'a', 'cat'],
             ['this', 'dog', 'is', 'like', 'a', 'cat']]
unique_words = list(set(sentences[0] + sentences[1]))

unigram_counts = count_n_grams(sentences, 1)
bigram_counts = count_n_grams(sentences, 2)
trigram_counts = count_n_grams(sentences, 3)
quadgram_counts = count_n_grams(sentences, 4)
qintgram_counts = count_n_grams(sentences, 5)

n_gram_counts_list = [unigram_counts, bigram_counts, trigram_counts, quadgram_counts, qintgram_counts]
previous_tokens = ["i", "like"]
tmp_suggest3 = get_suggestions(previous_tokens, n_gram_counts_list, unique_words, k=1.0)

print(f"The previous words are 'i like', the suggestions are:")
print(tmp_suggest3)



previous_tokens = ["i", "am", "to"]
tmp_suggest4 = get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0)

print(f"The previous words are {previous_tokens}, the suggestions are:")
print(tmp_suggest4)