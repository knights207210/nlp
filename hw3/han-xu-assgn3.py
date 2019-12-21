#Part 1

from collections import Counter
from collections import defaultdict
import numpy as np
import pandas as pd

text = open("hobbit-train.txt").read()
words = text.split()
rawFreqs = Counter(words)


word2index = defaultdict(lambda: len(word2index))
UNK = word2index["<UNK>"]
[word2index[word] for word, freq in rawFreqs.items() if freq > 1]
word2index = defaultdict(lambda: UNK, word2index)
#word2index['Gandalf']

unigrams = [word2index[word] for word in words]
unigramFreqs = Counter(unigrams)
V_uni = len(unigramFreqs)
def unigramProb(unigram): 
    return (unigramFreqs[unigram]+1)/(sum(unigramFreqs.values())+V_uni)


bigrams = [ (word2index[words[i-1]], word2index[words[i]]) for i in range(1, len(words)) ]
bigramFreqs = Counter(bigrams)
def bigramProb(bigram):
    return (bigramFreqs[bigram]+1)/(unigramFreqs[bigram[0]]+V_uni)


trigrams = [ (word2index[words[i-2]],word2index[words[i-1]], word2index[words[i]]) for i in range(2, len(words)) ]
trigramsFreqs = Counter(trigrams)
def trigramProb(trigram): 
    return (trigramsFreqs[trigram]+1)/(bigramFreqs[(trigram[0],trigram[1])]+V_uni)


def trigramSentenceLogProb(words):
    uni = np.log(unigramProb(word2index[words[0]]))
    bi = np.log(bigramProb((word2index[words[0]],word2index[words[1]])))
    tri = np.sum([np.log(trigramProb((word2index[words[i-2]], word2index[words[i-1]], word2index[words[i]]))) for i in range(2, len(words))])
    if len(words)>=3:
        return  uni + bi + tri
    elif len(words) == 2:
        return uni + bi
    elif len(words) == 1:
        return uni

def getPerplexity(words):
    return np.exp(-trigramSentenceLogProb(words)/len(words))    


test = open("hw-test.txt").read()
sentences = test.split('\n')
perplexity = []
for sentence in sentences:
    if sentence:
        perplexity.append(getPerplexity(sentence.split()))

with open('han-xu-test.txt', 'w') as output:
    for p in perplexity:
        output.write(str(p)+"\n")


#Part 2 Option 3
# Build a character level 3-gram model
text = open("hobbit-train.txt").read()
chars = [ch for ch in text]
rawFreqs = Counter(chars)


char2index = defaultdict(lambda: len(char2index))
UNK = char2index["<UNK>"]
[char2index[char] for char, freq in rawFreqs.items() if freq > 1]
char2index = defaultdict(lambda: UNK, char2index)


unigrams = [char2index[char] for char in chars]
unigramFreqs = Counter(unigrams)
V_uni = len(unigramFreqs)
def unigramProb(unigram): 
    return (unigramFreqs[unigram]+1)/(sum(unigramFreqs.values())+V_uni)


bigrams = [ (char2index[chars[i-1]], char2index[chars[i]]) for i in range(1, len(chars)) ]
bigramFreqs = Counter(bigrams)
def bigramProb(bigram):
    return (bigramFreqs[bigram]+1)/(unigramFreqs[bigram[0]]+V_uni)


trigrams = [ (char2index[chars[i-2]],char2index[chars[i-1]], char2index[chars[i]]) for i in range(2, len(chars)) ]
trigramsFreqs = Counter(trigrams)
def trigramProb(trigram): 
    return (trigramsFreqs[trigram]+1)/(bigramFreqs[(trigram[0],trigram[1])]+V_uni)


def trigramSentenceLogProb(chars):
    uni = np.log(unigramProb(char2index[chars[0]]))
    bi = np.log(bigramProb((char2index[chars[0]],char2index[chars[1]])))
    tri = np.sum([np.log(trigramProb((char2index[chars[i-2]], char2index[chars[i-1]], char2index[chars[i]]))) for i in range(2, len(chars))])
    if len(chars)>=3:
        return  uni + bi + tri
    elif len(chars) == 2:
        return uni + bi
    elif len(chars) == 1:
        return uni

def getPerplexity(chars):
    return np.exp(-trigramSentenceLogProb(chars)/len(chars))    


test = open("hw-test.txt").read()
sentences = test.split('\n')
chars_test = [[ch for ch in words] for words in sentences]
perplexity = []
for char_test in chars_test:
    if char_test:
        perplexity.append(getPerplexity(char_test))

with open('han-xu-test-option3.txt', 'w') as output:
    for p in perplexity:
        output.write(str(p)+"\n")
    


