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
    


