#spaCy is a free open-source library for Natural Language Processing in Python. It features NER, POS tagging, dependency parsing, word vectors and more.
#https://blog.csdn.net/u012436149/article/details/79321112
import spacy
from nltk import Tree
#from squids import squids_text

dependency_parser = spacy.load('en')

#now parsed_squids have become a memeber of spacy.english model, it has many attributes, including tokens, token’s reference index, part of speech tags, entities, vectors, sentiment, vocabulary etc
#parsed_squids = dependency_parser(squids_text)

# Assign my_sentence a new value:
my_sentence = "Your sentence goes here!"
my_parsed_sentence = dependency_parser(my_sentence)

def to_nltk_tree(node):
  if node.n_lefts + node.n_rights > 0:
    parsed_child_nodes = [to_nltk_tree(child) for child in node.children]
    return Tree(node.orth_, parsed_child_nodes)
  else:
    return node.orth_

#.sents means sentences. for sent in parsed_squids.sents means traverse all the senteces in squids_text
for sent in parsed_squids.sents:
  to_nltk_tree(sent.root).pretty_print()
  
for sent in my_parsed_sentence.sents:
 to_nltk_tree(sent.root).pretty_print()