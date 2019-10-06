# regex for removing punctuation!
import re
# nltk preprocessing magic
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# grabbing a part of speech function:
from part_of_speech import get_part_of_speech

text = "So many squids are jumping out of suitcases these days that you can barely go anywhere without seeing one burst forth from a tightly packed valise. I went to the dentist the other day, and sure enough I saw an angry one jump out of my dentist's bag within minutes of arriving. She hardly even noticed."

cleaned = re.sub('\W+', ' ', text)
#break sentence into words
tokenized = word_tokenize(cleaned)

#found stem
stemmer = PorterStemmer()
stemmed = [stemmer.stem(token) for token in tokenized]

#found root
lemmatizer = WordNetLemmatizer()
#lemmatize() recognize every words as noun in defalut, get_part_of_speech() tell lemmatize which part of the word in the sentence is.
lemmatized = [lemmatizer.lemmatize(token, get_part_of_speech(token)) for token in tokenized]

print("Stemmed text:")
print(stemmed)
print("\nLemmatized text:")
print(lemmatized)