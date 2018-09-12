import math
from nlp.cleaner import Cleaner
from collections import defaultdict

class TFIDFer(object):
  def __init__(self, dictionary, reverse_dictionary, cleaner = Cleaner()):
    self.dictionary = dictionary
    self.reverse_dictionary = reverse_dictionary
    self.cleaner = cleaner

  # here we use advanced definition of Term Frequence:
  # raw count of a term in a document adjusted for document length
  def compute_tf(self, documents_as_words):
    tf = defaultdict(int)

    for words in documents_as_words:
      for word in (x for x in words if x in self.dictionary):
        tf[self.dictionary[word]] += 1 / len(words)

    return tf

  def compute_idf(self, documents):
    df = defaultdict(int)

    tokenized_documents = self.tokenize(documents)
    print(f"Tokenized {len(tokenized_documents)} documents")

    for key in self.reverse_dictionary.keys():
      for document in tokenized_documents:
        if key in document: df[key] += 1

    idf = { key: math.log(len(documents) / df[key]) for key in df.keys() }
    print('Built IDF')

    return idf

  def tokenize(self, documents):
    tokenized_documents = []
    for document in documents:
      doc = { self.dictionary[word] for word in self.cleaner.words(document) if word in self.dictionary }
      tokenized_documents.append(doc)

    return tokenized_documents
