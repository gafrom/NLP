import collections
import tensorflow as tf
from nlp.cleaner import Cleaner
import numpy as np

class NBClassifier(object):
  VOCABULARY_SIZE = 50000

  def __init__(self):
    self.cleaner = Cleaner()

    self.dictionary = None

    self.prob_Ck            = None
    self.term_freq          = None
    self.term_freq_by_class = None

    self.result_class = collections.namedtuple('NaiveBayesClassifierResult', ('label', 'probs'))

  def train(self, labeled_documents):
    self.dictionary, self.reverse_dictionary, self.term_freq = \
      self.build_dictionary(labeled_documents)
    term_freq_by_class = []

    for _, documents in labeled_documents.groupby('label'):
      _, _, term_freq = self.build_dictionary(documents)
      term_freq_by_class.append(term_freq)

    self.term_freq_by_class = np.array(term_freq_by_class)
    self.prob_Ck = 1 / len(self.term_freq_by_class)

  def predict(self, text):
    assert self.prob_Ck, 'Cannot infer - model should be trained first.'

    tokens = self.tokenize(text)
    return self.infer_from(tokens)

# private

  def infer_from(self, tokens):
    prob_Ck_given_x = (self.prob_Ck / self.prob_x(tokens)) * self.prob_x_given_Ck(tokens)
    return self.result_class(
      int(round(np.amax(prob_Ck_given_x))), prob_Ck_given_x
    )

  def prob_x(self, x):
    freqs = np.vectorize(self.term_freq.get)(x) / self.VOCABULARY_SIZE
    return np.prod(freqs)

  def prob_x_given_Ck(self, x):
    fr_by_cl = []
    for term_freq in self.term_freq_by_class:
      freq = np.prod(np.vectorize(term_freq.get)(x) / self.VOCABULARY_SIZE)
      fr_by_cl.append(freq)

    return np.array(fr_by_cl)

  def tokenize(self, text):
    words = self.cleaner.words(text)

    return [self.dictionary[word] for word in words if word in self.dictionary]

  def build_dictionary(self, documents):
    words = []

    for document in documents['body']:
      words.extend(self.cleaner.words(document))

    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(self.VOCABULARY_SIZE - 1))
    dictionary = dict()
    for word, _ in count:
      dictionary[word] = len(dictionary)

    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    count_dict = {dictionary[key]:value for key, value in count}

    return dictionary, reverse_dictionary, count_dict
