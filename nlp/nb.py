import collections
from nlp.cleaner import Cleaner
from nlp.tfidfer import TFIDFer
import numpy as np
import math
import functools as fn

class NBClassifier(object):
  VOCABULARY_SIZE = 50000
  CORRELATION_LIMIT = 0.03
  TF_STRATEGY    = 'tf'
  TFIDF_STRATEGY = 'tfidf'

  def __init__(self, strategy = TFIDF_STRATEGY, cleaner = Cleaner(ngrams_size = 2),
                     tfidfer_class = TFIDFer):
    self.strategy = strategy
    self.cleaner  = cleaner
    self.tfidfer_class = tfidfer_class
    self.tfidefer    = None
    self.dictionary  = None
    self.prob_Ck     = None
    self.tf          = None
    self.tf_by_class = None
    self.idf         = None
    self.once        = None
    self.result_class = collections.namedtuple('NaiveBayesClassifierResult', ('label', 'probs'))

  def train(self, labeled_documents):
    documents_as_words = self.documents_as_words(labeled_documents)
    self.build_dictionary(documents_as_words)
    self.tfidfer = self.tfidfer_class(self.dictionary, self.reverse_dictionary, self.cleaner)
    self.tf = self.tfidfer.compute_tf(documents_as_words)
    self.once = 1 / (sum(len(x) for x in documents_as_words) / len(labeled_documents))

    tf_by_class = []

    for _, documents in labeled_documents.groupby('label'):
      tf = self.tfidfer.compute_tf(self.documents_as_words(documents))
      tf_by_class.append(tf)

    self.tf_by_class = np.array(tf_by_class)
    self.prob_Ck = 1 / len(self.tf_by_class)

    self.remove_highly_correlated_features()
    self.remove_stop_words()

    if self.strategy == self.TFIDF_STRATEGY:
      self.idf = self.tfidfer.compute_idf(labeled_documents['body'])
      self.remove_zero_idf_words()
    self.compute_prob_denominators()

  def predict(self, text):
    assert self.prob_Ck, 'Cannot infer - model should be trained first.'

    tokens = self.tokenize(text)
    return self.infer_from(tokens)

# private --------------------------------------------------------------------------

  def infer_from(self, tokens):
    log_prob_Ck_given_x = \
      math.log(self.prob_Ck) + self.log_likelihood(tokens) - math.log(self.evidence(tokens))

    return self.result_class(np.argmax(log_prob_Ck_given_x), log_prob_Ck_given_x)

  # prob_x_given_Ck
  def log_likelihood(self, tokens):
    log_likelihoods = []
    for class_index, tf in enumerate(self.tf_by_class):
      prob_numerators  = np.vectorize(self.smoothed(tf))(tokens) if tokens else np.empty(0)
      if np.any(prob_numerators == 0): import pdb; pdb.set_trace()
      prob_denominator = self.prob_denominators[self.strategy][class_index]
      probs = prob_numerators / prob_denominator

      log_prob = np.sum(np.log(probs))
      log_likelihoods.append(log_prob)

    return np.array(log_likelihoods)

  # Laplace smoothing
  def smoothed(self, tf):
    # We believe, that prior is made up in such a way, as if we had observed every feature
    # at least once. And now we are updating the prior with seen data.

    if self.strategy == self.TFIDF_STRATEGY:
      return lambda token: (tf[token] + self.once) * self.idf[token]
    else:
      return lambda token:  tf[token] + self.once

  # prob_x
  def evidence(self, x):
    # We do not adjust prior by P(X) since it is the same for all classes.
    # You may say that it would affect the final probability we get, but
    # there is no concern here because probability in Bayes Naive is
    # essentially undermined, garbled and usually very close to 0 or 1.
    return 1

  def tokenize(self, text):
    words = self.cleaner.words(text)

    return [self.dictionary[word] for word in words if word in self.dictionary]

  def build_dictionary(self, documents_as_words):
    words = [word for words in documents_as_words for word in words]
    count = collections.Counter(words).most_common(self.VOCABULARY_SIZE)
    self.dictionary = dict()
    for word, _ in count: self.dictionary[word] = len(self.dictionary)
    self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))

  def documents_as_words(self, documents):
    return [self.cleaner.words(document) for document in documents['body']]

  def remove_highly_correlated_features(self):
    to_be_removed = []

    for token in self.tf.keys():
      tfs = [tf[token] for tf in self.tf_by_class]

      # TODO: extrapolate for multinomial classifier
      if abs(tfs[0] - tfs[1]) / self.tf[token] < self.CORRELATION_LIMIT:
        to_be_removed.append((token, tfs))

    msg = []

    for token, tfs in to_be_removed:
      word = self.reverse_dictionary[token]
      msg.append(word)
      self.dictionary.pop(word, None)
      self.reverse_dictionary.pop(token, None)
      self.tf.pop(token, None)
      for tf in self.tf_by_class:
        tf.pop(token, None)

    print(f"Removed highly correlated words: {', '.join(msg)}.")

  def remove_stop_words(self):
    to_be_removed = [token for token in self.tf.keys() if self.tf[token] > 5000]

    for token in to_be_removed:
      word = self.reverse_dictionary[token]
      print(f"Removing stop word {word}")
      self.dictionary.pop(word, None)
      self.reverse_dictionary.pop(token, None)
      self.tf.pop(token, None)
      for tf in self.tf_by_class:
        tf.pop(token, None)

  def remove_zero_idf_words(self):
    to_be_removed = [token for token in self.idf if self.idf[token] == 0]
    sources_with_tokens = \
      [self.reverse_dictionary, self.tf, self.idf] + [tf for tf in self.tf_by_class]

    for token in to_be_removed:
      word = self.reverse_dictionary[token]
      print(f"Removing word {word} having zero idf")
      self.dictionary.pop(word, None)
      for source in sources_with_tokens:
        source.pop(token, None)

  def compute_prob_denominators(self):
    # `self.once` are needed to account for Laplace smoothing
    tfidf_den = []

    if self.strategy == self.TFIDF_STRATEGY:
      for tf in self.tf_by_class:
        den = fn.reduce((lambda total, key: total + (tf[key] + self.once) * self.idf[key]), tf.keys())
        tfidf_den.append(den)

    self.prob_denominators = {
      'tf'    : [(sum(tf.values()) + len(tf) * self.once) for tf in self.tf_by_class],
      'tfidf' : tfidf_den
    }
