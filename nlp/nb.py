import collections
from nlp.cleaner import Cleaner
import numpy as np
import math

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
    self.build_dictionary(labeled_documents)
    term_freq_by_class = []

    for _, documents in labeled_documents.groupby('label'):
      count = self.count_words(self.extract_words(documents))
      term_freq = self.build_term_frequencies(count)
      self.prepare_for_laplace_smoothing(term_freq)
      term_freq_by_class.append(term_freq)

    self.term_freq_by_class = np.array(term_freq_by_class)
    self.prob_Ck = 1 / len(self.term_freq_by_class)

  def predict(self, text):
    assert self.prob_Ck, 'Cannot infer - model should be trained first.'

    tokens = self.tokenize(text)
    return self.infer_from(tokens)

# private --------------------------------------------------------------------------

  def infer_from(self, tokens):
    prob_Ck_given_x = self.prob_Ck * self.likelihood(tokens) / self.evidence(tokens)
    return self.result_class(np.argmax(prob_Ck_given_x), prob_Ck_given_x)

  # prob_x_given_Ck
  def likelihood(self, tokens):
    likelihoods = []
    for term_freq in self.term_freq_by_class:
      probs = np.vectorize(self.smoothed(term_freq))(tokens) / self.VOCABULARY_SIZE
      prob = math.exp(np.sum(np.log(probs)))
      likelihoods.append(prob)

    return np.array(likelihoods)

  def prepare_for_laplace_smoothing(self, term_freq):
      for token in self.term_freq.keys():
        if token not in term_freq: term_freq[token] = 0

  # Laplace smoothing
  def smoothed(self, term_freq):
    # We believe, that prior is made up in such a way, as if we had observed every feature
    # at least once. And now we are updating the prior with seen data.
    return lambda token: term_freq.get(token) + 1

  # prob_x
  def evidence(self, x):
    # We do not adjust prior by P(X) since it is the same for all classes.
    # You may say that it would affect the final propability we get, but
    # there is no concern here because probability in Bayes Naive is
    # essentially undermined, garbled and usually very close to 0 or 1.
    return 1

  def tokenize(self, text):
    words = self.cleaner.words(text)

    return [self.dictionary[word] for word in words if word in self.dictionary]

  def build_dictionary(self, documents):
    words = self.extract_words(documents)
    count = self.count_words(words)
    self.dictionary = dict()
    for word, _ in count: self.dictionary[word] = len(self.dictionary)
    self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))
    self.term_freq = self.build_term_frequencies(count)

  def extract_words(self, documents):
    words = []

    for document in documents['body']:
      words.extend(self.cleaner.words(document))

    return words

  def count_words(self, words):
    return collections.Counter(words).most_common(self.VOCABULARY_SIZE)

  def build_term_frequencies(self, count):
    assert self.dictionary, 'Cannot count words - a dictionary should be build first.'
    return { self.dictionary[key]:value for key, value in count if key in self.dictionary }
