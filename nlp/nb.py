import collections
from nlp.cleaner import Cleaner
import numpy as np
import math
import functools as fn

class NBClassifier(object):
  VOCABULARY_SIZE = 50000
  CORRELATION_LIMIT = 0.1
  TF_STRATEGY    = 'tf'
  TFIDF_STRATEGY = 'tfidf'

  def __init__(self, strategy = TFIDF_STRATEGY, cleaner = Cleaner()):
    self.strategy = strategy
    self.cleaner  = cleaner
    self.dictionary  = None
    self.prob_Ck     = None
    self.tf          = None
    self.tf_by_class = None
    self.idf         = None
    self.result_class = collections.namedtuple('NaiveBayesClassifierResult', ('label', 'probs'))

  def train(self, labeled_documents):
    self.build_dictionary(labeled_documents)
    tf_by_class = []

    for _, documents in labeled_documents.groupby('label'):
      count = self.count_words(self.extract_words(documents))
      tf = self.build_term_frequencies(count)
      self.prepare_for_laplace_smoothing(tf)
      tf_by_class.append(tf)

    self.tf_by_class = np.array(tf_by_class)
    self.prob_Ck = 1 / len(self.tf_by_class)

    self.remove_highly_correlated_features()
    self.remove_stop_words()

    if self.strategy == self.TFIDF_STRATEGY: self.compute_idf(labeled_documents)
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
      prob_denominator = self.prob_denominators[self.strategy][class_index]
      probs = prob_numerators / prob_denominator

      log_prob = np.sum(np.log(probs))
      log_likelihoods.append(log_prob)

    return np.array(log_likelihoods)

  def prepare_for_laplace_smoothing(self, term_freq):
    for token in self.tf.keys():
      if token not in term_freq: term_freq[token] = 0

  # Laplace smoothing
  def smoothed(self, tf):
    # We believe, that prior is made up in such a way, as if we had observed every feature
    # at least once. And now we are updating the prior with seen data.
    if self.strategy == self.TFIDF_STRATEGY:
      return lambda token: tf[token] * self.idf[token] + 1
    else:
      return lambda token: tf[token] + 1

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

  def build_dictionary(self, documents):
    words = self.extract_words(documents)
    count = self.count_words(words)
    self.dictionary = dict()
    for word, _ in count: self.dictionary[word] = len(self.dictionary)
    self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))
    self.tf = self.build_term_frequencies(count)

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

  def remove_highly_correlated_features(self):
    to_be_removed = []

    for token in self.tf.keys():
      tfs = [tf[token] for tf in self.tf_by_class]

      # TODO: extrapolate for multinomial classifier
      if abs(tfs[0] - tfs[1]) / self.tf[token] < self.CORRELATION_LIMIT:
        to_be_removed.append((token, tfs))

    for token, tfs in to_be_removed:
      word = self.reverse_dictionary[token]
      print(f"Removing word {word} for {tfs}")
      self.dictionary.pop(word, None)
      self.reverse_dictionary.pop(token, None)
      self.tf.pop(token, None)
      for tf in self.tf_by_class:
        tf.pop(token, None)


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

  def compute_idf(self, labeled_documents):
    df = collections.defaultdict(int)

    tokenized_documents = []
    for document in labeled_documents['body']:
      doc = { self.dictionary[word] for word in self.cleaner.words(document) if word in self.dictionary }
      tokenized_documents.append(doc)

    print(f"Tokenized {len(tokenized_documents)} documents")
    for key in self.reverse_dictionary.keys():
      for document in tokenized_documents:
        if key in document: df[key] += 1

    self.idf = { key: math.log(len(labeled_documents) / df[key]) for key in df.keys() }
    print('Built IDF')

  def compute_prob_denominators(self):
    # + 1 and len(tf) are needed to account for Laplace smoothing
    tfidf_den = []

    if self.strategy == self.TFIDF_STRATEGY:
      for tf in self.tf_by_class:
        den = fn.reduce((lambda total, key: total + (tf[key] + 1) * self.idf[key]), tf.keys())
        tfidf_den.append(den)

    self.prob_denominators = {
      'tf'    : [(sum(tf.values()) + len(tf)) for tf in self.tf_by_class],
      'tfidf' : tfidf_den
    }
