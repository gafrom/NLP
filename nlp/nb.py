import collections
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
    self.build_dictionary(labeled_documents)
    term_freq_by_class = []

    for _, documents in labeled_documents.groupby('label'):
      count = self.count_words(self.extract_words(documents))
      term_freq_by_class.append(self.build_term_frequencies(count))

    self.term_freq_by_class = np.array(term_freq_by_class)
    self.prob_Ck = 1 / len(self.term_freq_by_class)

  def predict(self, text):
    assert self.prob_Ck, 'Cannot infer - model should be trained first.'

    tokens = self.tokenize(text)
    return self.infer_from(tokens)

# private

  def infer_from(self, tokens):
    prob_Ck_given_x = (self.prob_Ck / self.prob_x(tokens)) * self.prob_x_given_Ck(tokens)
    return self.result_class(np.argmax(prob_Ck_given_x), prob_Ck_given_x)

  def prob_x(self, x):
    freqs = np.vectorize(self.term_freq.get)(x) / self.VOCABULARY_SIZE
    return np.prod(freqs)

  def prob_x_given_Ck(self, x):
    # TODO: account for unknown words
    # known_tokens = [token for token in x if token in self.term]

    fr_by_cl = []
    for term_freq in self.term_freq_by_class:
      freq = np.prod(np.vectorize(term_freq.get)(x) / self.VOCABULARY_SIZE)
      fr_by_cl.append(freq)

    return np.array(fr_by_cl)

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
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(self.VOCABULARY_SIZE - 1))

    return count

  def build_term_frequencies(self, count):
    assert self.dictionary, 'Cannot count words - a dictionary should be build first.'
    return { self.dictionary[key]:value for key, value in count if key in self.dictionary }
