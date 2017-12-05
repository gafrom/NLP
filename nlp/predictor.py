import tensorflow as tf
import numpy as np # just to print out predictions

from nlp.cleaner import Cleaner
from nlp.dictionary import Dictionary

class Predictor(object):
  STORAGE_PATH = 'storage/model'

  def __init__(self):
    self.session = tf.Session()
    self._import_graph()
    self.sample_input, self.reset_sample_state, self.sample_prediction = self._load_brains()
    self._load_dictionary()
    self.cleaner = Cleaner()

    print('Predictor was successfully initialized ✅')

  def classify(self, text):
    words = self.cleaner.words(text)
    if len(words) == 0: return None

    self._replace_unique(words)

    self.session.run(self.reset_sample_state)
    predicted = ''
    for word in words:
      prediction = self.session.run(self.sample_prediction, { self.sample_input: [self.dictionary[word]] })
      final = prediction[0][0]
      predicted += np.array2string(np.round(prediction[0], 3), separator=',', precision=3)
      predicted += ','

    if final > 0.6:   klass = 'Good'
    elif final > 0.4: klass = 'Neutral'
    else:             klass = 'Bad'

    msg = f"{text}\n{words}\n{predicted[:-1]}"
    return f"Your article is {klass}:\n{msg}"

  def _import_graph(self):
    print('Loading neural network Graph...', end='')
    self.saver = tf.train.import_meta_graph(f"{self.STORAGE_PATH}.meta")
    print('Done ✅')

  def _load_brains(self):
    print('Loading neural network brains...', end='')
    self.saver.restore(self.session, self.STORAGE_PATH)
    graph = tf.get_default_graph()
    brains = (graph.get_tensor_by_name('sample_input:0'),
              graph.get_operation_by_name('reset_sample_state'),
              graph.get_tensor_by_name('sample_prediction:0'))
    print('Done ✅')

    return brains

  def _load_dictionary(self):
    print('Loading dictionary...', end='')
    self.dictionary, self.reverse_dictionary, _ = Dictionary(self.STORAGE_PATH).load()
    print('Done ✅')

  def _replace_unique(self, words):
    print('Loading dictionary...', end='')
    for i, word in enumerate(words):
      if word not in self.dictionary: words[i] = Dictionary.UNK
    print('Done ✅')
