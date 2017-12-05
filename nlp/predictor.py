import tensorflow as tf
from nlp.dictionary import Dictionary
from nlp.cleaner import Cleaner

class Predictor(object):
  STORAGE_PATH = './storage/model'

  def __init__(self):
    print('🚜 Loading neural network Graph... ', end='')
    self.session = tf.Session()
    saver = tf.train.import_meta_graph(f"{self.STORAGE_PATH}.meta")
    print('Done ✅')

    print('🚜 Loading neural network brains...', end='')
    saver.restore(self.session, self.STORAGE_PATH)
    self.graph = tf.get_default_graph()
    self.sample_input       = self.graph.get_tensor_by_name('sample_input:0')
    self.reset_sample_state = self.graph.get_operation_by_name('reset_sample_state')
    self.sample_prediction  = self.graph.get_tensor_by_name('sample_prediction:0')
    print('Done ✅')

    print('🚜 Loading dictionary...', end='')
    self.dictionary, self.reverse_dictionary, _ = Dictionary(self.STORAGE_PATH).load()
    print('Done ✅')
    self.cleaner = Cleaner()

  def classify(self, text):
    words = self.cleaner.words(text)

    self.session.run(self.reset_sample_state)
    predicted = []
    for word in words:
      prediction = self.session.run(self.sample_prediction, { self.sample_input: [self.dictionary[word]] })
      predicted.append(prediction[0])

    final = predicted[-1][0]
    if final > 0.6:   klass = 'Good'
    elif final > 0.4: klass = 'Neutral'
    else:             klass = 'Bad'

    msg = f"{text}\n{words}\n{predicted}"
    return f"Your article is {klass}:\n{msg}"
