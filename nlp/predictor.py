import tensorflow as tf
from pathlib import Path
import pickle
from nlp.cleaner import Cleaner

class Model(object):
  def __init__(self, embedding_size, num_nodes, vocabulary_size, num_classes):
    self.graph = tf.Graph()
    with self.graph.as_default():
      # Parameters:
      # Input gate: input, previous output, and bias.
      ix = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1), name='ix')
      im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1), name='im')
      ib = tf.Variable(tf.zeros([1, num_nodes]), name='ib')
      # Forget gate: input, previous output, and bias.
      fx = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1), name='fx')
      fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1), name='fm')
      fb = tf.Variable(tf.zeros([1, num_nodes]), name='fb')
      # Output gate: input, previous output, and bias.
      ox = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1), name='ox')
      om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1), name='om')
      ob = tf.Variable(tf.zeros([1, num_nodes]), name='ob')
      # Memory cell: input, state and bias.                             
      cx = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1), name='cx')
      cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1), name='cm')
      cb = tf.Variable(tf.zeros([1, num_nodes]), name='cb')
      # Variables saving state across unrollings.
      saved_output = tf.Variable(tf.zeros([1, num_nodes]), trainable=False)
      saved_state = tf.Variable(tf.zeros([1, num_nodes]), trainable=False)
      # Classifier weights and biases.
      w = tf.Variable(tf.truncated_normal([num_nodes, num_classes], -0.1, 0.1), name='w')
      b = tf.Variable(tf.zeros([num_classes]), name='b')
      # Embeddings
      embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name='embeddings')

      # Definition of the cell computation.
      def lstm_cell(i, o, state):
        # Create an LSTM cell
        input_gate =  tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
        forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
        output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
        update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
        state = forget_gate * state + input_gate * tf.tanh(update)

        return output_gate * tf.tanh(state), state

      # Sampling and validation eval: batch 1, no unrolling.
      self.sample_input = tf.placeholder(tf.int32, shape=[1])
      sample_lstm_input = tf.nn.embedding_lookup(embeddings, self.sample_input)
      saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
      saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
      self.reset_sample_state = tf.group(
        saved_sample_output.assign(tf.zeros([1, num_nodes])),
        saved_sample_state.assign(tf.zeros([1, num_nodes])))
      sample_output, sample_state = lstm_cell(
        sample_lstm_input, saved_sample_output, saved_sample_state)
      with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                    saved_sample_state.assign(sample_state)]):
        self.sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))

      variables_to_restore = [ix, im, ib, fx, fm, fb, ox, om, ob, cx, cm, cb, w, b, embeddings]
      self.saver = tf.train.Saver(variables_to_restore)


class Dictionary(object):
  def __init__(self, path):
    self.path = path

  def load(self):
    dictionary_file_name = f"{self.path}.dictionary"
    if Path(dictionary_file_name).is_file():
      with open(dictionary_file_name, 'rb') as fp:
        return pickle.load(fp)
    else:
      msg = f"Cannot find stored dictionary file: {dictionary_file_name}"
      raise FileNotFoundError(msg)

class Predictor(object):
  STORAGE_PATH = './storage/model'

  def __init__(self):
    print('Loading neural network Graph...')
    self.model = Model(128, 64, 25000, 2)
    self.saver              = self.model.saver
    self.graph              = self.model.graph
    self.sample_input       = self.model.sample_input
    self.reset_sample_state = self.model.reset_sample_state
    self.sample_prediction  = self.model.sample_prediction

    self.session = tf.Session(graph=self.graph)
    self.saver.restore(self.session, self.STORAGE_PATH)
    print('Done')
    print('Loading dictionary...')
    self.dictionary, self.reverse_dictionary, _ = Dictionary(self.STORAGE_PATH).load()
    print('Done')
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
