import tensorflow as tf


class Model(object):
  def __init__(self, embedding_size, num_nodes, vocabulary_size, num_classes):
    self.graph = tf.Graph()
    with self.graph.as_default():
      # Parameters:
      # Input gate: input, previous output, and bias.
      ix = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
      im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
      ib = tf.Variable(tf.zeros([1, num_nodes]))
      # Forget gate: input, previous output, and bias.
      fx = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
      fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
      fb = tf.Variable(tf.zeros([1, num_nodes]))
      # Output gate: input, previous output, and bias.
      ox = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
      om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
      ob = tf.Variable(tf.zeros([1, num_nodes]))
      # Memory cell: input, state and bias.                             
      cx = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
      cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
      cb = tf.Variable(tf.zeros([1, num_nodes]))
      # Variables saving state across unrollings.
      saved_output = tf.Variable(tf.zeros([1, num_nodes]), trainable=False)
      saved_state = tf.Variable(tf.zeros([1, num_nodes]), trainable=False)
      # Classifier weights and biases.
      w = tf.Variable(tf.truncated_normal([num_nodes, num_classes], -0.1, 0.1))
      b = tf.Variable(tf.zeros([num_classes]))
      # Embeddings
      embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))


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
      sample_input = tf.placeholder(tf.int32, shape=[1])
      sample_lstm_input = tf.nn.embedding_lookup(embeddings, sample_input)
      saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
      saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
      reset_sample_state = tf.group(
        saved_sample_output.assign(tf.zeros([1, num_nodes])),
        saved_sample_state.assign(tf.zeros([1, num_nodes])))
      sample_output, sample_state = lstm_cell(
        sample_lstm_input, saved_sample_output, saved_sample_state)
      with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                    saved_sample_state.assign(sample_state)]):
        sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))

      print("*****************************************************************************")
      print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
      print("*****************************************************************************")
      self.saver = tf.train.Saver()



class Predictor(object):
  STORAGE_PATH = './storage/model'

  def __init__(self):
    print('Loading neural network Graph...')
    self.model = Model(128, 64, 25000, 2)
    self.saver = self.model.saver
    self.graph = self.model.graph
    self.session = tf.Session(graph=self.graph)
    self.saver.restore(self.session, self.STORAGE_PATH)
    print('Done')

  def classify(self, text):
    return f"{self.prediction} => {text}"
