import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

class RNN:
    def __init__(
        self,
        input_dim,				# number of input's dimensions
        max_seq_length,			# limit of the number of timesteps
        output_n_vocab,			# number of output's vocabulary size
        restore_prefix=None,	#
        keep_prob=None,			# dropout
        input_embedding=None,	# the dim of input embedding
        num_hidden_layers=1,	# number of hidden layers
        num_hidden_units=256,	# number of hidden units in each layer
        forget_bias=0.5,		# forget bias of the LSTM cell
        learning_rate=0.01		# learning rate
        ):

        assert(keep_prob is None or 0 <= keep_prob <= 1)
        assert(0 <= forget_bias <= 1)

        self._input_dim = input_dim
        self._max_seq_length = max_seq_length
        self._output_n_vocab = output_n_vocab
        self._keep_prob = keep_prob
        self._input_embedding = input_embedding
        self._num_hidden_layers = num_hidden_layers
        self._num_hidden_units = num_hidden_units
        self._forget_bias = forget_bias
        self._learning_rate = learning_rate
        self._restore_prefix = restore_prefix

    def __str__(self):
        # output detail information about this model
        ret = 'Model details:\n'
        ret += '\tinput_dim: {}\n'.format(self._input_dim)
        ret += '\tmax_seq_length: {}\n'.format(self._max_seq_length)
        ret += '\toutput_n_vocab: {}\n'.format(self._output_n_vocab)
        ret += '\tkeep_prob: {}\n'.format(self._keep_prob)
        ret += '\tinput_embedding: {}\n'.format(self._input_embedding)
        ret += '\tnum_hidden_layers: {}\n'.format(self._num_hidden_layers)
        ret += '\tnum_hidden_units: {}\n'.format(self._num_hidden_units)
        ret += '\tforget_bias: {}\n'.format(self._forget_bias)
        ret += '\tlearning_rate: {}\n'.format(self._learning_rate)
        return ret

    def _pad_line(self, line):
        line = np.array(line)
        seq_length = line.shape[0]
        assert(seq_length <= self._max_seq_length)

        if seq_length == self._max_seq_length:
        	return line
        else:
        	pad = np.array([np.zeros(self._input_dim) \
        		for q in range(self._max_seq_length - seq_length)])
        	return np.concatenate([line, pad], axis=0)

    def _pad_batch(self, batch):
        return np.array([self._pad_line(line) for line in batch])

    def _single_cell(self):
        # cell = rnn.BasicRNNCell(self._num_hidden_units) # Vanilla RNN
        cell = rnn.BasicLSTMCell(self._num_hidden_units, forget_bias=self._forget_bias)
        if self._keep_prob is not None:
        	cell = rnn.DropoutWrapper(cell, output_keep_prob=self._keep_prob)
        return cell

    def _RNNLayer(self, x):
        '''
        The input x should have the shape:
        	[batch_size, max_seq_length, input_dim]
        Use the following tf.unstack() function to convert it to as list
        Which contains max_seq_length tensors of shape:
        	[batch_size, input_dim]
        '''
        x = tf.unstack(x, self._max_seq_length, 1)
        if self._input_embedding != None: # should be deprecated
        	input_embed = tf.Variable(
        		tf.random_normal([self._input_dim, self._input_embedding]),
        		dtype=tf.float32
        		)
        	x = [tf.matmul(e, input_embed) for e in x]

        if self._num_hidden_layers == 1:
        	cell = self._single_cell()
        else:
        	cell = rnn.MultiRNNCell(
        		[self._single_cell() for _ in range(self._num_hidden_layers)])

        outputs, states = tf.contrib.rnn.static_rnn(
        	cell, 
        	x, 
        	dtype=tf.float32, 
        	sequence_length=self._batch_seq_lengths
        	)

        # Convert the outputs back to:
        # 	[max_seq_length, batch_size, num_hidden_units]
        outputs = tf.stack(outputs)
        # Then reorder to [batch_size, max_seq_length, num_hidden_units]
        outputs = tf.transpose(outputs, [1, 0, 2])
        batch_size = tf.shape(outputs)[0]
        index = tf.range(0, batch_size) * self._max_seq_length + \
        	(self._batch_seq_lengths - 1)
        outputs = tf.gather(
        	tf.reshape(outputs, [-1, self._num_hidden_units]), 
        	index
        	)

        output_w1 = tf.Variable(
        	tf.random_normal(
        		[self._num_hidden_units, self._output_n_vocab]), 
        	dtype=tf.float32
        	)
        b = tf.Variable(
        	tf.random_normal([self._output_n_vocab]), 
        	dtype=tf.float32
        	)
        return tf.matmul(outputs, output_w1) + b

    def create_model(self):
        # placeholders for input x & y
        self._batch_seq_lengths = tf.placeholder(tf.int32, shape=[None])
        self._x_pl = tf.placeholder(dtype=tf.float32, 
        	shape=[None, self._max_seq_length, self._input_dim])
        self._y_pl = tf.placeholder(dtype=tf.float32, 
        	shape=[None, self._output_n_vocab])

        # the result of LSTM
        self._logits = self._RNNLayer(self._x_pl)

        self._prediction = tf.nn.softmax(self._logits)
        self._loss = tf.reduce_mean(
        	tf.nn.softmax_cross_entropy_with_logits(
        		logits=self._logits, 
        		labels=self._y_pl
        		)
        	)

        self._optimizer = tf.train.GradientDescentOptimizer(
        	learning_rate=self._learning_rate)
        self._train_op = self._optimizer.minimize(self._loss)

        self._correct = tf.equal(
        	tf.argmax(self._prediction, axis=1), 
        	tf.argmax(self._y_pl, axis=1)
        	)
        self._accuracy = tf.reduce_mean(tf.cast(self._correct, tf.float32))

        self._sess = tf.Session()

        self._saver = tf.train.Saver()
        if self._restore_prefix is None:
        	self._sess.run(tf.global_variables_initializer())
        else:
        	self._saver.restore(self._sess, self._restore_prefix)

    def _make_input_feed(self, batch_x, batch_y):
        input_feed = {}
        input_feed[self._x_pl] = self._pad_batch(batch_x)
        input_feed[self._y_pl] = batch_y
        input_feed[self._batch_seq_lengths] = [len(e) for e in batch_x]
        return input_feed

    def train_step(self, batch_x, batch_y):
        self._sess.run(
        	self._train_op, 
        	feed_dict=self._make_input_feed(batch_x, batch_y)
        	)

    def get_accuracy(self, batch_x, batch_y):
        return self._sess.run(
        	self._accuracy, 
        	feed_dict=self._make_input_feed(batch_x, batch_y)
        	)

    def get_loss(self, batch_x, batch_y):
        return self._sess.run(
        	self._loss, 
        	feed_dict=self._make_input_feed(batch_x, batch_y)
        	)

    def get_prediction(self, batch_x, batch_y):
        return self._sess.run(
        	self._prediction, 
        	feed_dict=self._make_input_feed(batch_x, batch_y)
        	)

    def get_logits(self, batch_x, batch_y):
        return self._sess.run(
        	self._logits, 
        	feed_dict=self._make_input_feed(batch_x, batch_y)
        	)

    def save(self, file_prefix, nstep):
        self._saver.save(self._sess, file_prefix, global_step=nstep)

    def save_model(self, fpath):
        self._saver.save(self._sess, fpath)

