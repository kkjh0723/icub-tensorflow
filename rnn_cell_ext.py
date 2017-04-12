
import tensorflow as tf

class BasicCTRNNCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, num_units, time_const , input_size=None, state_is_tuple=False, activation=tf.tanh):

        '''
        :param num_units:
        :param eta: 1/time_constant, should be set value between 0.0 and 1.0
        :param input_size:
        :param state_is_tuple:
        :param activation:
        '''


        self._num_units = num_units
        self._state_is_tuple = state_is_tuple
        self._activation = activation
        self._eta = 1.0/(time_const + 1e-8)

    @property
    def state_size(self):
        return 2 * self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):

        with tf.variable_scope(scope or type(self).__name__):  # "BasicCTRNNCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(1, 2, state)

            concat = tf.nn.rnn_cell._linear([inputs, h], self._num_units, True)
            new_c = (c * (self._eta) + (1-self._eta)* concat)
            new_h = self._activation(new_c)

            if self._state_is_tuple:
                new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
                #new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat(1, [new_c, new_h])
            return new_h, new_state


class AdaptiveCTRNNCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units, input_size=None, state_is_tuple=False, activation=tf.tanh):

        '''
        :param num_units:
        :param input_size:
        :param state_is_tuple:
        :param activation:
        '''


        if state_is_tuple == True:
            raise ValueError("\"state_is_tuple\" is not yet implemented")

        self._num_units = num_units
        self._state_is_tuple = state_is_tuple
        self._activation = activation
        self._eta_state = tf.Variable(tf.random_uniform([self._num_units],0.0,1.0))

    @property
    def state_size(self):
        return 2 * self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):

        with tf.variable_scope(scope or type(self).__name__):  # "AdaptiveCTRNNCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(1, 2, state)

            concat = tf.nn.rnn_cell._linear([inputs, h], self._num_units, True)
            eta = tf.sigmoid(self._eta_state)
            new_c = tf.mul(c , eta) + tf.mul((1 - eta), concat)
            new_h = self._activation(new_c)

            if self._state_is_tuple:
                new_state = (new_c, new_h)
                # new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat(1, [new_c, new_h])

            return new_h, new_state
