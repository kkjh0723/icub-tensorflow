
import tensorflow as tf
import functools
import rnn_cell_ext as rx
import BasicConvLSTMCell as bc

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper

def tanh_mod(x):
    return 1.7159 * tf.tanh(0.66666667 * x)

class Model(object):

    def __init__(self, vision, motor, c_vf, c_vm, c_vs, c_mf, c_ms, c_as, v_init, m_init, cl_ratio, learning_rate, opt, rnn_cell = None, state_is_tuple=True):

        # hyperparameters
        self._out_size_vrow = opt.out_size_vrow
        self._out_size_vcol = opt.out_size_vcol
        self._out_size_vision = self._out_size_vrow*self._out_size_vcol
        self._out_size_smdim = opt.out_size_smdim  # softmax dim. of motor
        self._out_size_mdim = opt.out_size_mdim  # motor dimension
        self._out_size_motor = self._out_size_smdim * self._out_size_mdim
        self._eps = 1e-8 #epsilon
        self._state_is_tuple = state_is_tuple  # state should be always tuple

        self._mo_unit = self._out_size_motor
        self._mf_unit = opt.mf_unit # motor fast
        self._ms_unit = opt.ms_unit # motor slow
        self._as_unit = opt.as_unit # associative (PFC)

        self._vs_unit = opt.vs_unit # vision slow
        self._vs_msize = opt.vs_msize
        self._vs_size = opt.vs_size
        self._vs_fsize = (opt.vs_fsize, opt.vs_fsize)  # filter size

        self._vm_unit = opt.vm_unit # vision middle
        self._vm_msize = opt.vm_msize
        self._vm_size = opt.vm_size
        self._vm_fsize = (opt.vm_fsize, opt.vm_fsize) #filter size

        self._vf_unit = opt.vf_unit  # vision fast
        self._vf_msize = opt.vf_msize
        self._vf_size = opt.vf_size
        self._vf_fsize = (opt.vf_fsize, opt.vf_fsize)  # filter size

        self._vo_fsize = (opt.vo_fsize, opt.vo_fsize)  # filter size

        # input/ouput Nodes
        self._v_in = vision[:,:-1,:,:]  # vision input [batch,time,vrow,vcol]
        self._v_out = vision[:, 1:, :, :]  # vision output target
        self._v_init = v_init  # initial vision input for closed loop generation (not used)

        self._m_in = motor[:,:-1,:,:] # motor input [batch, time, dim, smdim]
        self._m_out = motor[:, 1:, :, :] # motor output target
        self._m_init = m_init  # initial motor input for closed loop generation

        self._c_as = c_as
        self._c_vs = c_vs
        self._c_vm = c_vm
        self._c_vf = c_vf
        self._c_ms = c_ms
        self._c_mf = c_mf

        self._cl = cl_ratio # closed loop ratio
        self._lr = learning_rate # learning rate


        # Variables
        # vision fast
        with tf.variable_scope('vf'):
            self.cell_vf = bc.BasicConvCTNNCell(self._vf_size, self._vf_fsize, self._vf_unit, 1.0,
                                                state_is_tuple=self._state_is_tuple)
            # self.cell_vf = bc.BasicConvCTNNCell(self._vf_size, self._vf_fsize, self._vf_unit, 1.0,
            #                                 state_is_tuple=self._state_is_tuple)

        # vision mid
        with tf.variable_scope('vm'):
            self.cell_vm = bc.BasicConvCTNNCell(self._vm_size, self._vm_fsize, self._vm_unit,1.0,
                                                state_is_tuple=self._state_is_tuple)
            # self.cell_vm = bc.BasicConvCTNNCell(self._vm_size, self._vm_fsize, self._vm_unit, 4.0,
            #                                 state_is_tuple=self._state_is_tuple)

        #vision slow
        with tf.variable_scope('vs'):
            self.cell_vs = bc.BasicConvCTNNCell(self._vs_size, self._vs_fsize, self._vs_unit, 1.0, state_is_tuple=self._state_is_tuple)
            # self.cell_vs = bc.BasicConvCTNNCell(self._vs_size, self._vs_fsize, self._vs_unit, 16.0,
            #                                 state_is_tuple=self._state_is_tuple)

        # associative
        with tf.variable_scope('as'):
            self.cell_as = tf.nn.rnn_cell.BasicLSTMCell(self._as_unit, state_is_tuple=self._state_is_tuple)
            #self.cell_as = rx.BasicCTRNNCell(self._as_unit, 150.0, state_is_tuple=self._state_is_tuple)

        #motor slow
        with tf.variable_scope('ms'):
            self.cell_ms = tf.nn.rnn_cell.BasicLSTMCell(self._ms_unit, state_is_tuple=self._state_is_tuple)
            #self.cell_ms = rx.BasicCTRNNCell(self._ms_unit, 70.0, state_is_tuple=self._state_is_tuple)

        #motor fast
        with tf.variable_scope('mf'):
            self.cell_mf = tf.nn.rnn_cell.BasicLSTMCell(self._mf_unit, state_is_tuple=self._state_is_tuple)
            #self.cell_mf = rx.BasicCTRNNCell(self._mf_unit, 4.0, state_is_tuple=self._state_is_tuple)

        #motor output
        self.W_m_out = tf.get_variable('W_m_out', shape=[self._as_unit, self._out_size_motor])
        self.b_m_out = tf.get_variable('b_m_out', shape=[self._out_size_motor],
                                       initializer=tf.constant_initializer(0.0))
        #vision output
        self.W_v_out = tf.get_variable('W_v_out', shape=[self._vo_fsize[0], self._vo_fsize[1], self._vf_unit, 1])
        self.b_v_out = tf.get_variable('b_v_out', shape=[1],
                                     initializer=tf.constant_initializer(0.0))

        #graphs
        self.prediction
        self.optimize


    def model_step(self, input, model_out_prev):
        input_vision, input_motor = input #(vision, motor)
        prev_vision, prev_motor, prev_cell_vf, prev_cell_vm, prev_cell_vs, prev_cell_mf, prev_cell_ms , prev_cell_as = model_out_prev

        input_vision = tf.reshape(input_vision, [-1, self._out_size_vrow, self._out_size_vcol, 1])
        prev_vision = tf.reshape(prev_vision, [-1, self._out_size_vrow, self._out_size_vcol, 1])

        # seperate all cell states
        prev_out_mf, prev_state_mf = prev_cell_mf
        prev_out_ms, prev_state_ms = prev_cell_ms
        prev_out_as, prev_state_as = prev_cell_as
        prev_out_vf, prev_state_vf = prev_cell_vf
        prev_out_vm, prev_state_vm = prev_cell_vm
        prev_out_vs, prev_state_vs = prev_cell_vs

        # switching btw. open-loop and closed-loop
        cur_vision = tf.mul(prev_vision, self._cl) + tf.mul(input_vision, 1 - self._cl)
        #cur_motor = tf.mul(prev_motor, self._cl) + tf.mul(input_motor, 1 - self._cl)

        # vision fast (from vm and vision input)
        input_vf = cur_vision
        cell_vf = self.cell_vf(input_vf, prev_cell_vf, scope='vf')
        cell_out_vf, cell_state_vf = cell_vf

        # vision mid(from vs and vf)
        prev_out_vf_p = tf.nn.avg_pool(cell_out_vf, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')  # pooled
        input_vm = prev_out_vf_p
        cell_vm = self.cell_vm(input_vm, prev_cell_vm, scope='vm')
        cell_out_vm, cell_state_vm = cell_vm

        # vision slow (from as and vm)
        prev_out_vm_p = tf.nn.avg_pool(cell_out_vm, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')  # pooled
        input_vs = prev_out_vm_p
        cell_vs = self.cell_vs(input_vs, prev_cell_vs, scope='vs')
        cell_out_vs, cell_state_vs = cell_vs

        # Associative (from ms and vs)
        prev_out_vs_p = tf.nn.avg_pool(cell_out_vs, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')  # pooled
        input_as = tf.concat(1, [tf.reshape(prev_out_ms, [-1, self._ms_unit]),
                                 tf.reshape(prev_out_vs_p, [-1, self._vs_size/4 * self._vs_unit])])
        cell_as = self.cell_as(input_as, prev_cell_as, scope='as')
        cell_out_as, cell_state_as = cell_as

        # motor slow (from mf and as)
        input_ms = tf.concat(1, [tf.reshape(prev_out_mf, [-1, self._mf_unit]),
                                 tf.reshape(cell_out_as, [-1, self._as_unit])])
        cell_ms = self.cell_ms(input_ms, prev_cell_ms, scope='ms')
        cell_out_ms, cell_state_ms = cell_ms

        # motor fast (from motor input and ms)
        input_mf = tf.reshape(cell_out_ms, [-1, self._ms_unit])
        cell_mf = self.cell_mf(input_mf, prev_cell_mf, scope='mf')
        cell_out_mf, cell_state_mf = cell_mf

        # motor output
        logit_motor = tf.matmul(cell_out_as,self.W_m_out) + self.b_m_out
        logit_motor_rs = tf.reshape(logit_motor,[-1,self._out_size_mdim, self._out_size_smdim])#reshape for softmax
        pred_step_motor = tf.nn.softmax(logit_motor_rs)  # softmax connot be at the outside b.o. closed loop (it might be faster...)

        # vision output (not used)
        logit_vision = tanh_mod(tf.nn.conv2d(cell_out_vf, self.W_v_out, strides = [1,1,1,1], padding = 'SAME') + self.b_v_out)
        #logit_vision = tanh_mod(tf.add(tf.matmul(cell_out_vf,self.W_v_out),self.b_v_out))
        pred_step_vision = tf.reshape(logit_vision,[-1,self._out_size_vrow,self._out_size_vcol])

        model_out = (pred_step_vision,pred_step_motor, cell_vf[1], cell_vm[1], cell_vs[1], cell_mf[1], cell_ms[1] , cell_as[1])

        return model_out

    @lazy_property
    def prediction(self):
        # recurrent network. (same length in a batch)

        # transpose inputs for scan and make tuple
        v_t = tf.transpose(self._v_in, perm=[1, 0, 2, 3])
        m_t = tf.transpose(self._m_in, perm=[1, 0, 2, 3])
        input_t = (v_t, m_t)

        # make initializer for the scan (only support state_is_tuple)
        init_state = (self._v_init, self._m_init, self._c_vf, self._c_vm, self._c_vs, self._c_mf, self._c_ms, self._c_as)

        scan_outputs = tf.scan(lambda a, x: self.model_step(x, a), input_t,
                               initializer=init_state)

        pred_v_t, pred_m_t, c_vf_t, c_vm_t, c_vs_t, c_mf_t, c_ms_t, c_as_t = scan_outputs

        pred_v = tf.transpose(pred_v_t, perm=[1, 0, 2, 3], name='pred_v')
        pred_m = tf.transpose(pred_m_t, perm=[1, 0, 2, 3], name='pred_m')

        c_vf = tf.transpose(c_vf_t, perm=[2, 1, 0, 3, 4, 5], name='states_vf')
        c_vm = tf.transpose(c_vm_t, perm=[2, 1, 0, 3, 4, 5], name='states_vm')
        c_vs = tf.transpose(c_vs_t, perm=[2, 1, 0, 3, 4, 5], name='states_vs')
        c_mf = tf.transpose(c_mf_t, perm=[2, 1, 0, 3], name='states_mf')
        c_ms = tf.transpose(c_ms_t, perm=[2, 1, 0, 3], name='states_ms')
        c_as = tf.transpose(c_as_t, perm=[2, 1, 0, 3], name='states_as')

        return pred_v, pred_m, c_vf, c_vm, c_vs, c_mf, c_ms, c_as

    @lazy_property
    def cost(self):
        pred_v, pred_m, _, _, _, _, _, _ = self.prediction
        loss_vision = tf.reduce_mean((self._v_out*0.0 - pred_v*0.0) ** 2, name='loss_vision')
        tar_m_crop = self._m_out
        pre_m_crop = pred_m
        loss_motor = tf.reduce_mean(-tf.reduce_sum(tar_m_crop * (tf.log(pre_m_crop + self._eps) - tf.log(tar_m_crop + self._eps)), reduction_indices=[2,3]), name = 'loss_motor')

        loss = loss_motor
        return loss, loss_vision, loss_motor

    @lazy_property
    def optimize(self):
        loss_sum, _, _ = self.cost
        #optimizer = tf.train.GradientDescentOptimizer(self._lr)
        optimizer = tf.train.AdamOptimizer(self._lr)
        return optimizer.minimize(loss_sum, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE) #, aggregation_method=2


