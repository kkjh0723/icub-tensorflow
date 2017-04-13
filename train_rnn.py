
#Import from public
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import math
from tensorflow.python.framework import ops

#Import from custom
import model

# FLAGS (options)
tf.flags.DEFINE_string("data_dir","./../NUMPY_DATASET/","directory which contains dataset")
tf.flags.DEFINE_string("data_fn","vrpgp_db_161116_reduced.npz","data file name")
tf.flags.DEFINE_string("log_dir","./log_dir", "directory for saving model checkpoint file and training log")
tf.flags.DEFINE_string("checkpoint","rnnmodel_min.ckpt", "set check point file name")
tf.flags.DEFINE_string("device","/gpu:0", "device and device number to use (e.g. /gpu:0, /cpu:0)")
tf.flags.DEFINE_integer("batch_size","8", "batch size")
tf.flags.DEFINE_integer("max_epoch","10000", "maximum number of epochs to train")
tf.flags.DEFINE_float("lr","1e-3", "learning rate")
tf.flags.DEFINE_float("drop_out","0.7", "dropout in vision input keep_prob rate (0,1]")
#tf.flags.DEFINE_float("input_noise_m","0.3", "Injecting noise in motor input (0,1]")

tf.flags.DEFINE_integer("mf_unit","64", "motor fast layer size")
tf.flags.DEFINE_integer("ms_unit","32", "motor slow layer size")
tf.flags.DEFINE_integer("as_unit","32", "associative (PFC) layer size")
tf.flags.DEFINE_integer("vs_unit","32", "vision slow layer size")
tf.flags.DEFINE_integer("vm_unit","16", "vision middle layer size")
tf.flags.DEFINE_integer("vf_unit","8", "vision fast layer size")
tf.flags.DEFINE_integer("vs_fsize","5", "vision slow layer filter size")
tf.flags.DEFINE_integer("vm_fsize","5", "vision middle layer filter size")
tf.flags.DEFINE_integer("vf_fsize","5", "vision fast layer filter size")
tf.flags.DEFINE_integer("vo_fsize","7", "vision output layer filter size")

tf.flags.DEFINE_boolean("load_ckpt", False, "Load previously trained checkpoint file")

tf.flags.DEFINE_integer("print_epoch","100", "number of epochs to print the cost")
tf.flags.DEFINE_integer("min_save_ratio","4", "starts to save check point from (max_epoch/min_save_ratio)")
tf.flags.DEFINE_integer("max_to_keep","5", "maximum number of ckpt file to keep")
#tf.flags.DEFINE_boolean("read_attn", True, "enable attention for reader")
#tf.flags.DEFINE_boolean("write_attn",True, "enable attention for writer")
flag = tf.flags.FLAGS

# make directory
isdir = os.path.exists(flag.log_dir)
if not isdir:
    os.makedirs(flag.log_dir)

# DEVICE(CPU or GPU)
device_name = flag.device[0:4]
if device_name != '/cpu' and device_name != '/gpu':
    print 'device sould be cpu or gpu'
    assert False

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)

# parameters
min_save_epoch = flag.max_epoch/flag.min_save_ratio
print_epoch = flag.print_epoch
eps=1e-8 # epsilon for numerical stability
ckpt_file = os.path.join(flag.log_dir, flag.checkpoint)
keep_interval = flag.max_epoch/flag.max_to_keep

# robot data
print "Load datasets..."
class DataSets(object):
    pass

dbs = DataSets()
data_raw = np.load(flag.data_dir+flag.data_fn)
dbs.motor_tr = data_raw['motor_tr']
dbs.vision_tr = data_raw['vision_tr']
dbs.idxd_tr = data_raw['idxd_tr']
dbs.idxt_tr = data_raw['idxt_tr']
dbs.motor_te = data_raw['motor_te']
dbs.vision_te = data_raw['vision_te']
dbs.idxd_te = data_raw['idxd_te']
dbs.idxt_te = data_raw['idxt_te']

num_type = np.max(dbs.idxt_tr) + 1 # human provide while testing (not used)
num_data = dbs.idxd_tr.shape[0]
num_data_te = dbs.idxd_te.shape[0]

flag.out_size_vrow = dbs.vision_tr.shape[-2]
flag.out_size_vcol = dbs.vision_tr.shape[-1]
flag.out_size_mdim = dbs.motor_tr.shape[-2]
flag.out_size_smdim = dbs.motor_tr.shape[-1]

# network parameters
flag.vs_msize = (flag.out_size_vrow / 4, flag.out_size_vcol / 4)
flag.vs_size = flag.vs_msize[0] * flag.vs_msize[1]

flag.vm_msize = (flag.out_size_vrow / 2, flag.out_size_vcol / 2)
flag.vm_size = flag.vm_msize[0] * flag.vm_msize[1]

flag.vf_msize = (flag.out_size_vrow, flag.out_size_vcol)
flag.vf_size = flag.vf_msize[0] * flag.vf_msize[1]

# initial state
dbs.vf_h = np.zeros([num_data,flag.vf_msize[0],flag.vf_msize[1],flag.vf_unit])
dbs.vf_c = np.zeros([num_data,flag.vf_msize[0],flag.vf_msize[1],flag.vf_unit])

dbs.vm_h = np.zeros([num_data,flag.vm_msize[0],flag.vm_msize[1],flag.vm_unit])
dbs.vm_c = np.zeros([num_data,flag.vm_msize[0],flag.vm_msize[1],flag.vm_unit])

dbs.vs_h = np.zeros([num_data,flag.vs_msize[0],flag.vs_msize[1],flag.vs_unit])
dbs.vs_c = np.zeros([num_data,flag.vs_msize[0],flag.vs_msize[1],flag.vs_unit])

dbs.mf_h = np.zeros([num_data,flag.mf_unit])
dbs.mf_c = np.zeros([num_data,flag.mf_unit])

dbs.ms_h = np.zeros([num_data,flag.ms_unit])
dbs.ms_c = np.zeros([num_data,flag.ms_unit])

dbs.as_h = np.zeros([num_data,flag.as_unit])
dbs.as_c = np.zeros([num_data,flag.as_unit])

# check flag
assert num_data>=flag.batch_size

# save flag
f_flag = open(os.path.join(flag.log_dir, "flags.txt"), 'w')
for key, value in tf.flags.FLAGS.__flags.iteritems():
    f_flag.write("%s\t%s\n" % (key, value))
f_flag.close()

if __name__ == '__main__':

    '''
    while True:
        a = 10
    '''

    with tf.device(flag.device):

        # convert numpy array to TF tensor
        all_motor_tr = ops.convert_to_tensor(dbs.motor_tr, dtype=tf.float32)
        all_vision_tr = ops.convert_to_tensor(dbs.vision_tr, dtype=tf.float32)
        all_motor_tr_init = ops.convert_to_tensor(dbs.motor_tr[:,0,:,:]*0.0, dtype=tf.float32)
        all_vision_tr_init = ops.convert_to_tensor(dbs.vision_tr[:,0,:,:]*0.0, dtype=tf.float32)

        all_motor_te = ops.convert_to_tensor(dbs.motor_te, dtype=tf.float32)
        all_vision_te = ops.convert_to_tensor(dbs.vision_te, dtype=tf.float32)
        all_motor_te_init = ops.convert_to_tensor(dbs.motor_te[:, 0, :, :] * 0.0, dtype=tf.float32)
        all_vision_te_init = ops.convert_to_tensor(dbs.vision_te[:, 0, :, :] * 0.0, dtype=tf.float32)

        all_vf_h = ops.convert_to_tensor(dbs.vf_h, dtype=tf.float32)
        all_vm_h = ops.convert_to_tensor(dbs.vm_h, dtype=tf.float32)
        all_vs_h = ops.convert_to_tensor(dbs.vs_h, dtype=tf.float32)
        all_mf_h = ops.convert_to_tensor(dbs.mf_h, dtype=tf.float32)
        all_ms_h = ops.convert_to_tensor(dbs.ms_h, dtype=tf.float32)
        all_as_h = ops.convert_to_tensor(dbs.as_h, dtype=tf.float32)

        all_vf_c = ops.convert_to_tensor(dbs.vf_c, dtype=tf.float32)
        all_vm_c = ops.convert_to_tensor(dbs.vm_c, dtype=tf.float32)
        all_vs_c = ops.convert_to_tensor(dbs.vs_c, dtype=tf.float32)
        all_mf_c = ops.convert_to_tensor(dbs.mf_c, dtype=tf.float32)
        all_ms_c = ops.convert_to_tensor(dbs.ms_c, dtype=tf.float32)
        all_as_c = ops.convert_to_tensor(dbs.as_c, dtype=tf.float32)

        # input feed
        clratio = tf.placeholder(tf.float32, [1])  # closed loop ratio (cannot be used in current architecture)
        lr = tf.placeholder(tf.float32, [])  # learning rate

        # queue
        train_input_queue = tf.train.slice_input_producer(
            [all_vision_tr, all_motor_tr, all_vf_c, all_vf_h, all_vm_c, all_vm_h, all_vs_c, all_vs_h, all_mf_c,
             all_mf_h, all_ms_c, all_ms_h, all_as_c, all_as_h, all_vision_tr_init, all_motor_tr_init],
            shuffle=False,
            capacity=5 * flag.batch_size
        )

        test_input_queue = tf.train.slice_input_producer(
            [all_vision_te, all_motor_te, all_vf_c, all_vf_h, all_vm_c, all_vm_h, all_vs_c, all_vs_h, all_mf_c,
             all_mf_h, all_ms_c, all_ms_h, all_as_c, all_as_h, all_vision_te_init, all_motor_te_init],
            shuffle=False,
            capacity=5 * flag.batch_size
        )

        vision_batch_tr, motor_batch_tr, vf_c_batch_tr, vf_h_batch_tr, vm_c_batch_tr, vm_h_batch_tr, vs_c_batch_tr, vs_h_batch_tr, \
        mf_c_batch_tr, mf_h_batch_tr, ms_c_batch_tr, ms_h_batch_tr, as_c_batch_tr, as_h_batch_tr, v_init_batch_tr, m_init_batch_tr = tf.train.shuffle_batch(
            train_input_queue,
            # [train_input_queue[0], train_input_queue[1],train_input_queue[2], train_input_queue[3], train_input_queue[4], train_input_queue[5]],
            batch_size=num_data_te,
            capacity=5 * num_data_te,
            min_after_dequeue=1 * num_data_te,
            enqueue_many=False,
            allow_smaller_final_batch=True
            # ,num_threads=1
        )

        vision_batch_te, motor_batch_te, vf_c_batch_te, vf_h_batch_te, vm_c_batch_te, vm_h_batch_te, vs_c_batch_te, vs_h_batch_te, \
        mf_c_batch_te, mf_h_batch_te, ms_c_batch_te, ms_h_batch_te, as_c_batch_te, as_h_batch_te, v_init_batch_te, m_init_batch_te = tf.train.batch(
            test_input_queue,
            # [train_input_queue[0], train_input_queue[1],train_input_queue[2], train_input_queue[3], train_input_queue[4], train_input_queue[5]],
            batch_size=flag.batch_size,
            capacity=5 * flag.batch_size,
            enqueue_many=False,
            allow_smaller_final_batch=True
            # ,num_threads=1
        )

        vf_batch_tr = tf.nn.rnn_cell.LSTMStateTuple(vf_c_batch_tr, vf_h_batch_tr)
        vm_batch_tr = tf.nn.rnn_cell.LSTMStateTuple(vm_c_batch_tr, vm_h_batch_tr)
        vs_batch_tr = tf.nn.rnn_cell.LSTMStateTuple(vs_c_batch_tr, vs_h_batch_tr)
        mf_batch_tr = tf.nn.rnn_cell.LSTMStateTuple(mf_c_batch_tr, mf_h_batch_tr)
        ms_batch_tr = tf.nn.rnn_cell.LSTMStateTuple(ms_c_batch_tr, ms_h_batch_tr)
        as_batch_tr = tf.nn.rnn_cell.LSTMStateTuple(as_c_batch_tr, as_h_batch_tr)

        vf_batch_te = tf.nn.rnn_cell.LSTMStateTuple(vf_c_batch_te, vf_h_batch_te)
        vm_batch_te = tf.nn.rnn_cell.LSTMStateTuple(vm_c_batch_te, vm_h_batch_te)
        vs_batch_te = tf.nn.rnn_cell.LSTMStateTuple(vs_c_batch_te, vs_h_batch_te)
        mf_batch_te = tf.nn.rnn_cell.LSTMStateTuple(mf_c_batch_te, mf_h_batch_te)
        ms_batch_te = tf.nn.rnn_cell.LSTMStateTuple(ms_c_batch_te, ms_h_batch_te)
        as_batch_te = tf.nn.rnn_cell.LSTMStateTuple(as_c_batch_te, as_h_batch_te)


        # model
        vision_batch_tr_drop = tf.nn.dropout(vision_batch_tr, flag.drop_out)
        rnn_model_tr = model.Model(vision_batch_tr_drop, motor_batch_tr, vf_batch_tr, vm_batch_tr, vs_batch_tr, mf_batch_tr, ms_batch_tr, as_batch_tr,
                                v_init_batch_tr, m_init_batch_tr, clratio, lr, flag)
        tf.get_variable_scope().reuse_variables()
        rnn_model_te = model.Model(vision_batch_te, motor_batch_te, vf_batch_te, vm_batch_te, vs_batch_te, mf_batch_te, ms_batch_te, as_batch_te,
                            v_init_batch_te, m_init_batch_te, clratio, lr, flag)

    # for saving ckpt and error
    saver = tf.train.Saver(max_to_keep = flag.max_to_keep)  # saves variables learned during training
    if flag.load_ckpt == False:
        f = open(os.path.join(flag.log_dir, "loss.txt"), 'w')
    else:
        f = open(os.path.join(flag.log_dir, "loss.txt"), 'a')

    # set min loss
    min_loss_step = 99999.0
    min_loss_epoch = 0

    with tf.Session(config=config) as sess:

        if flag.load_ckpt == False:
            sess.run(tf.initialize_all_variables())
        else:
            sess.run(tf.initialize_all_variables())
            saver.restore(sess, ckpt_file)
            print("Model restored.")

        # initialize the queue threads to start to shovel data
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        tset = time.time()
        cl_schedule = 0.0 # for closed loop training which cannot be used in current architecture
        max_iter_tr = int(math.ceil(num_data/flag.batch_size))
        max_iter_te = int(math.ceil(num_data_te / flag.batch_size))
        for epoch in xrange(flag.max_epoch):
            # cl schedule
            '''
            if (epoch + 1) % 500 == 0:
                cl_schedule = cl_schedule + 0.1
                if (cl_schedule > 1.0): cl_schedule = 1.0
            '''
            # learning(optimization)
            for i in xrange(max_iter_tr):
                sess.run(rnn_model_tr.optimize, feed_dict={clratio: [cl_schedule], lr:flag.lr})

            # check loss
            if (epoch + 1) % print_epoch == 0:
                interval = time.time() - tset
                loss_step = 0.0
                loss_v_step = 0.0
                loss_m_step = 0.0
                for i in xrange(max_iter_tr):
                    tmp_loss, tmp_loss_v, tmp_loss_m = sess.run(rnn_model_tr.cost, feed_dict={clratio: [cl_schedule], lr:flag.lr})
                    loss_step += tmp_loss
                    loss_v_step += tmp_loss_v
                    loss_m_step += tmp_loss_m

                loss_step /= max_iter_tr
                loss_v_step /= max_iter_tr
                loss_m_step /= max_iter_tr

                if loss_step < min_loss_step and epoch > min_save_epoch:
                    min_loss_step = loss_step
                    min_loss_epoch = epoch
                    saver.save(sess, ckpt_file)

                loss_step_te = 0.0
                loss_v_step_te = 0.0
                loss_m_step_te = 0.0
                for j in xrange(1):
                    tmp_loss_te, tmp_loss_v_te, tmp_loss_m_te = sess.run(rnn_model_te.cost,
                                                                feed_dict={clratio: [cl_schedule], lr: flag.lr})
                    loss_step_te += tmp_loss_te
                    loss_v_step_te += tmp_loss_v_te
                    loss_m_step_te += tmp_loss_m_te

                loss_step_te /= 1
                loss_v_step_te /= 1
                loss_m_step_te /= 1

                print('\rStep %d. Loss_tr: %.6f. Loss_te: %.6f. Time(/%d epochs): %.4f, CL: %.1f' % (
                    epoch + 1, loss_step, loss_step_te, print_epoch, interval, cl_schedule))
                f.write("%d\t%.9f\t%.9f\n" % (epoch + 1, loss_step, loss_step_te))
                tset = time.time()

                    # print("Model saved in file: %s" % saver.save(sess, ckpt_file))
            if(epoch + 1) % keep_interval == 0:
                saver.save(sess, ckpt_file, global_step=epoch)
                

        coord.request_stop()
        coord.join(threads)
        sess.close()

        print("Model saved at epoch: %d, minimum loss: %.6f" % (min_loss_epoch + 1, min_loss_step))

    f.close()
