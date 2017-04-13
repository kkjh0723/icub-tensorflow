#Import from public
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import math

#Import from custom
import model

# FLAGS (options)
tf.flags.DEFINE_string("data_dir","./../NUMPY_DATASET/","directory which contains dataset")
tf.flags.DEFINE_string("data_fn","vrpgp_db_161116_reduced.npz","data file name")
tf.flags.DEFINE_string("log_dir","./log_dir", "directory for loading trained model")
tf.flags.DEFINE_string("checkpoint","rnnmodel_min.ckpt", "set check point file name")
tf.flags.DEFINE_float("cl_ratio","0.0", "set ratio of transferring model's previous output to current input (0.0: open-loop <--> 1.0: closed-loop)") # not used
tf.flags.DEFINE_string("device","/gpu:0", "device and device number to use (e.g. /gpu:0, /cpu:0)")
tf.flags.DEFINE_integer("batch_size","8", "batch size")

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

tf.flags.DEFINE_boolean("save_states", True, "save activations and states of the model")
tf.flags.DEFINE_string("save_filename","outputs_test", "filename for saving the activations and states of the model")

tf.flags.DEFINE_boolean("train_data", True, "use training data to generate output")

flag = tf.flags.FLAGS

# check directory and FLAGs
ckpt_path = os.path.join(flag.log_dir,flag.checkpoint)
#isexist = os.path.exists(ckpt_path)
#assert isexist

if flag.cl_ratio > 1.0 or flag.cl_ratio < 0.0:
    print ("cl_ratio is out of range [0.0 1.0]: %.2f" %flag.cl_ratio)
    assert False

# DEVICE(CPU or GPU)
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
#config.gpu_options.allow_growth = True

# parameters
eps=1e-8 # epsilon for numerical stability

# robot data
print "Load datasets..."
class DataSets(object):
    pass

dbs = DataSets()
data_raw = np.load(flag.data_dir+flag.data_fn)
if flag.train_data:
    dbs.motor = data_raw['motor_tr']
    dbs.vision = data_raw['vision_tr']
    dbs.idxd = data_raw['idxd_tr']
    dbs.idxt = data_raw['idxt_tr']
else:
    dbs.motor = data_raw['motor_te']
    dbs.vision = data_raw['vision_te']
    dbs.idxd = data_raw['idxd_te']
    dbs.idxt = data_raw['idxt_te']

num_type = np.max(dbs.idxt) + 1 # human provide while testing (not used)
num_data = dbs.idxd.shape[0]

flag.out_size_vrow = dbs.vision.shape[-2]
flag.out_size_vcol = dbs.vision.shape[-1]
flag.out_size_mdim = dbs.motor.shape[-2]
flag.out_size_smdim = dbs.motor.shape[-1]


# load minmax file for motor transform
sm_file = np.genfromtxt(flag.data_dir+'/dimMinMaxFile.txt', delimiter='\t')
sm_file = sm_file.reshape([-1,2])
sm_shape = sm_file.shape
assert flag.out_size_mdim == sm_shape[0]
sm_ref = np.zeros([sm_shape[0],flag.out_size_smdim])
for i in xrange(flag.out_size_mdim):
    sm_ref[i] = np.linspace(sm_file[i,0], sm_file[i,1], num=flag.out_size_smdim)

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
#assert num_data>=flag.batch_size
save_fn = flag.log_dir + '/' + flag.save_filename + '.npz'
isexist_save = os.path.exists(save_fn)
if isexist_save:
    print "Output file is already saved. Remove it if it is out of date"
    assert False


if __name__ == '__main__':

    with tf.device(flag.device):

        # input feed
        motor = tf.placeholder(tf.float32, [None, None, flag.out_size_mdim, flag.out_size_smdim])
        vision = tf.placeholder(tf.float32, [None, None, flag.out_size_vrow, flag.out_size_vcol])

        motor_init = tf.placeholder(tf.float32, [None, flag.out_size_mdim, flag.out_size_smdim])
        vision_init = tf.placeholder(tf.float32, [None, flag.out_size_vrow, flag.out_size_vcol])

        vf_h = tf.placeholder(tf.float32, [None, flag.vf_msize[0],flag.vf_msize[1],flag.vf_unit])
        vm_h = tf.placeholder(tf.float32, [None, flag.vm_msize[0],flag.vm_msize[1],flag.vm_unit])
        vs_h = tf.placeholder(tf.float32, [None, flag.vs_msize[0],flag.vs_msize[1],flag.vs_unit])
        mf_h = tf.placeholder(tf.float32, [None, flag.mf_unit])
        ms_h = tf.placeholder(tf.float32, [None, flag.ms_unit])
        as_h = tf.placeholder(tf.float32, [None, flag.as_unit])

        vf_c = tf.placeholder(tf.float32, [None, flag.vf_msize[0], flag.vf_msize[1], flag.vf_unit])
        vm_c = tf.placeholder(tf.float32, [None, flag.vm_msize[0], flag.vm_msize[1], flag.vm_unit])
        vs_c = tf.placeholder(tf.float32, [None, flag.vs_msize[0], flag.vs_msize[1], flag.vs_unit])
        mf_c = tf.placeholder(tf.float32, [None, flag.mf_unit])
        ms_c = tf.placeholder(tf.float32, [None, flag.ms_unit])
        as_c = tf.placeholder(tf.float32, [None, flag.as_unit])

        clratio = tf.placeholder(tf.float32, [1]) #closed loop ratio (cannot be used in current architecture)
        lr = tf.placeholder(tf.float32, []) #learning rate

        vf_state = tf.nn.rnn_cell.LSTMStateTuple(vf_c, vf_h)
        vm_state = tf.nn.rnn_cell.LSTMStateTuple(vm_c, vm_h)
        vs_state = tf.nn.rnn_cell.LSTMStateTuple(vs_c, vs_h)
        mf_state = tf.nn.rnn_cell.LSTMStateTuple(mf_c, mf_h)
        ms_state = tf.nn.rnn_cell.LSTMStateTuple(ms_c, ms_h)
        as_state = tf.nn.rnn_cell.LSTMStateTuple(as_c, as_h)

        # model
        rnn_model = model.Model(vision, motor, vf_state, vm_state, vs_state, mf_state, ms_state,
                                as_state, vision_init, motor_init, clratio, lr, flag)

    saver = tf.train.Saver()  # saves variables learned during training
    with tf.Session(config=config) as sess:
        # load the ckpt file (don't need initialize)
        saver.restore(sess, ckpt_path)
        print("Model restored.")

        # testing
        max_iter = int(math.ceil(num_data / flag.batch_size))
        if num_data % flag.batch_size != 0: max_iter = max_iter + 1
        num_save_batch = max_iter
        num_save_file = int(math.ceil(max_iter / num_save_batch))
        if max_iter % num_save_batch != 0: num_save_file = num_save_file + 1
        cur_iter = max_iter
        print "num. of data:%03d. batch size:%03d. max interation:%03d num_save_file:%03d" %(num_data,flag.batch_size, max_iter, num_save_file )

        for save_iter in xrange(num_save_file):
            if cur_iter>num_save_batch:
                max_batch_iter = num_save_batch
                cur_iter = cur_iter - num_save_batch
            else:
                max_batch_iter =cur_iter
                cur_iter = 0

            # container for keeping generated outputs and states
            pred_vs = np.array([])
            pred_ms = np.array([])
            c_vfs = np.array([])
            c_vms = np.array([])
            c_vss = np.array([])
            c_mfs = np.array([])
            c_mss = np.array([])
            c_ass = np.array([])
            h_vfs = np.array([])
            h_vms = np.array([])
            h_vss = np.array([])
            h_mfs = np.array([])
            h_mss = np.array([])
            h_ass = np.array([])

            loss_step = 0.0
            loss_v_step = 0.0
            loss_m_step = 0.0
            for batch_iter in xrange(max_batch_iter):
                i = save_iter*num_save_batch + batch_iter
                print "Interation %03d/%03d in save file %03d/%03d" %(batch_iter+1,max_batch_iter,save_iter+1,num_save_file)
                # take mini-batch
                idxs = np.int32(i * flag.batch_size)
                idxe = np.int32((i + 1) * flag.batch_size if (i + 1) * flag.batch_size < num_data else num_data)
                vision_batch = np.array(dbs.vision[idxs:idxe], np.float32)
                motor_batch = np.array(dbs.motor[idxs:idxe], np.float32)
                vf_h_batch = np.array(dbs.vf_h[idxs:idxe], np.float32)
                vf_c_batch = np.array(dbs.vf_c[idxs:idxe], np.float32)
                vm_h_batch = np.array(dbs.vm_h[idxs:idxe], np.float32)
                vm_c_batch = np.array(dbs.vm_c[idxs:idxe], np.float32)
                vs_h_batch = np.array(dbs.vs_h[idxs:idxe], np.float32)
                vs_c_batch = np.array(dbs.vs_c[idxs:idxe], np.float32)
                mf_h_batch = np.array(dbs.mf_h[idxs:idxe], np.float32)
                mf_c_batch = np.array(dbs.mf_c[idxs:idxe], np.float32)
                ms_h_batch = np.array(dbs.ms_h[idxs:idxe], np.float32)
                ms_c_batch = np.array(dbs.ms_c[idxs:idxe], np.float32)
                as_h_batch = np.array(dbs.as_h[idxs:idxe], np.float32)
                as_c_batch = np.array(dbs.as_c[idxs:idxe], np.float32)

                pred_batch, cost_batch = sess.run([rnn_model.prediction,rnn_model.cost],
                                                                              feed_dict={motor: motor_batch,
                                                                                         vision: vision_batch,
                                                                                         motor_init: motor_batch[:,0,:,:],
                                                                                         vision_init: vision_batch[:,0,:,:],
                                                                                         vf_h: vf_h_batch, vm_h: vm_h_batch,
                                                                                         vs_h: vs_h_batch, mf_h: mf_h_batch,
                                                                                         ms_h: ms_h_batch,
                                                                                         as_h: as_h_batch, vf_c: vf_c_batch,
                                                                                         vm_c: vm_c_batch, vs_c: vs_c_batch,
                                                                                         mf_c: mf_c_batch, ms_c: ms_c_batch,
                                                                                         as_c: as_c_batch,
                                                                                         clratio: [flag.cl_ratio]})

                pred_v, pred_m, s_vf, s_vm, s_vs, s_mf, s_ms, s_as = pred_batch

                c_vf, h_vf = np.split(s_vf,2,axis=2)
                c_vm, h_vm = np.split(s_vm,2,axis=2)
                c_vs, h_vs = np.split(s_vs,2,axis=2)
                c_mf, h_mf = np.split(s_mf,2,axis=2)
                c_ms, h_ms = np.split(s_ms,2,axis=2)
                c_as, h_as = np.split(s_as,2,axis=2)

                if batch_iter == 0:
                    pred_vs = pred_v
                    pred_ms = pred_m
                    c_vfs = c_vf[:,:,0,:,:,:]
                    c_vms = c_vm[:,:,0,:,:,:]
                    c_vss = c_vs[:,:,0,:,:,:]
                    c_mfs = c_mf[:,:,0,:]
                    c_mss = c_ms[:,:,0,:]
                    c_ass = c_as[:,:,0,:]
                    h_vfs = h_vf[:,:,0,:,:,:]
                    h_vms = h_vm[:,:,0,:,:,:]
                    h_vss = h_vs[:,:,0,:,:,:]
                    h_mfs = h_mf[:,:,0,:]
                    h_mss = h_ms[:,:,0,:]
                    h_ass = h_as[:,:,0,:]
                else:
                    pred_vs = np.append(pred_vs, pred_v, axis = 0)
                    pred_ms = np.append(pred_ms, pred_m, axis=0)
                    c_vfs = np.append(c_vfs, c_vf[:,:,0,:,:,:], axis=0)
                    c_vms = np.append(c_vms, c_vm[:,:,0,:,:,:], axis=0)
                    c_vss = np.append(c_vss, c_vs[:,:,0,:,:,:], axis=0)
                    c_mfs = np.append(c_mfs, c_mf[:,:,0,:], axis=0)
                    c_mss = np.append(c_mss, c_ms[:,:,0,:], axis=0)
                    c_ass = np.append(c_ass, c_as[:,:,0,:], axis=0)
                    h_vfs = np.append(h_vfs, c_vf[:,:,0,:,:,:], axis=0)
                    h_vms = np.append(h_vms, c_vm[:,:,0,:,:,:], axis=0)
                    h_vss = np.append(h_vss, c_vs[:,:,0,:,:,:], axis=0)
                    h_mfs = np.append(h_mfs, c_mf[:,:,0,:], axis=0)
                    h_mss = np.append(h_mss, c_ms[:,:,0,:], axis=0)
                    h_ass = np.append(h_ass, c_as[:,:,0,:], axis=0)

                tmp_loss, tmp_loss_vision, tmp_loss_motor = cost_batch

                loss_step += tmp_loss
                loss_v_step += tmp_loss_vision
                loss_m_step += tmp_loss_motor

            loss_step /= max_iter
            loss_v_step /= max_iter
            loss_m_step /= max_iter

            print('\rLoss: %.6f. Loss_v: %.6f. Loss_m: %.6f.' % (
                loss_step, loss_v_step, loss_m_step))

            pred_vs = np.array(pred_vs)
            pred_ms = np.array(pred_ms)
            c_vfs = np.array(c_vfs)
            c_vms = np.array(c_vms)
            c_vss = np.array(c_vss)
            c_mfs = np.array(c_mfs)
            c_mss = np.array(c_mss)
            c_ass = np.array(c_ass)
            h_vfs = np.array(h_vfs)
            h_vms = np.array(h_vms)
            h_vss = np.array(h_vss)
            h_mfs = np.array(h_mfs)
            h_mss = np.array(h_mss)
            h_ass = np.array(h_ass)

            print ("Inverse Transform...")
            # take inverse transform to the motor output
            m_pred_inv = np.sum(np.multiply(pred_ms, sm_ref),axis=3)
            m_tar_inv = np.sum(np.multiply(dbs.motor[:,1:,:,:], sm_ref), axis=3)

            print ("Generation Finished.")

            # save generated outputs and states
            if flag.save_states == True:
                print "Saving the outputs and states"
                save_file_str = '_%02d' %(save_iter)
                tmpfn = flag.log_dir + '/' + flag.save_filename + save_file_str+'.npz'
                np.savez_compressed(tmpfn, v_pred=pred_vs, m_pred=pred_ms, m_pred_inv=m_pred_inv,
                                    c_vfs=c_vfs, c_vms=c_vms, c_vss=c_vss, c_mfs=c_mfs, c_mss=c_mss, c_ass=c_ass,
                                    h_vfs=h_vfs, h_vms=h_vms, h_vss=h_vss, h_mfs=h_mfs, h_mss=h_mss, h_ass=h_ass,
                                    v_tar=dbs.vision[:, 1:, :, :], m_tar=dbs.motor[:, 1:, :, :],
                                    m_tar_inv=m_tar_inv) #target starts from -1 because of initial state




