import readline
import yarp as yp
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import tensorflow as tf
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent directory to path
import model

# FLAGS (options)
# tf.flags.DEFINE_string("data_dir","./../data/dataset_np","")
tf.flags.DEFINE_string("log_dir","./../log_dir01", "directory for saving model and training log")
tf.flags.DEFINE_string("checkpoint","rnnmodel_min.ckpt", "set check point file location")
tf.flags.DEFINE_float("cl_ratio","0.0", "set ratio of transferring model's previous output to current input (0.0: open-loop <--> 1.0: closed-loop)")
tf.flags.DEFINE_string("device","/gpu:0", "device and device number to use (e.g. /gpu:0, /cpu:0)")
tf.flags.DEFINE_integer("batch_size","1", "batch size")
tf.flags.DEFINE_integer("max_leng","130", "batch size")

tf.flags.DEFINE_integer("mf_unit","64", "motor fast layer size")
tf.flags.DEFINE_integer("ms_unit","32", "motor slowlayer size")
tf.flags.DEFINE_integer("as_unit","32", "associative (PFC) layer size")
tf.flags.DEFINE_integer("vs_unit","32", "vision slow layer size")
tf.flags.DEFINE_integer("vm_unit","16", "vision middle layer size")
tf.flags.DEFINE_integer("vf_unit","8", "vision fast layer size")
tf.flags.DEFINE_integer("vs_fsize","5", "vision slow layer filter size")
tf.flags.DEFINE_integer("vm_fsize","5", "vision middle layer filter size")
tf.flags.DEFINE_integer("vf_fsize","5", "vision fast layer filter size")
tf.flags.DEFINE_integer("vo_fsize","7", "vision fast layer filter size")


tf.flags.DEFINE_boolean("use_data_vision","False", "Use training data for testing(for debugging)")
tf.flags.DEFINE_boolean("use_data_motor","False", "Use training data for testing(for debugging)")
tf.flags.DEFINE_boolean("show_vision","False", "Show current visual input")

tf.flags.DEFINE_boolean("save_states","False", "Save states of the network")
tf.flags.DEFINE_string("save_dir","./tmp_dir", "Save directory for online generated states")


flag = tf.flags.FLAGS

# Initialise YARP
yp.Network.init()

# Open ports
port = yp.BufferedPortBottle()  # This port receives softmax target from the network.
port.open("/network/softmaxTarget:o")

port_calcForward = yp.RpcServer()  # This port receives from the controller
port_calcForward.open("/network/calcForward:rpcServer")

port_img = yp.RpcServer()  # This port receives softmax target from the network.
port_img.open("/network/image:rpcServer")

# defines in controller.h
SOFTMAX_DIMENSION = 10
NUM_SIG_DIM = 10
TOTAL_SOFTMAX_DIM = SOFTMAX_DIMENSION*NUM_SIG_DIM
IMG_ROW = 48
IMG_COL = 64
DELAY_NETWORK = 0.001
CALC = 1
NOT_CALC = 0
INIT_NET = 2
STEP_INTERVAL = 1
PREFIX = 'online'
if flag.use_data_vision == True:
    PREFIX = 'data'

#Check directory and FLAGs
ckpt_path = os.path.join(flag.log_dir,flag.checkpoint)
#isexist = os.path.exists(ckpt_path)
#assert isexist

isexist_sv = os.path.exists(flag.save_dir)
if not isexist_sv:
    os.makedirs(flag.save_dir)

if flag.cl_ratio > 1.0 or flag.cl_ratio < 0.0:
    print ("cl_ratio is out of range [0.0 1.0]: %.2f" %flag.cl_ratio)
    assert False

#DEVICE(CPU or GPU)
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
#config.gpu_options.allow_growth = True

#Parameters
eps=1e-8 # epsilon for numerical stability
num_data = 1
#Network parameters
flag.out_size_vrow = IMG_ROW
flag.out_size_vcol = IMG_COL
flag.out_size_mdim = NUM_SIG_DIM #!!
flag.out_size_smdim = SOFTMAX_DIMENSION

flag.vs_msize = (flag.out_size_vrow / 4, flag.out_size_vcol / 4)
flag.vs_size = flag.vs_msize[0] * flag.vs_msize[1]

flag.vm_msize = (flag.out_size_vrow / 2, flag.out_size_vcol / 2)
flag.vm_size = flag.vm_msize[0] * flag.vm_msize[1]

flag.vf_msize = (flag.out_size_vrow, flag.out_size_vcol)
flag.vf_size = flag.vf_msize[0] * flag.vf_msize[1]



# network model
with tf.device(flag.device):
    # Input feed
    motor = tf.placeholder(tf.float32, [None, None, flag.out_size_mdim, flag.out_size_smdim])
    vision = tf.placeholder(tf.float32, [None, None, flag.out_size_vrow, flag.out_size_vcol])

    motor_init = tf.placeholder(tf.float32, [None, flag.out_size_mdim, flag.out_size_smdim])
    vision_init = tf.placeholder(tf.float32, [None, flag.out_size_vrow, flag.out_size_vcol])

    vf_h = tf.placeholder(tf.float32, [None, flag.vf_msize[0], flag.vf_msize[1], flag.vf_unit])
    vm_h = tf.placeholder(tf.float32, [None, flag.vm_msize[0], flag.vm_msize[1], flag.vm_unit])
    vs_h = tf.placeholder(tf.float32, [None, flag.vs_msize[0], flag.vs_msize[1], flag.vs_unit])
    mf_h = tf.placeholder(tf.float32, [None, flag.mf_unit])
    ms_h = tf.placeholder(tf.float32, [None, flag.ms_unit])
    as_h = tf.placeholder(tf.float32, [None, flag.as_unit])

    vf_c = tf.placeholder(tf.float32, [None, flag.vf_msize[0], flag.vf_msize[1], flag.vf_unit])
    vm_c = tf.placeholder(tf.float32, [None, flag.vm_msize[0], flag.vm_msize[1], flag.vm_unit])
    vs_c = tf.placeholder(tf.float32, [None, flag.vs_msize[0], flag.vs_msize[1], flag.vs_unit])
    mf_c = tf.placeholder(tf.float32, [None, flag.mf_unit])
    ms_c = tf.placeholder(tf.float32, [None, flag.ms_unit])
    as_c = tf.placeholder(tf.float32, [None, flag.as_unit])

    clratio = tf.placeholder(tf.float32, [1])  # closed loop ratio
    lr = tf.placeholder(tf.float32, [])  # learning rate

    vf_state = tf.nn.rnn_cell.LSTMStateTuple(vf_c, vf_h)
    vm_state = tf.nn.rnn_cell.LSTMStateTuple(vm_c, vm_h)
    vs_state = tf.nn.rnn_cell.LSTMStateTuple(vs_c, vs_h)
    mf_state = tf.nn.rnn_cell.LSTMStateTuple(mf_c, mf_h)
    ms_state = tf.nn.rnn_cell.LSTMStateTuple(ms_c, ms_h)
    as_state = tf.nn.rnn_cell.LSTMStateTuple(as_c, as_h)

    # Model
    rnn_model = model.Model(vision, motor, vf_state, vm_state, vs_state, mf_state, ms_state,
                            as_state, vision_init, motor_init, clratio, lr, flag)

#start session and restore model
saver = tf.train.Saver()
sess = tf.Session(config=config)
saver.restore(sess, ckpt_path)
print("Model restored.")

#make container
batch_size = flag.batch_size
batch_length = flag.max_leng + 1  # 1 step prediction

sim_v = np.zeros([batch_size, batch_length, flag.out_size_vrow, flag.out_size_vcol])
input_v = np.zeros([batch_size, batch_length, flag.out_size_vrow, flag.out_size_vcol])
input_m = np.zeros([batch_size, batch_length, flag.out_size_mdim, flag.out_size_smdim])
pred_v = np.zeros([batch_size, batch_length, flag.out_size_vrow, flag.out_size_vcol])
pred_m = np.zeros([batch_size, batch_length, flag.out_size_mdim, flag.out_size_smdim])
c_vf = np.zeros([batch_size, batch_length, flag.vf_msize[0], flag.vf_msize[1], flag.vf_unit])
c_vm = np.zeros([batch_size, batch_length, flag.vm_msize[0], flag.vm_msize[1], flag.vm_unit])
c_vs = np.zeros([batch_size, batch_length, flag.vs_msize[0], flag.vs_msize[1], flag.vs_unit])
c_mf = np.zeros([batch_size, batch_length, flag.mf_unit])
c_ms = np.zeros([batch_size, batch_length, flag.ms_unit])
c_as = np.zeros([batch_size, batch_length, flag.as_unit])
h_vf = np.zeros([batch_size, batch_length, flag.vf_msize[0], flag.vf_msize[1], flag.vf_unit])
h_vm = np.zeros([batch_size, batch_length, flag.vm_msize[0], flag.vm_msize[1], flag.vm_unit])
h_vs = np.zeros([batch_size, batch_length, flag.vs_msize[0], flag.vs_msize[1], flag.vs_unit])
h_mf = np.zeros([batch_size, batch_length, flag.mf_unit])
h_ms = np.zeros([batch_size, batch_length, flag.ms_unit])
h_as = np.zeros([batch_size, batch_length, flag.as_unit])

#read initial motor softmax file
fn_motor_sub = './softmaxConfig/target_home_softmax.txt'
fn_motor = fn_motor_sub
tmp_motor = np.genfromtxt(fn_motor,delimiter='\t')
motor_init_in = tmp_motor[:-1]
motor_init_in = np.tile(motor_init_in,(batch_size))
motor_init_in = motor_init_in.reshape([batch_size,flag.out_size_mdim, flag.out_size_smdim])
pred_m[:,0,:,:] = motor_init_in

#Softmax
sm_file = np.genfromtxt('./softmaxConfig/dimMinMaxFile.txt', delimiter='\t')
sm_shape = sm_file.shape
assert flag.out_size_mdim == sm_shape[0]
sm_ref = np.zeros([sm_shape[0],flag.out_size_smdim])
for i in xrange(flag.out_size_mdim):
    sm_ref[i] = np.linspace(sm_file[i,0], sm_file[i,1], num=flag.out_size_smdim)

while True:
    netOut = np.zeros([SOFTMAX_DIMENSION], dtype=np.float32)

    # receive config info from controller
    cmdConfig = yp.Bottle()
    cmdConfig.clear()
    responseConfig = yp.Bottle()
    responseConfig.clear()

    port_calcForward.read(cmdConfig, True)  # waiting until receiving calculate signal
    obj1st = cmdConfig.get(0).asInt()
    other1st = cmdConfig.get(1).asInt()
    gesture1st = cmdConfig.get(2).asInt()
    obj2nd = cmdConfig.get(3).asInt()
    other2nd = cmdConfig.get(4).asInt()
    gesture2nd = cmdConfig.get(5).asInt()
    classNum = cmdConfig.get(6).asInt()

    responseConfig.addDouble(0)
    port_calcForward.reply(responseConfig)

    if flag.use_data_vision or flag.show_vision:
        #image
        fn_image = './../data/vision/vision_%04d_%04d_%04d_%04d_%04d_%04d.txt' %(obj1st,0,0,0,0,0)
        print fn_image
        tmp_vision = np.genfromtxt(fn_image, delimiter='\t')
        vision_in = tmp_vision[:,:-1]
        vision_in = np.reshape(vision_in,[-1, flag.out_size_vrow, flag.out_size_vcol])
    if flag.use_data_motor:
        fn_label = './../data/label/target_softmax_%04d.txt' %(obj1st)
        tmp_motor_in = np.genfromtxt(fn_label, delimiter='\t')
        motor_in = tmp_motor_in[:,:-1]
        motor_in = np.reshape(motor_in, [-1, flag.out_size_mdim, flag.out_size_smdim])

    #start sequence
    step = 0
    while True:
        recvImg = np.zeros([IMG_ROW * IMG_COL])

        cmd = yp.Bottle()
        cmd.clear()
        response = yp.Bottle()
        response.clear()
        port_img.read(cmd, True)  # waiting until receiving image.
        if (cmd.size() != (IMG_ROW * IMG_COL)):
            print "\nProblem while reciving the 'input_img' from the vision module!\n"
            assert 0;

        response.addString("...received")
        port_img.reply(response)

        # Check for calculation
        cmdCalc = yp.Bottle()
        cmdCalc.clear()
        responseCalc = yp.Bottle()
        responseCalc.clear()

        port_calcForward.read(cmdCalc, True) # waiting until receiving image.
        if (cmdCalc.size() != 1):
            print  "\nProblem while reciving the 'cmdCalc' from the controller module!\n";
            assert 0

        calcNow = cmdCalc.get(0).asInt()

        # FORWARD CALC.
        if (calcNow == INIT_NET):

            # Save network outputs
            if flag.save_states:
                print "Save states to " + flag.save_dir
                fn_sim_v = flag.save_dir + '/' + PREFIX + '_simVision_%04d_%04d_%04d_%04d_%04d_%04d.csv' % (
                    obj1st, other1st, gesture1st, obj2nd, other2nd, gesture2nd)
                fn_input_v = flag.save_dir + '/' + PREFIX + '_inputVision_%04d_%04d_%04d_%04d_%04d_%04d.csv' % (
                obj1st, other1st, gesture1st, obj2nd, other2nd, gesture2nd)
                fn_pred_m = flag.save_dir + '/' + PREFIX + '_outputMotor_%04d_%04d_%04d_%04d_%04d_%04d.csv' % (
                obj1st, other1st, gesture1st, obj2nd, other2nd, gesture2nd)
                fn_vf_h = flag.save_dir + '/' + PREFIX + '_stateVfHidden_%04d_%04d_%04d_%04d_%04d_%04d.csv' % (
                obj1st, other1st, gesture1st, obj2nd, other2nd, gesture2nd)
                fn_vm_h = flag.save_dir + '/' + PREFIX + '_stateVmHidden_%04d_%04d_%04d_%04d_%04d_%04d.csv' % (
                obj1st, other1st, gesture1st, obj2nd, other2nd, gesture2nd)
                fn_vs_h = flag.save_dir + '/' + PREFIX + '_stateVsHidden_%04d_%04d_%04d_%04d_%04d_%04d.csv' % (
                obj1st, other1st, gesture1st, obj2nd, other2nd, gesture2nd)
                fn_mf_h = flag.save_dir + '/' + PREFIX + '_stateMfHidden_%04d_%04d_%04d_%04d_%04d_%04d.csv' % (
                obj1st, other1st, gesture1st, obj2nd, other2nd, gesture2nd)
                fn_ms_h = flag.save_dir + '/' + PREFIX + '_stateMsHidden_%04d_%04d_%04d_%04d_%04d_%04d.csv' % (
                obj1st, other1st, gesture1st, obj2nd, other2nd, gesture2nd)
                fn_as_h = flag.save_dir + '/' + PREFIX + '_stateAsHidden_%04d_%04d_%04d_%04d_%04d_%04d.csv' % (
                obj1st, other1st, gesture1st, obj2nd, other2nd, gesture2nd)

                tmp_v_sim = sim_v.reshape([batch_length, -1])
                np.savetxt(fn_sim_v, tmp_v_sim, delimiter=',')
                print fn_sim_v + " is saved"

                tmp_v = input_v.reshape([batch_length,-1])
                np.savetxt(fn_input_v, tmp_v, delimiter=',')
                print fn_input_v + " is saved"

                tmp_m_pred_inv = np.sum(np.multiply(pred_m[0], sm_ref), axis=2)
                np.savetxt(fn_pred_m, tmp_m_pred_inv, delimiter=',')
                print fn_pred_m + " is saved"

                tmp_vf = h_vf[0].reshape([batch_length,-1])
                np.savetxt(fn_vf_h, tmp_vf, delimiter=',')
                print fn_vf_h + " is saved"
                tmp_vm = h_vm[0].reshape([batch_length, -1])
                np.savetxt(fn_vm_h,tmp_vm, delimiter=',')
                print fn_vm_h + " is saved"
                tmp_vs = h_vs[0].reshape([batch_length, -1])
                np.savetxt(fn_vs_h, tmp_vs, delimiter=',')
                print fn_vs_h + " is saved"

                np.savetxt(fn_mf_h, h_mf[0], delimiter=',')
                print fn_mf_h + " is saved"
                np.savetxt(fn_ms_h, h_ms[0], delimiter=',')
                print fn_ms_h + " is saved"
                np.savetxt(fn_as_h, h_as[0], delimiter=',')
                print fn_as_h + " is saved"


            print "Start initializing the network..."

            #network initialize
            sim_v = sim_v*0.0
            input_v = input_v*0.0
            input_m = input_m*0.0
            pred_v = pred_v*0.0
            pred_m = pred_m*0.0
            c_vf = c_vf*0.0
            c_vm = c_vm*0.0
            c_vs = c_vs*0.0
            c_mf = c_mf*0.0
            c_ms = c_ms*0.0
            c_as = c_as*0.0
            h_vf = h_vf*0.0
            h_vm = h_vm*0.0
            h_vs = h_vs*0.0
            h_mf = h_mf*0.0
            h_ms = h_ms*0.0
            h_as = h_as*0.0

            initNetResp = yp.Bottle()
            initNetResp.clear()
            initNetResp.addDouble(0)
            port_calcForward.reply(initNetResp)
            print "...done \n"
            break

        elif (calcNow == CALC): # if (calcNow)

            # Now we have image here
            for idxp in xrange(cmd.size()):
                recvImg[idxp] = cmd.get(idxp).asDouble()

            sim_v[0, step] = recvImg.reshape([flag.out_size_vrow, flag.out_size_vcol])
            input_v[0, step] = sim_v[0, step]
            input_m[0, step] = pred_m[0, step]

            tmp_step = step
            if step == flag.max_leng: tmp_step = flag.max_leng

            if flag.use_data_vision == True:
                input_v[0, step] = vision_in[tmp_step]

            if flag.show_vision:
                # draw vision
                ax1 = plt.subplot(1, 3, 1)
                plt.imshow(input_v[0, step], cmap=plt.gray(), vmin=-1.0, vmax=1.0, origin='upper')
                plt.axis('off')
                ax2 = plt.subplot(1, 3, 2)
                plt.imshow(vision_in[tmp_step], cmap=plt.gray(), vmin=-1.0, vmax=1.0, origin='upper')
                plt.axis('off')
                ax3 = plt.subplot(1, 3, 3)
                plt.imshow((input_v[0, step] - vision_in[tmp_step]) ** 2, cmap=plt.gray(), vmin=0.0, vmax=1.0,
                           origin='upper')
                plt.axis('off')

                plt.tight_layout()
                plt.draw()
                plt.pause(DELAY_NETWORK)



            # network calculate step
            #feed_dict={motor: }need to be input_m
            pred_v_step, pred_m_step, s_vf_step, s_vm_step, s_vs_step, s_mf_step, s_ms_step, s_as_step = sess.run(
                rnn_model.prediction,
                feed_dict={motor: input_m[:, step:step + 2, :, :],
                           vision: input_v[:, step:step + 2, :, :],
                           motor_init: pred_m[:, step, :, :],
                           vision_init: pred_v[:, step, :, :],
                           vf_h: h_vf[:, step, :, :, :], vm_h: h_vm[:, step, :, :, :],
                           vs_h: h_vs[:, step, :, :, :], mf_h: h_mf[:, step, :],
                           ms_h: h_ms[:, step, :],
                           as_h: h_as[:, step, :], vf_c: c_vf[:, step, :, :, :],
                           vm_c: c_vm[:, step, :, :, :], vs_c: c_vs[:, step, :, :, :],
                           mf_c: c_mf[:, step, :], ms_c: c_ms[:, step, :],
                           as_c: c_as[:, step, :],
                           clratio: [flag.cl_ratio]})

            c_vf_step, h_vf_step = np.split(s_vf_step, 2, axis=2)
            c_vm_step, h_vm_step = np.split(s_vm_step, 2, axis=2)
            c_vs_step, h_vs_step = np.split(s_vs_step, 2, axis=2)
            c_mf_step, h_mf_step = np.split(s_mf_step, 2, axis=2)
            c_ms_step, h_ms_step = np.split(s_ms_step, 2, axis=2)
            c_as_step, h_as_step = np.split(s_as_step, 2, axis=2)

            pred_v[:, step + 1, :, :] = pred_v_step[:, 0, :, :]
            pred_m[:, step + 1, :, :] = pred_m_step[:, 0, :, :]
            c_vf[:, step + 1, :, :, :] = c_vf_step[:, 0, 0, :, :, :]
            c_vm[:, step + 1, :, :, :] = c_vm_step[:, 0, 0, :, :, :]
            c_vs[:, step + 1, :, :, :] = c_vs_step[:, 0, 0, :, :, :]
            c_mf[:, step + 1, :] = c_mf_step[:, 0, 0, :]
            c_ms[:, step + 1, :] = c_ms_step[:, 0, 0, :]
            c_as[:, step + 1, :] = c_as_step[:, 0, 0, :]
            h_vf[:, step + 1, :, :, :] = h_vf_step[:, 0, 0, :, :, :]
            h_vm[:, step + 1, :, :, :] = h_vm_step[:, 0, 0, :, :, :]
            h_vs[:, step + 1, :, :, :] = h_vs_step[:, 0, 0, :, :, :]
            h_mf[:, step + 1, :] = h_mf_step[:, 0, 0, :]
            h_ms[:, step + 1, :] = h_ms_step[:, 0, 0, :]
            h_as[:, step + 1, :] = h_as_step[:, 0, 0, :]

            # cnn.Calculate_online(recvImg, netOut, step, (double)obj1st, (double)other1st, (double)gesture1st, (double)classNum);
            netOut = np.zeros([TOTAL_SOFTMAX_DIM])
            # netOut[:TOTAL_SOFTMAX_DIM-SOFTMAX_DIMENSION] = pred_m_step.reshape([flag.out_size_mdim*flag.out_size_smdim])
            netOut[:TOTAL_SOFTMAX_DIM] = pred_m_step.reshape(
                [flag.out_size_mdim * flag.out_size_smdim])
            if flag.use_data_motor:
                netOut[:TOTAL_SOFTMAX_DIM] = motor_in[step].reshape([flag.out_size_mdim * flag.out_size_smdim])

            #pass to the controller
            output = port.prepare()
            output.clear()

            for idxSM in xrange(TOTAL_SOFTMAX_DIM):
                output.addDouble(netOut[idxSM])

            port.write()

            responseCalc.addString("...Calc Done")

        else:
            print "calc not done. \n"
            responseCalc.addString("...Calc NOT Done")

        yp.Time.delay(DELAY_NETWORK)

        #End of forward dynamic
        port_calcForward.reply(responseCalc)
        step += STEP_INTERVAL
        print step

del eps
