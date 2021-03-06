import numpy as np
import matplotlib.pyplot as plt
import time
import os

#Names
trialname = '_simple'
dir_motor = './../label/'
dir_vision = './../vision/'
fn_db = './icub_db'+trialname+'.npz'

#load data list
dlist_tr = np.genfromtxt('./../TRIAL_LIST/trialList.txt',delimiter='\t')
dlist_tr = np.int32(dlist_tr)
print dlist_tr.shape
num_data_tr = dlist_tr.shape[0]

dlist_te = np.genfromtxt('./../TRIAL_LIST/trialList.txt',delimiter='\t')
dlist_te = np.int32(dlist_te)
print dlist_te.shape
num_data_te = dlist_te.shape[0]

#dim
vrow = 48
vcol = 64
motor = 10
motor_sm = 10
motor_dim = motor*motor_sm

#length
#tlth = 20
lth = 130

motors_tr = np.array([])
visions_tr = np.array([])
idx_data_tr = np.zeros([num_data_tr],np.int32) #index data
idx_type_tr = np.zeros([num_data_tr],np.int32) #index gesture type

motors_te = np.array([])
visions_te = np.array([])
idx_data_te = np.zeros([num_data_te],np.int32) #index data
idx_type_te = np.zeros([num_data_te],np.int32) #index gesture type


for i in xrange(num_data_te):
    seq_idx = dlist_te[i, 6]
    seq_tar = dlist_te[i,0]
    #seq_dum = dlist_te[i,1]
    #seq_gesture = dlist_te[i,2]
    seq_dum = 0
    seq_gesture = 0
    seq_zeros = dlist_te[i,3]

    #target info (?)
    xpos = seq_tar//1000
    ypos = seq_tar//100%10
    tar_pos = seq_tar//10%10

    print 'Loading test trial #%d target:%d, dummy:%d, gesture:%d' %(seq_idx, seq_tar, seq_dum, seq_gesture)

    #load motor file (copy first time step)
    fn_motor_sub = 'target_softmax_%s.txt' %(seq_tar)
    fn_motor = dir_motor+fn_motor_sub
    #print fn_motor
    tmp_motor = np.genfromtxt(fn_motor,delimiter='\t')      
    tmp_motor = tmp_motor[:lth,:motor_dim]
    tmp_motor = np.append([tmp_motor[0]],tmp_motor,axis=0)
    tmp_motor = np.reshape(tmp_motor,[-1,motor,motor_sm])    
    if i == 0 :
        motors_te = np.array([tmp_motor])
    else:
        motors_te = np.append(motors_te,[tmp_motor],axis=0)

    #load vision file (copy last time step)
    fn_vision_sub = 'vision_%04d_%04d_%04d_%04d_%04d_%04d.txt' % (seq_tar, seq_dum, seq_gesture, seq_zeros, seq_zeros, seq_zeros)
    fn_vision = dir_vision + fn_vision_sub
    tmp_vision = np.genfromtxt(fn_vision,delimiter='\t')
    tmp_vision = tmp_vision[:lth,:-1]
    tmp_vision = np.append(tmp_vision, [tmp_vision[-1]], axis=0)
    tmp_vision = tmp_vision.reshape([-1,vrow,vcol])
    if i == 0 :
        visions_te = np.array([tmp_vision])
    else:
        visions_te = np.append(visions_te,[tmp_vision],axis=0)

    #idxes
    idx_data_te[i] = seq_idx
    idx_type_te[i] = seq_gesture
    #print idx_data[i], idx_pos[i], idx_type[i]

#for i in xrange(num_data%2*3):
for i in xrange(num_data_tr):
    seq_idx = dlist_tr[i, 6]
    seq_tar = dlist_tr[i,0]
    #seq_dum = dlist_te[i,1]
    #seq_gesture = dlist_te[i,2]
    seq_dum = 0
    seq_gesture = 0
    seq_zeros = dlist_tr[i,3]

    #target info (?)
    xpos = seq_tar//1000
    ypos = seq_tar//100%10
    tar_pos = seq_tar//10%10

    print 'Loading training trial #%d target:%d, dummy:%d, gesture:%d' %(seq_idx, seq_tar, seq_dum, seq_gesture)

    #load motor file (copy first time step)
    fn_motor_sub = 'target_softmax_%s.txt' %(seq_tar)
    fn_motor = dir_motor+fn_motor_sub
    #print fn_motor
    tmp_motor = np.genfromtxt(fn_motor,delimiter='\t')    
    tmp_motor = tmp_motor[:lth,:motor_dim]
    tmp_motor = np.append([tmp_motor[0]],tmp_motor,axis=0)
    tmp_motor = np.reshape(tmp_motor,[-1,motor,motor_sm])
    if i == 0 :
        motors_tr = np.array([tmp_motor])
    else:
        motors_tr = np.append(motors_tr,[tmp_motor],axis=0)

    #load vision file (copy last time step)
    fn_vision_sub = 'vision_%04d_%04d_%04d_%04d_%04d_%04d.txt' % (seq_tar, seq_dum, seq_gesture, seq_zeros, seq_zeros, seq_zeros)
    fn_vision = dir_vision + fn_vision_sub
    tmp_vision = np.genfromtxt(fn_vision,delimiter='\t')
    tmp_vision = tmp_vision[:lth,:-1]
    tmp_vision = np.append(tmp_vision, [tmp_vision[-1]], axis=0)
    tmp_vision = tmp_vision.reshape([-1,vrow,vcol])
    if i == 0 :
        visions_tr = np.array([tmp_vision])
    else:
        visions_tr = np.append(visions_tr,[tmp_vision],axis=0)

    #idxes
    idx_data_tr[i] = seq_idx
    idx_type_tr[i] = seq_gesture
    #print idx_data[i], idx_pos[i], idx_type[i]



print motors_tr.shape
print visions_tr.shape
print idx_data_tr.shape
print idx_type_tr.shape

print motors_te.shape
print visions_te.shape
print idx_data_te.shape
print idx_type_te.shape

np.savez_compressed(fn_db, motor_tr =motors_tr, vision_tr = visions_tr, idxd_tr = idx_data_tr, idxt_tr = idx_type_tr,
                    motor_te=motors_te, vision_te=visions_te, idxd_te=idx_data_te, idxt_te=idx_type_te)

'''
for i in range(85):
    print i
    plt.imshow(visions_te[0,i,:,:],cmap=plt.gray())
    plt.pause(1.0/60.0)
    plt.draw()
'''


