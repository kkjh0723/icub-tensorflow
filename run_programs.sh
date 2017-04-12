#!/bin/bash

FILES=(/usr/lib/libtcmalloc_minimal.so.4)


LD_PRELOAD="${FILES[@]}" CUDA_VISIBLE_DEVICES=0 python train_rnn.py --device /gpu:0 --log_dir ./log_dir01_01 --max_epoch 20000 --print_epoch 100 --batch_size 32 --lr 3e-3 --data_dir ./data/dataset_np/ --data_fn icub_db_simple.npz
LD_PRELOAD="${FILES[@]}" CUDA_VISIBLE_DEVICES=0 python test_rnn.py --device /gpu:0 --log_dir ./log_dir01_01 --data_dir ./data/dataset_np/ --data_fn icub_db_simple.npz --train_data True --save_filename outputs_test_tr --batch_size 32
#LD_PRELOAD="${FILES[@]}" CUDA_VISIBLE_DEVICES=0 python test_rnn.py --device /gpu:0 --log_dir ./log_dir01_01 --data_dir ./data/dataset_np/ --data_fn icub_db_simple_pre.npz --train_data False --save_filename outputs_test_te --batch_size 32


