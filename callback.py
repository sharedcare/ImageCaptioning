#!/usr/bin/env python
# title           :callback.py
# description     :List of keras.callbacks.Callback instances. List of callbacks to apply during training.
# author          :Tiancheng Luo
# date            :Apr. 26, 2018
# python_version  :3.6.3

from keras.callbacks import ModelCheckpoint, TensorBoard


def callback(path_checkpoint=None, log_dir=None):
    callbacks = []

    if path_checkpoint is not None:
        checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                     verbose=1,
                                     save_weights_only=True)
        callbacks.append(checkpoint)

    if log_dir is not None:
        tensorboard = TensorBoard(log_dir=log_dir,
                                  histogram_freq=0,
                                  write_graph=False)
        callbacks.append(tensorboard)
        
    return callbacks