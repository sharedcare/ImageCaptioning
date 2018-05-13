#!/usr/bin/env python
# title           :callback.py
# description     :List of keras.callbacks.Callback instances. List of callbacks to apply during training.
# author          :Tiancheng Luo
# date            :Apr. 26, 2018
# python_version  :3.6.3

from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau


def callback(path_checkpoint=None, log_dir=None, reduce_lr=True):
    callbacks = []

    if path_checkpoint is not None:
        checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                     verbose=1,
                                     save_weights_only=False)
        callbacks.append(checkpoint)

    if log_dir is not None:
        tensorboard = TensorBoard(log_dir=log_dir,
                                  histogram_freq=0,
                                  write_graph=False)
        callbacks.append(tensorboard)

    if reduce_lr:
        reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                   patience=5, verbose=1)
        callbacks.append(reduce)

    return callbacks
