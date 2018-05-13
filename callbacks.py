#!/usr/bin/env python
# title           :callback.py
# description     :List of keras.callbacks.Callback instances. List of callbacks to apply during training.
# author          :Tiancheng Luo
# date            :Apr. 26, 2018
# python_version  :3.6.3

from keras.callbacks import ModelCheckpoint, TensorBoard, Callback


class FloydMetrics(Callback):
    def __init__(self, display):
        Callback.__init__(self)

        self.seen = 0
        self.display = display

    def on_epoch_end(self, epoch, logs=None):
        if self.seen % self.display == 0:
            for k in self.params['metrics']:
                if k in logs:
                    v = logs[k]

                    print('{{"metric": "{}", "value": {}}}'.format(k, v))


def callback(path_checkpoint=None, log_dir=None):
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

    floyd = FloydMetrics(20)
    callbacks.append(floyd)
        
    return callbacks
