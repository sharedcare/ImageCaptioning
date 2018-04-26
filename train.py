#!/usr/bin/env python
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import RMSprop

from .model import ImageCaptioningModel


config = {
    'learning_rate': None,
    'vocab_size': None,
    'rnn_mode': 'lstm',
    'drop_rate': 0.0,
    'hidden_dim': 3,
    'rnn_state_size': 512,
    'embedding_size': 128,
    'rnn_activation': 'tanh',
    'cnn_model': InceptionV3,
    'optimizer': RMSprop,
    'reg_l1': None,
    'reg_l2': None,
    'num_word': None
}

def run():

    image_captioning_model = ImageCaptioningModel(learning_rate=None,
                                                  vocab_size=None,
                                                  rnn_mode='lstm',
                                                  drop_rate=0.0,
                                                  hidden_dim=3,
                                                  rnn_state_size=512,
                                                  embedding_size=128,
                                                  rnn_activation='tanh',
                                                  cnn_model=InceptionV3,
                                                  optimizer=RMSprop,
                                                  reg_l1=None,
                                                  reg_l2=None,
                                                  num_word=None)

    path_checkpoint = './'

    decoder_model = image_captioning_model.build_model()
    try:
        decoder_model.load_weights(path_checkpoint)
    except Exception as error:
        print("Error trying to load checkpoint.")
        print(error)

    decoder_model.fit_generator(generator=generator,
                                steps_per_epoch=steps_per_epoch,
                                epochs=20,
                                callbacks=callbacks)
'''
https://keras.io/models/model/#fit_generator

fit_generator(self, generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)

Trains the model on data generated batch-by-batch by a Python generator or an instance of Sequence.

The generator is run in parallel to the model, for efficiency. For instance, this allows you to do real-time data augmentation on images on CPU in parallel to training your model on GPU.

The use of keras.utils.Sequence guarantees the ordering and guarantees the single use of every input per epoch when using use_multiprocessing=True.

e.g.

def generate_arrays_from_file(path):
    while True:
        with open(path) as f:
            for line in f:
                # create numpy arrays of input data
                # and labels, from each line in the file
                x1, x2, y = process_line(line)
                yield ({'input_1': x1, 'input_2': x2}, {'output': y})

model.fit_generator(generate_arrays_from_file('/my_file.txt'),
                    steps_per_epoch=10000, epochs=10)
'''
