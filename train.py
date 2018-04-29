#!/usr/bin/env python
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import RMSprop
from keras.metrics import mae, categorical_accuracy
from keras.losses import categorical_crossentropy
from keras.preprocessing.image import list_pictures
from keras.callbacks import ModelCheckpoint

from models import ImageCaptioningModel
from callbacks import callback
from generator import generator


config = {
    'learning_rate': None,                  # Learning rate for model optimizer
    'rnn_mode': 'lstm',                     # RNN cell type, it can be either LSTM or GRU ---> https://keras.io/layers/recurrent/
    'drop_rate': 0.0,                       # Dropout rate for RNN
    'hidden_dim': 3,                        # Hidden layer size for RNN
    'rnn_state_size': 512,                  # RNN state size, indicates the size of outputs from RNN cell
    'embedding_size': 128,                  # Embedding layer output size ---> See what is embedding layer: https://keras.io/layers/embeddings/
    'rnn_activation': 'tanh',               # RNN activation function, e.g. tanh, relu, linear, softmax ---> https://keras.io/activations/
    'cnn_model': InceptionV3,               # CNN image classification model, can be either Inception V3 or Inception V4
    'optimizer': RMSprop,                   # Optimizer for reducing error. ---> https://keras.io/optimizers/
    'reg_l1': None,                         # Regularizer l1    ---> https://keras.io/regularizers/
    'reg_l2': None,                         # Regularizer l2
    'num_word': None,                       # Vocabulary size
    'is_trainable': False,                  # Indicates whether CNN is trainable
    'metrics': [mae, categorical_accuracy], # Used to judge the performance of the model. ---> https://keras.io/metrics/
    'loss': categorical_crossentropy        # Loss function ---> https://keras.io/losses/
}


def run():

    image_data_path = './flickr8k/Flicker8k_Dataset/'

    caption_path = './flickr8k/dataset.json'

    images = list_pictures(image_data_path)

    num_image = len(images)

    batch_size = 35

    num_seq_per_image = 5

    total_seq = num_image * num_seq_per_image

    steps_per_epoch = total_seq // batch_size

    image_captioning_model = ImageCaptioningModel(rnn_mode='lstm',
                                                 drop_rate=0.1,
                                                 hidden_dim=3,
                                                 rnn_state_size=256,
                                                 embedding_size=512,
                                                 rnn_activation='tanh',
                                                 cnn_model=InceptionV3,
                                                 optimizer=RMSprop,
                                                 initializer='random_uniform',
                                                 learning_rate=0.001,
                                                 reg_l1=0.001,
                                                 reg_l2=0.001,
                                                 num_word=8387,
                                                 is_trainable=False,
                                                 metrics=None,
                                                 loss='categorical_crossentropy')

    save_path = 'model.h5'


    ckpt_path = None

    image_captioning_model.build()
    decoder_model = image_captioning_model.image_captioning_model
    decoder_model.save(save_path)

    if ckpt_path:
        decoder_model.load_weights(ckpt_path)

    decoder_model.fit_generator(generator=generator(image_data_path, caption_path, batch_size),
                                steps_per_epoch=steps_per_epoch,
                                epochs=20,
                                callbacks=callback('checkpoint.keras', './logs/'))


if __name__ == '__main__':
    run()

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
