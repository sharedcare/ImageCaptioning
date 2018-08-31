#!/usr/bin/env python
# title           :train.py
# description     :Image captioning model train and predict steps.
# author          :Yizhen Chen
# date            :Apr. 24, 2018
# python_version  :3.6.3
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.optimizers import RMSprop, Adam
from keras.metrics import mae, categorical_accuracy
from keras.losses import categorical_crossentropy
from keras.preprocessing.image import list_pictures

from models import ImageCaptioningModel
from preprocessing.image_processing import ImagePreprocessor
from callbacks import callback
from generator import generator

import os

CONFIG = {
    'epoch': 50,                            # Epoch, 50 is sufficient
    'batch_size': 35,                       # Batch size
    'num_seq_per_image': 5,                 # Number of sequences for each image
    'max_seq_len': 30,                      # Maximum sequence length excluding <sos> and <eos>
    'learning_rate': 1e-3,                  # Learning rate for model optimizer
    'rnn_mode': 'lstm',                     # RNN cell type, it can be either lstm or gru ---> https://keras.io/layers/recurrent/
    'drop_rate': 0.5,                       # Dropout rate for RNN
    'hidden_dim': 3,                        # Hidden layer size for RNN
    'rnn_state_size': 128,                  # RNN state size, indicates the size of outputs from RNN cell
    'embedding_size': 128,                  # Embedding layer output size ---> See what is embedding layer: https://keras.io/layers/embeddings/
    'activation': 'softmax',                # RNN activation function, e.g. tanh, relu, linear, softmax ---> https://keras.io/activations/
    'cnn_model': 'inception_v4',            # CNN image classification model, can be inception_v3, inception_v4 or vgg16
    'initializer': 'random_uniform',        # Initialization for each tensor ---> https://keras.io/initializers/
    'optimizer': Adam,                      # Optimizer for reducing error. ---> https://keras.io/optimizers/
    'reg_l1': 1e-7,                         # Regularizer l1    ---> https://keras.io/regularizers/
    'reg_l2': 1e-7,                         # Regularizer l2
    'cnn_is_trainable': False,              # Indicates whether CNN is trainable
    'metrics': ['accuracy'],                # Used to judge the performance of the model. ---> https://keras.io/metrics/
    'loss': 'categorical_crossentropy',     # Loss function ---> https://keras.io/losses/
    'mode': 1
}


class Run(object):

    def __init__(self,
                 image_path=None,
                 caption_path=None,
                 config=None,
                 model_path=None,
                 ckpt_path=None):
        self._image_path = image_path
        self._caption_path = caption_path
        self._model_path = model_path
        self._ckpt_path = ckpt_path
        self._batch_size = config['batch_size']
        self._num_seq_per_image = config['num_seq_per_image']
        self._epoch = config['epoch']

        # Model configuration
        self._max_seq_length = config['max_seq_len'] + 1
        self._rnn_mode = config['rnn_mode']
        self._drop_rate = config['drop_rate']
        self._hidden_dim = config['hidden_dim']
        self._rnn_state_size = config['rnn_state_size']
        self._embedding_size = config['embedding_size']
        self._activation = config['activation']
        self._cnn_model = config['cnn_model']
        self._optimizer = config['optimizer']
        self._initializer = config['initializer']
        self._lr = config['learning_rate']
        self._mode = config['mode']
        self._reg_l1 = config['reg_l1']
        self._reg_l2 = config['reg_l2']
        self._is_trainable = config['cnn_is_trainable']
        self._metrics = config['metrics']
        self._loss = config['loss']

        self._generator = None
        self._image_captioning_model = None

        self._model = None

    def _build_generator(self):
        generator_func = generator(self._image_path, self._caption_path, self._batch_size)
        self._generator = generator_func

    def train(self):

        images = list_pictures(self._image_path)

        num_image = len(images)

        total_seq = num_image * self._num_seq_per_image

        steps_per_epoch = total_seq // self._batch_size

        self._build_generator()

        image_captioning_model = ImageCaptioningModel(self._max_seq_length,
                                                      rnn_mode=self._rnn_mode,
                                                      drop_rate=self._drop_rate,
                                                      hidden_dim=self._hidden_dim,
                                                      rnn_state_size=self._rnn_state_size,
                                                      embedding_size=self._embedding_size,
                                                      rnn_activation=self._activation,
                                                      cnn_model=self._cnn_model,
                                                      optimizer=self._optimizer,
                                                      initializer=self._initializer,
                                                      learning_rate=self._lr,
                                                      mode=self._mode,
                                                      reg_l1=self._reg_l1,
                                                      reg_l2=self._reg_l2,
                                                      num_word=len(self._generator.caption_processor.word_index) + 1,
                                                      is_trainable=self._is_trainable,
                                                      metrics=self._metrics,
                                                      loss=self._loss)

        image_captioning_model.build_model()
        model = image_captioning_model.image_captioning_model

        save_path = self._model_path

        ckpt_path = self._ckpt_path

        if ckpt_path and os.path.isfile(ckpt_path):
            print("Load Check Point")
            model.load_weights(ckpt_path)

        self._image_captioning_model = model

        model.fit_generator(generator=self._generator,
                            steps_per_epoch=steps_per_epoch,
                            epochs=self._epoch,
                            callbacks=callback(ckpt_path, './logs/'))

        model.save(save_path)

    def predict(self, filename):
        if not self._model:
            model = load_model(self._model_path)
            self._model = model
        else:
            model = self._model

        self._build_generator()

        preprocessor = ImagePreprocessor(is_training=False)
        if type(filename) == str:
            image = preprocessor.process_image(filename)
            image_batch = np.expand_dims(image, axis=0)
        elif type(filename) == list:
            image_batch = preprocessor.process_batch(filename)
        else:
            raise ValueError('Input image name is not vaild')

        text_input_shape = (1, self._max_seq_length, len(self._generator.preprocessor.word_index) + 1)
        text_input = np.zeros(shape=text_input_shape)
        start_token_id = self._generator.preprocessor.word_index['sos']
        end_token_id = self._generator.preprocessor.word_index['eos']
        text_input[0, 0, start_token_id] = 1

        output = []

        for seq_index in range(31):
            input_data = \
                {
                    'input_1': image_batch,
                    'text_input': text_input
                }
            predictions = model.predict(input_data) # Predict next word
            word_id = np.argmax(predictions[0, seq_index, :])
            if word_id == end_token_id:
                break
            text_input[0, seq_index + 1, word_id] = 1
            word = self._generator.preprocessor.vocabs[word_id-1]
            output.append(word)

        print(' '.join(output) + '.')
        plt.imshow(plt.imread(filename))
        plt.show()


if __name__ == '__main__':
    run = Run(image_path='./flickr8k/Flicker8k_Dataset/',
              caption_path='./flickr8k/dataset.json',
              config=CONFIG,
              model_path='model.h5',
              ckpt_path='checkpoint.h5')
    run.train()

    for image in os.listdir('./tests'):
        file = './tests/' + image

        if image == '.DS_Store':
            continue

        print(file)

        run.predict(file)
