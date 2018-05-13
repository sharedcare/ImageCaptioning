#!/usr/bin/env python
# title           :train.py
# description     :Image captioning model training steps.
# author          :Yizhen Chen
# date            :Apr. 24, 2018
# python_version  :3.6.3
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt

from keras.applications.inception_v3 import InceptionV3
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

    batch_size = 50

    num_seq_per_image = 5

    total_seq = num_image * num_seq_per_image

    steps_per_epoch = total_seq // batch_size

    generator_func = generator(image_data_path, caption_path, batch_size)

    # save_path = 'model.h5'
    save_path = False

    ckpt_path = 'checkpoint.h5'

    if save_path and os.path.isfile(save_path):
        print("Load Saved Model")
        model = load_model(save_path)
    else:
        image_captioning_model = ImageCaptioningModel(31,
                                                      rnn_mode='lstm',
                                                      drop_rate=0.5,
                                                      hidden_dim=3,
                                                      rnn_state_size=128,
                                                      embedding_size=128,
                                                      rnn_activation='softmax',
                                                      cnn_model='inception_v3',
                                                      optimizer=Adam,
                                                      initializer='random_uniform',
                                                      learning_rate=1e-3,
                                                      mode=1,
                                                      reg_l1=1e-7,
                                                      reg_l2=1e-7,
                                                      num_word=len(generator_func.preprocessor.word_index) + 1,
                                                      is_trainable=False,
                                                      metrics=['accuracy'],
                                                      loss='categorical_crossentropy')

        image_captioning_model.build_model()
        model = image_captioning_model.image_captioning_model

    model.fit_generator(generator=generator_func,
                        steps_per_epoch=steps_per_epoch,
                        epochs=50,
                        verbose=2,
                        callbacks=callback('checkpoint.h5', './logs/'))

    model.save(save_path)


def predict(filename):
    model_path = 'model.h5'
    model = load_model(model_path)
    image_data_path = './flickr8k/Flicker8k_Dataset/'
    caption_path = './flickr8k/dataset.json'
    batch_size = 50

    generator_func = generator(image_data_path, caption_path, batch_size)

    preprocessor = ImagePreprocessor(is_training=False)
    if type(filename) == str:
        image = preprocessor.process_image(filename)
        image_batch = np.expand_dims(image, axis=0)
    elif type(filename) == list:
        image_batch = preprocessor.process_batch(filename)
    else:
        raise ValueError('Input image name is not vaild')

    text_input_shape = (1, 31, len(generator_func.preprocessor.word_index) + 1)
    text_input = np.zeros(shape=text_input_shape)
    start_token_id = generator_func.preprocessor.word_index['sos']
    end_token_id = generator_func.preprocessor.word_index['eos']
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
            print('eos')
            break
        text_input[0, seq_index + 1, word_id] = 1
        word = generator_func.preprocessor.vocabs[word_id-1]
        print(word)
        output.append(word)

    plt.imshow(plt.imread(filename))
    plt.show()


if __name__ == '__main__':
    # predict('./download.jpg')
    run()
