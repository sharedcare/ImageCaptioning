#!/usr/bin/env python
# title           :model.py
# description     :Image captioning model.
# author          :Yizhen Chen
# date            :Apr. 24, 2018
# python_version  :3.6.3
# ==============================================================================
from keras.applications.vgg16 import VGG16
from keras.layers import Input, BatchNormalization, RepeatVector, Embedding, Dense, GRU, LSTM, concatenate
from keras.models import Model
from keras import backend as K
from keras.preprocessing.image import load_img, list_pictures
from keras.regularizers import l1, l2, l1_l2
from keras.optimizers import Adam, RMSprop
from keras.initializers import RandomUniform

import tensorflow as tf
import sys
import os
import numpy as np

from losses import sparse_cross_entropy as loss

def image_caption_model(word_size):
    image_model = VGG16(weights='imagenet', include_top=True)
    transfer_layer = image_model.get_layer('fc2')
    image_model_transfer = Model(inputs=image_model.input,
                             outputs=transfer_layer.output)
    for layer in image_model_transfer.layers:
        layer.trainable = False
    '''
    img_size = K.int_shape(image_model.input)[1:3]
    print(img_size)
    transfer_values_size = K.int_shape(transfer_layer.output)[1]
    print(transfer_values_size)
    '''

    decoder_transfer_map = Dense(512,
                                 activation='tanh',
                                 name='decoder_transfer_map')
    decoder_input = Input(shape=(None,), name='decoder_input')
    decoder_embedding = Embedding(input_dim=word_size,
                                  output_dim=128,
                                  name='decoder_embedding')
    decoder_gru1 = GRU(512, name='decoder_gru1',
                       return_sequences=True)
    decoder_gru2 = GRU(512, name='decoder_gru2',
                       return_sequences=True)
    decoder_gru3 = GRU(512, name='decoder_gru3',
                       return_sequences=True)
    decoder_dense = Dense(word_size,
                          activation='linear',
                          name='decoder_output')

    initial_state = decoder_transfer_map(image_model_transfer.output)

    # Start the decoder-network with its input-layer.
    net = decoder_input

    # Connect the embedding-layer.
    net = decoder_embedding(net)

    # Connect all the GRU layers.
    net = decoder_gru1(net, initial_state=initial_state)
    net = decoder_gru2(net, initial_state=initial_state)
    net = decoder_gru3(net, initial_state=initial_state)

    # Connect the final dense layer that converts to
    # one-hot encoded arrays.
    decoder_output = decoder_dense(net)

    decoder_model = Model(inputs=[image_model_transfer.input, decoder_input],
                          outputs=[decoder_output])

    return decoder_model

if __name__ == '__main__':
    model = image_caption_model(8000)
    model.summary()