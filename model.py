#!/usr/bin/env python
# title           :model.py
# description     :Image captioning model.
# author          :Yizhen Chen
# date            :Apr. 24, 2018
# python_version  :3.6.3
# ==============================================================================
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, BatchNormalization, Embedding, Dense, GRU, LSTM
from keras.models import Model
from keras import backend as K
from keras.regularizers import l1, l2, l1_l2
from keras.optimizers import Adam, RMSprop
import tensorflow as tf

from .losses import sparse_cross_entropy as loss

class ImageCaptioningModel(object):

    def __init__(self,
                 learning_rate=None,
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
                 num_word=None):
        self._rnn_mode = rnn_mode
        self._drop_rate = drop_rate
        self._rnn_state_size = rnn_state_size
        self._embedding_size = embedding_size
        self._rnn_activation = rnn_activation
        self._word_size = num_word
        self._hidden_dim = hidden_dim
        self._cnn_model = cnn_model
        self._eta = learning_rate
        self._optimizer = optimizer
        if reg_l1 and reg_l2:
            self._regularizer = l1_l2(reg_l1, reg_l2)
        elif reg_l1:
            self._regularizer = l1(reg_l1)
        elif reg_l2:
            self._regularizer = l2(reg_l2)
        else:
            self._regularizer = None

    def _cell(self):
        if self._rnn_mode == 'gru' or 'GRU':
            cell = GRU
        elif self._rnn_mode == 'lstm' or 'LSTM':
            cell = LSTM
        else:
            raise ValueError('rnn_mode must be lstm or gru')

        return cell

    def _activation_fn(self):
        activation_fns = ['softmax',
                          'elu',
                          'selu',
                          'softplus',
                          'softsign',
                          'relu',
                          'tanh',
                          'sigmoid',
                          'hard_sigmoid',
                          'linear']
        if self._rnn_activation not in activation_fns:
            raise ValueError('rnn_activation must be a valid activation function')
        else:
            return self._rnn_activation

    def build_model(self):
        image_input, image_embedding = self._image_embedding()
        decoder_input, decoder_embedding = self._word_embedding()
        Model(inputs=image_input,
              outputs=image_embedding)
        decoder_output = self._decoder_model(decoder_input)
        # decoder_target = tf.placeholder(dtype='int32', shape=(None, None))
        decoder_model = Model(inputs=[image_input, decoder_input],
                              outputs=decoder_output)
        decoder_model.compile(optimizer=self._optimizer(lr=self._eta),
                              loss='categorical_crossentropy')
        return decoder_model

    def _image_embedding(self):
        model_no_top = self._cnn_model(weights='imagenet', include_top=False, pooling='avg')
        for layer in model_no_top.layers:
            layer.trainable = False

        x = model_no_top.output
        x = BatchNormalization()(x)
        x = Dense(self._embedding_size,
                  kernel_regularizer=self._regularizer)(x)

        image_input = model_no_top.input

        return image_input, x

    def _word_embedding(self):
        decoder_input = Input(shape=(None, ), name='decoder_input')
        decoder_embedding = Embedding(input_dim=self._word_size,
                                      output_dim=self._embedding_size,
                                      embeddings_regularizer=self._regularizer,
                                      name='decoder_embedding')(decoder_input)
        return decoder_input, decoder_embedding

    def _decoder_model(self, decoder_input):
        hidden_layer = self._cell()(units=self._rnn_state_size,
                                    dropout=self._drop_rate,
                                    recurrent_dropout=self._drop_rate,
                                    return_sequences=True,
                                    recurrent_regularizer=self._regularizer,
                                    implementation=2)

        decoder_dense = Dense(self._word_size,
                              activation=self._rnn_activation,
                              name='decoder_output')

        input_ = decoder_input
        for _ in range(self._hidden_dim):
            input_ = hidden_layer(input_)

        return decoder_dense(input_)
