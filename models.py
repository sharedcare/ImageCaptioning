#!/usr/bin/env python
# title           :model.py
# description     :Image captioning model.
# author          :Yizhen Chen
# date            :Apr. 24, 2018
# python_version  :3.6.3
# ==============================================================================
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, BatchNormalization, RepeatVector, Embedding, Dense, GRU, LSTM, concatenate
from keras.models import Model
from keras import backend as K
from keras.regularizers import l1, l2, l1_l2
from keras.optimizers import Adam, RMSprop
from keras.initializers import RandomUniform

import tensorflow as tf

from losses import sparse_cross_entropy as loss

class ImageCaptioningModel(object):

    def __init__(self,
                 rnn_mode='lstm',
                 drop_rate=0.0,
                 hidden_dim=3,
                 rnn_state_size=512,
                 embedding_size=128,
                 rnn_activation='tanh',
                 cnn_model=InceptionV3,
                 optimizer=None,
                 initializer=None,
                 learning_rate=None,
                 reg_l1=None,
                 reg_l2=None,
                 num_word=None,
                 is_trainable=False,
                 metrics=None,
                 loss=None):

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
        self._is_trainable = is_trainable
        self._initializer = initializer
        self._metrics = metrics
        self._loss = loss

        if reg_l1 and reg_l2:
            self._regularizer = l1_l2(reg_l1, reg_l2)
        elif reg_l1:
            self._regularizer = l1(reg_l1)
        elif reg_l2:
            self._regularizer = l2(reg_l2)
        else:
            self._regularizer = None

        # A Tensor with shape (batch_size, height, width, channels)
        self.images = None

        # An Tensor with shape (batch_size, seq_length)
        self._seq_input = None

        # A Tensor with shape (batch_size, embedding_size)
        self._image_embedding = None

        # A Tensor with shape (batch_size, seq_length, embedding_size)
        self._seq_embedding = None

        # Output keras model
        self._image_captioning_model = None

    @property
    def image_captioning_model(self):
        return self._image_captioning_model

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

    def _build_image_embedding(self):
        model_no_top = self._cnn_model(weights='imagenet', include_top=False, pooling='avg')
        for layer in model_no_top.layers:
            layer.trainable = self._is_trainable

        image_model_out = model_no_top.output
        x = BatchNormalization(axis=-1)(image_model_out)
        image_embedding = Dense(self._embedding_size,
                  kernel_regularizer=self._regularizer,
                  kernel_initializer='random_uniform')(x)

        image_input = model_no_top.input

        self._image_embedding = image_embedding
        self._image_input = image_input

        return image_input, image_embedding

    def _build_seq_embedding(self):
        decoder_input = Input(shape=(None, ), name='decoder_input')
        decoder_embedding = Embedding(input_dim=self._word_size,
                                      output_dim=self._embedding_size,
                                      embeddings_regularizer=self._regularizer,
                                      name='decoder_embedding')(decoder_input)
        self._seq_embedding = decoder_embedding
        self._seq_input = decoder_input
        return decoder_input, decoder_embedding

    def _build_decoder_model(self, decoder_input):

        input_ = decoder_input

        for _ in range(self._hidden_dim):
            input_ = LSTM(units=self._rnn_state_size,
                                    dropout=self._drop_rate,
                                    recurrent_dropout=self._drop_rate,
                                    return_sequences=True,
                                    kernel_regularizer=self._regularizer,
                                    kernel_initializer=self._initializer,
                                    implementation=2)(input_)

        return Dense(self._word_size,
                              activation=self._rnn_activation,
                              name='decoder_output')(input_)

    def build_model(self):
        '''
        image_output = self._cell()(units=self._rnn_state_size,
                                    dropout=self._drop_rate,
                                    recurrent_dropout=self._drop_rate,
                                    return_sequences=True,
                                    kernel_regularizer=self._regularizer,
                                    kernel_initializer=self._initializer,
                                    implementation=2)(self._image_embedding)
        '''
        image_output = RepeatVector(1)(self._image_embedding)
        decoder_input = concatenate([image_output, self._seq_embedding], axis=1)
        decoder_output = self._build_decoder_model(decoder_input)

        # Model(inputs=image_input,
        #      outputs=image_embedding)
        # decoder_target = tf.placeholder(dtype='int32', shape=(None, None))
        decoder_model = Model(inputs=[self._image_input, self._seq_input],
                              outputs=decoder_output)
        decoder_model.compile(loss=self._loss,
                              optimizer=self._optimizer(lr=self._eta),
                              metrics=self._metrics)
        self._image_captioning_model = decoder_model

    def build(self):
        self._build_image_embedding()
        self._build_seq_embedding()
        self.build_model()


if __name__ == '__main__':
    image_captioning_model = ImageCaptioningModel(rnn_mode='lstm',
                                                 drop_rate=0.0,
                                                 hidden_dim=3,
                                                 rnn_state_size=256,
                                                 embedding_size=512,
                                                 rnn_activation='tanh',
                                                 cnn_model=InceptionV3,
                                                 optimizer=RMSprop,
                                                 initializer='random_uniform',
                                                 learning_rate=0.001,
                                                 reg_l1=None,
                                                 reg_l2=None,
                                                 num_word=10000,
                                                 is_trainable=False,
                                                 metrics=None,
                                                 loss='categorical_crossentropy')

    image_captioning_model.build()
    decoder_model = image_captioning_model.image_captioning_model
    decoder_model.summary()
