#!/usr/bin/env python
# title           :model.py
# description     :Image captioning model.
# author          :Yizhen Chen
# date            :Apr. 24, 2018
# python_version  :3.6.3
# ==============================================================================
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.layers import Input, BatchNormalization, RepeatVector, Embedding, Dense, GRU, LSTM, TimeDistributed, Dropout, \
    concatenate, add, Masking, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K
from keras.regularizers import l1, l2, l1_l2
from keras.optimizers import Adam, RMSprop
from keras.initializers import RandomUniform
from keras.utils import plot_model

import inception_v4


class ImageCaptioningModel(object):
    def __init__(self,
                 max_seq_length,
                 rnn_mode='lstm',
                 drop_rate=0.0,
                 hidden_dim=3,
                 rnn_state_size=512,
                 embedding_size=128,
                 rnn_activation='tanh',
                 cnn_model='inception_v3',
                 mode=1,
                 optimizer=None,
                 initializer=None,
                 learning_rate=None,
                 reg_l1=None,
                 reg_l2=None,
                 num_word=None,
                 is_trainable=False,
                 metrics=None,
                 loss=None):
        self._max_seq_length = max_seq_length
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
        self._mode = mode

        if reg_l1 and reg_l2:
            self._regularizer = l1_l2(reg_l1, reg_l2)
        elif reg_l1:
            self._regularizer = l1(reg_l1)
        elif reg_l2:
            self._regularizer = l2(reg_l2)
        else:
            self._regularizer = None

        # A Tensor with shape (batch_size, height, width, channels)
        self._image_input = None

        # An Tensor with shape (batch_size, seq_length)
        self._text_input = None

        # A Tensor with shape (batch_size, embedding_size)
        self._image_embedding = None

        # A Tensor with shape (batch_size, seq_length, embedding_size)
        self._text_embedding = None

        # A Tensor with shape (batch_size, num_features)
        self._transfer_output = None

        # A Tensor with shape (batch_size, seq_length, embedding_size)
        self._decoder_output = None

        # Output keras model
        self._image_captioning_model = None

    @property
    def image_captioning_model(self):
        return self._image_captioning_model

    def _cell(self):
        if self._rnn_mode == 'gru':
            cell = GRU
        elif self._rnn_mode == 'lstm':
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

    def image_embedding(self):
        if self._cnn_model == 'inception_v3':
            base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
            for layer in base_model.layers:
                layer.trainable = self._is_trainable

            base_model_output = base_model.output

        elif self._cnn_model == 'vgg16':
            base_model = VGG16(weights='imagenet')
            for layer in base_model.layers:
                layer.trainable = self._is_trainable

            transfer_layer = base_model.get_layer('fc2')
            base_model_output = transfer_layer.output

        elif self._cnn_model == 'inception_v4':
            base_model = inception_v4.create_model(weights='imagenet', include_top=False)
            for layer in base_model.layers:
                layer.trainable = self._is_trainable

            base_model_output = GlobalAveragePooling2D()(base_model.output)

        else:
            raise ValueError('Image CNN is not available')

        image_output = RepeatVector(self._max_seq_length)(base_model_output)
        image_output = BatchNormalization()(image_output)
        image_input = base_model.input
        image_embedding = TimeDistributed(Dense(units=self._embedding_size,
                                                kernel_regularizer=self._regularizer,
                                                kernel_initializer=self._initializer,
                                                name='image_embedding'))(image_output)
        image_dropout = Dropout(self._drop_rate, name='image_dropout')(image_embedding)

        self._image_input = image_input
        self._image_embedding = image_dropout
        self._transfer_output = base_model_output

    def text_embedding(self):
        text_input = Input(shape=(self._max_seq_length, self._word_size), name='text_input')
        text_mask = Masking(mask_value=0.0, name='text_mask')(text_input)
        text_embedding = TimeDistributed(Dense(units=self._embedding_size,
                                               kernel_regularizer=self._regularizer,
                                               kernel_initializer=self._initializer,
                                               name='text_embedding'))(text_mask)

        text_dropout = Dropout(self._drop_rate, name='text_dropout')(text_embedding)
        self._text_input = text_input
        self._text_embedding = text_dropout

    def build_decoder_model(self):

        image_embedding = self._image_embedding
        text_embedding = self._text_embedding

        decoder_input = add([image_embedding, text_embedding])

        initial_state = None
        if self._mode == 2:
            initial_state = Dense(self._rnn_state_size, name='image_map')(self._transfer_output)

        input_ = decoder_input

        for _ in range(self._hidden_dim):
            input_ = BatchNormalization()(input_)
            input_ = self._cell()(units=self._rnn_state_size,
                                  return_sequences=True,
                                  recurrent_regularizer=self._regularizer,
                                  kernel_regularizer=self._regularizer,
                                  bias_regularizer=self._regularizer,
                                  kernel_initializer=self._initializer,
                                  implementation=1)(input_, initial_state=initial_state)

        self._decoder_output = TimeDistributed(Dense(self._word_size,
                                                     kernel_regularizer=self._regularizer,
                                                     activation=self._rnn_activation),
                                               name='output')(input_)

    def build_model(self):

        self.image_embedding()
        self.text_embedding()
        self.build_decoder_model()

        inputs = [self._image_input, self._text_input]

        decoder_model = Model(inputs=inputs,
                              outputs=self._decoder_output)
        decoder_model.compile(loss=self._loss,
                              optimizer=self._optimizer(lr=self._eta),
                              metrics=self._metrics)
        self._image_captioning_model = decoder_model


if __name__ == '__main__':
    image_captioning_model = ImageCaptioningModel(30,
                                                  rnn_mode='lstm',
                                                  drop_rate=0.0,
                                                  hidden_dim=3,
                                                  rnn_state_size=256,
                                                  embedding_size=512,
                                                  rnn_activation='softmax',
                                                  cnn_model='inception_v3',
                                                  optimizer=RMSprop,
                                                  initializer='random_uniform',
                                                  learning_rate=0.001,
                                                  mode=1,
                                                  reg_l1=None,
                                                  reg_l2=None,
                                                  num_word=2000,
                                                  is_trainable=False,
                                                  metrics=None,
                                                  loss='categorical_crossentropy')

    image_captioning_model.build_model()
    model = image_captioning_model.image_captioning_model
    plot_model(model, to_file='model.png')
    model.summary()
