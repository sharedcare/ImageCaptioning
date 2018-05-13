#!/usr/bin/env python
# title           :train.py
# description     :Image captioning model training steps.
# author          :Yizhen Chen
# date            :Apr. 24, 2018
# python_version  :3.6.3
# ==============================================================================
import tensorflow as tf
import numpy as np

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

    save_path = 'model.h5'

    ckpt_path = 'checkpoint.h5'

    if ckpt_path:
        model.load_weights(ckpt_path)

    model.fit_generator(generator=generator_func,
                        steps_per_epoch=steps_per_epoch,
                        epochs=5,
                        callbacks=callback('checkpoint.h5', './logs/'))

    model.save(save_path)


def predict(filename):
    model_path = 'model.h5'

    decoder_model = load_model(model_path)

    preprocessor = ImagePreprocessor(is_training=False)
    if type(filename) == str:
        image = preprocessor.process_image(filename)
        image_batch = np.expand_dims(image, axis=0)
    elif type(filename) == list:
        image_batch = preprocessor.process_batch(filename)
    else:
        raise ValueError('Input image name is not vaild')

    decoder_input_shape = (1, 31)
    decoder_input = np.zeros(shape=decoder_input_shape, dtype=np.int)

    token_int = 2

    output = []
    '''
    x_data = \
        {
            'input_1': image_batch,
            'decoder_input': decoder_input
        }
    decoder_output = decoder_model.predict(x_data)
    
    for i in range(len(decoder_output[0])):
        token = np.argmax(decoder_output[0][i])
        output.append(token)
    print(output)
    '''

    count_tokens = 0
    while token_int != 3 and count_tokens < seq_length:
        # Update the input-sequence to the decoder
        # with the last token that was sampled.
        # In the first iteration this will set the
        # first element to the start-token.
        # decoder_input[0, count_tokens] = token_int
        decoder_input[0, count_tokens] = token_int

        print(decoder_input)

        # Wrap the input-data in a dict for clarity and safety,
        # so we are sure we input the data in the right order.
        x_data = \
            {
                'input_1': image_batch,
                'decoder_input': decoder_input
            }

        # Note that we input the entire sequence of tokens
        # to the decoder. This wastes a lot of computation
        # because we are only interested in the last input
        # and output. We could modify the code to return
        # the GRU-states when calling predict() and then
        # feeding these GRU-states as well the next time
        # we call predict(), but it would make the code
        # much more complicated.

        # Input this data to the decoder and get the predicted output.
        decoder_output = decoder_model.predict(x_data)

        #print(decoder_output)
        # Get the last predicted token as a one-hot encoded array.
        # Note that this is not limited by softmax, but we just
        # need the index of the largest element so it doesn't matter.
        token_onehot = decoder_output[0, count_tokens, 1:]

        # Convert to an integer-token.
        token_int = np.argmax(token_onehot)
        output.append(token_int)

        # Lookup the word corresponding to this integer-token.
        # sampled_word = tokenizer.token_to_word(token_int)

        # Append the word to the output-text.
        # output_text += " " + sampled_word

        # Increment the token-counter.
        count_tokens += 1

    # print(image_batch.shape)
    print(output)



if __name__ == '__main__':
    # predict(['./flickr8k/Flicker8k_Dataset/3452411712_5b42d2a1b5.jpg'])
    run()
