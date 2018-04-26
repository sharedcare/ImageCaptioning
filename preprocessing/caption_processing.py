#!/usr/bin/env python
# title           :caption_processor.py
# description     :Preprocessing the captions.
# author          :Tiancheng Luo
# date            :Apr. 24, 2018
# python_version  :3.6.3
# ==============================================================================

import numpy as np
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing import sequence as keras_seq

'''For this preprocessing method

You may use text_to_word_sequence to transfer all caption string to tokens

Resources:
https://github.com/keras-team/keras/blob/master/keras/preprocessing/text.py#L134
https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/22_Image_Captioning.ipynb --> Tokenizer
https://github.com/danieljl/keras-image-captioning/blob/master/keras_image_captioning/preprocessors.py#L49
'''


class CaptionPreprocessor(object):
    EOS_TOKEN = '<eos>'

    def __init__(self, rare_words_handling=None, words_min_occur=None):
        self._tokenizer = Tokenizer()
        self._word_of = {}
        self._rare_words_handling = (rare_words_handling or 'discard')
        self._words_min_occur = (words_min_occur or 5)

    @property
    def vocabs(self):
        word_index = self._tokenizer.word_index
        return sorted(word_index, key=word_index.get)

    def _add_eos(self, captions):
        return map(lambda x: x + ' ' + self.EOS_TOKEN, captions)

    def _caption_lengths(self, captions_output):
        one_hot_sum = captions_output.sum(axis=2)
        return (one_hot_sum != 0).sum(axis=1)

    def _handle_rare_words(self, captions):
        if self._rare_words_handling == 'nothing':
            return captions
        elif self._rare_words_handling == 'discard':
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(captions)
            new_captions = []
            for caption in captions:
                words = text_to_word_sequence(caption)
                new_words = [w for w in words
                             if tokenizer.word_counts.get(w, 0) >=
                             self._words_min_occur]
                new_captions.append(' '.join(new_words))
            return new_captions

        raise NotImplementedError('rare_words_handling={} is not implemented '
                                  'yet!'.format(self._rare_words_handling))

    def fit_on_captions(self, captions_txt):
        captions_txt = self._handle_rare_words(captions_txt)
        captions_txt = self._add_eos(captions_txt)
        self._tokenizer.fit_on_texts(captions_txt)
        self._word_of = {i: w for w, i in self._tokenizer.word_index.items()}

    def encode_captions(self, captions_txt):
        captions_txt = self._add_eos(captions_txt)
        return self._tokenizer.texts_to_sequences(captions_txt)

    def decode_captions(self, captions_output, captions_output_expected=None):
        captions = captions_output[:, :-1, :]  # Discard the last word (dummy)
        label_encoded = captions.argmax(axis=-1)
        num_batches, num_words = label_encoded.shape

        if captions_output_expected is not None:
            caption_lengths = self._caption_lengths(captions_output_expected)
        else:
            caption_lengths = [num_words] * num_batches

        captions_str = []
        for caption_i in range(num_batches):
            caption_str = []
            for word_i in range(caption_lengths[caption_i]):
                label = label_encoded[caption_i, word_i]
                label += 1  # Real label = label in model + 1
                caption_str.append(self._word_of[label])
            captions_str.append(' '.join(caption_str))

        return captions_str

    def preprocess_batch(self, captions_label_encoded):
        captions = keras_seq.pad_sequences(captions_label_encoded,
                                           padding='post')
        # Because the number of timesteps/words resulted by the model is
        # maxlen(captions) + 1 (because the first "word" is the image).
        captions_extended1 = keras_seq.pad_sequences(captions,
                                                     maxlen=captions.shape[-1] + 1,
                                                     padding='post')
        captions_one_hot = list(map(self._tokenizer.sequences_to_matrix,
                               np.expand_dims(captions_extended1, -1)))
        captions_one_hot = np.array(captions_one_hot, dtype='int')

        # Decrease/shift word index by 1.
        # Shifting `captions_one_hot` makes the padding word
        # (index=0, encoded=[1, 0, ...]) encoded all zeros ([0, 0, ...]),
        # so its cross entropy loss will be zero.
        captions_decreased = captions.copy()
        captions_decreased[captions_decreased > 0] -= 1
        captions_one_hot_shifted = captions_one_hot[:, :, 1:]

        captions_input = captions_decreased
        captions_output = captions_one_hot_shifted
        return captions_input, captions_output


if __name__ == '__main__':
    preprocessor = CaptionPreprocessor(rare_words_handling='nothing')

    captions = ['Closeup of bins of food that include broccoli and bread.',
                'A meal is presented in brightly colored plastic trays.',
                'there are containers filled with different kinds of foods',
                'Colorful dishes holding meat, vegetables, fruit, and bread.',
                'A bunch of trays that have different food.']

    preprocessor.fit_on_captions(captions)

    encoded_captions = preprocessor.encode_captions(captions)

    batch = preprocessor.preprocess_batch(encoded_captions)

    vocabs = preprocessor.vocabs

