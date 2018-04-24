#!/usr/bin/env python
#title           :caption_processor.py
#description     :Preprocessing the captions.
#author          :Yizhen Chen
#date            :Apr. 24, 2018
#python_version  :3.6.3
#==============================================================================
import numpy as np
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
'''For this preprocessing method

You may use text_to_word_sequence to transfer all caption string to tokens

Resources:
https://github.com/keras-team/keras/blob/master/keras/preprocessing/text.py#L134
https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/22_Image_Captioning.ipynb --> Tokenizer
https://github.com/danieljl/keras-image-captioning/blob/master/keras_image_captioning/preprocessors.py#L49
'''