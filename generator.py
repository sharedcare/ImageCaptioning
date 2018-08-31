#!/usr/bin/env python
# title           :generator.py
# description     :Generate batch data for fit_generator
# author          :Tiancheng Luo
# date            :Apr. 21, 2018
# python_version  :3.6.3

import json
from preprocessing.image_processing import ImagePreprocessor
from preprocessing.caption_processing import CaptionPreprocessor
from keras.preprocessing.image import list_pictures
from keras.utils import Sequence
import numpy as np
from keras.preprocessing.sequence import pad_sequences


def generator(img_dir, cap_path, batch_size, max_len=30):
    fh = open(cap_path)
    raw_data = fh.read()
    data = json.loads(raw_data)

    img_groups = {}
    all_captions = []
    for img_group in data['images']:
        img_groups[img_group['filename']] = []

        for sentence in img_group['sentences']:
            img_groups[img_group['filename']].append(sentence['raw'])
            all_captions.append(sentence['raw'])

    preprocessor = CaptionPreprocessor(rare_words_handling='nothing')
    preprocessor.fit_on_captions(all_captions)

    image_files = list_pictures(img_dir)

    image_processor = ImagePreprocessor(is_training=True, img_size=(299, 299))

    '''
    global img_array
    if os.path.exists('./img_array.npy'):
        img_array = np.load('./img_array.npy')
    else:
        print("Preprocessing images...\n")
        img_array = image_processor.process_images(image_files)
        print("\nImages preprocessed.")
        np.save('./img_array', img_array)
    '''

    return ImgSequence(img_groups, image_files, img_dir, batch_size, (image_processor, preprocessor), max_len)


class ImgSequence(Sequence):
    def __init__(self, img_groups, image_files, img_dir, batch_size, preprocessor, max_len):
        self.img_groups, self.image_files, self.img_dir, self.max_len = img_groups, image_files, img_dir, max_len
        self.batch_size = batch_size
        self.img_processor, self.caption_processor = preprocessor

    def __len__(self):
        return int(np.ceil(len(self.img_groups) / float(self.batch_size)))

    def __getitem__(self, idx):
        idx = np.random.randint(len(self.img_groups), size=self.batch_size)

        cap_data = []
        img_file_names = []
        # img_input_data = self.img_array[idx]

        for id in idx:
            img_file_name = self.image_files[id].split('/')[-1]
            img_file_names.append(self.img_dir + img_file_name)
            captions = self.img_groups[img_file_name]

            encoded_captions = self.caption_processor.encode_captions(captions)

            i = np.random.choice(len(encoded_captions))

            cap_data.append(encoded_captions[i])

        cap_padded = pad_sequences(cap_data, maxlen=self.max_len+2, padding='post', truncating='post')

        new_cap = list(map(self.caption_processor.tokenizer.sequences_to_matrix, np.expand_dims(cap_padded, -1)))

        cap_final = np.asarray(new_cap)

        cap_input_data = cap_final[:, 0:-1, :]

        output_data = cap_final[:, 1:, :]

        img_input_data = self.img_processor.process_images(img_file_names)

        x_data = \
            {
                'text_input': cap_input_data,
                'input_1': img_input_data
            }

        y_data = \
            {
                'output': output_data
            }

        return x_data, y_data


if __name__ == '__main__':
    sequence = generator('./flickr8k/Flicker8k_Dataset/', './flickr8k/dataset.json', 30)

    for test in sequence:
        x_data, y_data = test

        print(x_data['text_input'].shape)
        break
