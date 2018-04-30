import json
from preprocessing.image_processing import ImagePreprocessor
from preprocessing.caption_processing import CaptionPreprocessor
from keras.preprocessing.image import list_pictures
from keras.utils import Sequence
import numpy as np
import os
from pprint import pformat
from keras.preprocessing.sequence import pad_sequences


def generator(img_dir, cap_path, batch_size):
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

    global img_array
    if os.path.exists('./img_array.npy'):
        img_array = np.load('./img_array.npy')
    else:
        print("Preprocessing images...\n")
        img_array = image_processor.process_images(image_files)
        print("\nImages preprocessed.")
        np.save('./img_array', img_array)

    return ImgSequence(img_groups, image_files, img_array, batch_size, preprocessor)


class ImgSequence(Sequence):
    def __init__(self, img_groups, image_files, img_array, batch_size, preprocessor):
        self.img_groups, self.image_files, self.img_array = img_groups, image_files, img_array
        self.batch_size = batch_size
        self.preprocessor = preprocessor

    def __len__(self):
        return int(np.ceil(len(self.img_groups) / float(self.batch_size)))

    def __getitem__(self, idx):
        idx = np.random.randint(len(self.img_groups), size=self.batch_size)

        cap_input_data = []
        img_input_data = self.img_array[idx]
        output_data = []

        for id in idx:
            img_file_name = self.image_files[id].split('/')[-1]
            captions = self.img_groups[img_file_name]

            encoded_captions = self.preprocessor.encode_captions(captions)
            captions_input, captions_output = self.preprocessor.preprocess_batch(encoded_captions)

            i = np.random.choice(len(captions_input))

            cap_input_data.append(captions_input[i])
            output_data.append(captions_output[i])

        len_cap = [len(t) for t in cap_input_data]
        max_len = np.max(len_cap)

        cap_padded = pad_sequences(cap_input_data, maxlen=max_len, padding='post', truncating='post')

        cap_input_data = cap_padded[:, 0:-1]
        # output_data = cap_padded[:, 1:]

        len_cap = [len(t) for t in output_data]
        max_len = np.max(len_cap)

        cap_padded = pad_sequences(output_data, maxlen=max_len, padding='post', truncating='post')

        output_data = cap_padded[:, 1:]

        x_data = \
            {
                'decoder_input': np.array(cap_input_data),
                'input_1': img_input_data
            }

        y_data = \
            {
                'decoder_output': np.array(output_data).reshape(self.batch_size, len(output_data[0]), len(output_data[0][0]))
            }

        return x_data, y_data


if __name__ == '__main__':
    sequence = generator('./flickr8k/Flicker8k_Dataset/', './flickr8k/dataset.json', 1)

    for test in sequence:
        x_data, y_data = test

        output = []
        decoder_input = x_data['decoder_input']
        print(decoder_input.shape)
        print(decoder_input)
        for i in range(len(y_data['decoder_output'][0])):
            token = np.argmax(y_data['decoder_output'][0][i])
            output.append(token)
        print(output)
        break
