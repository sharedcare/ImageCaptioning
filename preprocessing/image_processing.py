#!/usr/bin/env python
#title           :image_processor.py
#description     :Preprocessing the images.
#author          :Yizhen Chen
#date            :Apr. 24, 2018
#python_version  :3.6.3
#==============================================================================
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.inception_v3 import preprocess_input
from tqdm import tqdm
from keras.preprocessing.image import list_pictures

class ImagePreprocessor(object):

    def __init__(self, is_training, img_size=(299, 299)):
        self._img_size = img_size
        self._is_training = is_training

    def _process_image(self, img_dir):
        '''Load and resize the image.

        This returns the image as a numpy-array.

        Args:
            img_dir: the directory of the image file.
            _image_size: the target size of image

        Returns:
            A preprocessed numpy array.
        '''
        img = load_img(img_dir, target_size=self._img_size)
        img_array = img_to_array(img)
        #if self._is_training:
        #    img_array = distort_img(img_array)  # Add distort_img method
        return preprocess_input(img_array)

    def process_batch(self, images):
        batch_size = len(images)
        batch_shape = (batch_size,) + self._img_size + (3,)
        image_batch = np.zeros(shape=batch_shape, dtype=np.float16)
        for i in range(batch_size):
            processed_array = self._process_image(images[i])
            image_batch[i] = processed_array
        return image_batch

    def process_images(self, images, batch_size=35):
        num_images = len(images)
        pbar = tqdm(total=num_images)
        processed_img = []
        for i in range(0, num_images, batch_size):
            end_index = i + batch_size
            image_batch = images[i:end_index]
            processed_batch = self.process_batch(image_batch)
            if end_index > num_images:
                batch_size = end_index - i
            pbar.update(batch_size)
            processed_img.append(processed_batch)

        return np.array(processed_img)

image_processor = ImagePreprocessor(is_training=True, img_size=(299, 299))
data_dir = "../flickr8k/Flicker8k_Dataset/"
image_files = list_pictures(data_dir)
print("Preprocessing images...")
array = image_processor.process_images(image_files)
print("Images preprocessed.")
print(array.shape)