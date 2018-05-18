import json
from preprocessing.image_processing import ImagePreprocessor
from preprocessing.caption_processing import CaptionPreprocessor
from keras.preprocessing.image import list_pictures
from keras.utils import Sequence
import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences

image_files = list_pictures('./flickr30k/flickr30k_images/')

image1 = image_files[:len(image_files)//3]
image2 = image_files[len(image_files)//3:(2*len(image_files))//3]
image3 = image_files[(2*len(image_files))//3:]

image_processor = ImagePreprocessor(is_training=True, img_size=(299, 299))

if os.path.exists('./img_array1.npy') and os.path.exists('./img_array2.npy') and os.path.exists('./img_array3.npy') and os.path.exists('./img_array.npy'):
    img_array1 = np.load('./img_array1.npy')
    img_array2 = np.load('./img_array2.npy')
    img_array3 = np.load('./img_array3.npy')
    img_array_og = np.load('./img_array.npy')
else:
    print("Preprocessing images...\n")
    img_array1 = image_processor.process_images(image1)
    np.save('./img_array1', img_array1)
    img_array2 = image_processor.process_images(image2)
    np.save('./img_array2', img_array2)
    img_array3 = image_processor.process_images(image3)
    np.save('./img_array3', img_array3)
    print("\nImages preprocessed.")

img_array = np.concatenate((img_array1, img_array2, img_array3))
print(img_array.shape)
print(img_array1.shape)
print(np.allclose(img_array_og, img_array))