
# coding: utf-8

# # Image Captioning

# This is the project for Deep Learning using Tensorflow: [Project Proposal](https://sharedcare.io/ImageCaptioning/proposal/)

# # Prepare the Training Data
# ##### Location to save the MSCOCO data.
# MSCOCO_DIR="data/mscoco"
# 
# ##### Run the preprocessing script.
# sh ./download_and_preprocess_mscoco.sh \${MSCOCO_DIR}
# 
# ##### Build the vocabulary.
# DATA_DIR="mscoco/raw-data"
# 
# OUTPUT_DIR="mscoco/output"
# 
# python build_mscoco_data.py --train_image_dir=\${DATA_DIR}/train2014/ --val_image_dir=\${DATA_DIR}/val2014/ --train_captions_file=\${DATA_DIR}/annotations/captions_train2014.json --val_captions_file=\${DATA_DIR}/annotations/captions_val2014.json --output_dir=\${OUTPUT_DIR}/ --word_counts_output_file=\${OUTPUT_DIR}/word_counts.txt
# 

# # Download the Inception v3 Checkpoint
# ##### Location to save the Inception v3 checkpoint.
# INCEPTION_DIR="data"
# 
# mkdir -p ${INCEPTION_DIR}
# 
# wget "http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"
# 
# tar -xvf "inception_v3_2016_08_28.tar.gz" -C ${INCEPTION_DIR}
# 
# rm "inception_v3_2016_08_28.tar.gz"

# # Train a Model
# ##### Directory containing preprocessed MSCOCO data.
# MSCOCO_DIR="data/data/mscoco"
# 
# ##### Inception v3 checkpoint file.
# INCEPTION_CHECKPOINT="data/inception_v3.ckpt"
# 
# ##### Directory to save the model.
# MODEL_DIR="model"
# 
# ##### Train the model.
# (tensorflow) device:im2txt user\$ python train.py --input_file_pattern="\${MSCOCO_DIR}/train-?????-of-00256" --inception_checkpoint_file="\${INCEPTION_CHECKPOINT}" --train_dir="\${MODEL_DIR}/train" --train_inception=false --number_of_steps=1000000

# ##### ＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿ Divider ＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿

# # Using pre-trained models
# First, we need to extract the tag from image using image recognition network.
# Hence, we use [Interception V4](http://arxiv.org/abs/1602.07261) model for this project. To fit our purpose, we need use pre-trained models.
# The code for train a Interception V4 model is avaliable [here](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v4.py) and the pre-trained model is avaliable [here](http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz).

# ### Download the Inception V4 checkpoint

# In[7]:


import tensorflow as tf
from datasets import dataset_utils

url = "http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz"
checkpoints_dir = './tmp/checkpoints'

if not tf.gfile.Exists(checkpoints_dir):
    tf.gfile.MakeDirs(checkpoints_dir)

dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)


# ### Apply Pre-trained Inception V4 model to Images.

# In[3]:


import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib

from datasets import imagenet
from nets import inception_v4
from nets.inception_v4 import inception_v4
from nets.inception_v4 import inception_v4_arg_scope
from nets.inception_v4 import inception_v4_base
from preprocessing import inception_preprocessing

from tensorflow.contrib import slim

image_size = inception_v4.default_image_size

with tf.Graph().as_default():
    # Get image from url
    url = 'https://static01.nyt.com/images/2018/04/02/nyregion/02nytoday1/02nytoday1-master768.jpg'
    image_string = urllib.urlopen(url).read()
    
    # Get image from local
    # image_dir = './flickr8k/Flicker8k_Dataset/10815824_2997e03d76.jpg'
    # image_file = open(image_dir, 'rb')
    # image_string = image_file.read()
    # image_file.close()
    
    image = tf.image.decode_jpeg(image_string, channels=3)
    
    # Process and apply the pre-trained model on the image   
    processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
    # Shape of processed_image is (image_size, image_size, 3)
    processed_images  = tf.expand_dims(processed_image, 0)
    # Shape of processed_images is (1, image_size, image_size, 3)
    
    # Create the model, use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(inception_v4_arg_scope()):
        # Get prediction of the model
        logits, _ = inception_v4(processed_images, num_classes=1001, is_training=False)
    probabilities = tf.nn.softmax(logits)
    
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'inception_v4.ckpt'),
        slim.get_model_variables('InceptionV4'))
    
    with tf.Session() as sess:
        init_fn(sess)
        np_image, probabilities = sess.run([image, probabilities])
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]
        
    plt.figure()
    plt.imshow(np_image.astype(np.uint8))
    plt.axis('off')
    plt.show()

    names = imagenet.create_readable_names_for_imagenet_labels()
    for i in range(5):
        index = sorted_inds[i]
        print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, names[index]))


# In[5]:


import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib
    
from datasets import imagenet
from nets import inception_v4
from nets.inception_v4 import inception_v4
from nets.inception_v4 import inception_v4_arg_scope
from nets.inception_v4 import inception_v4_base
from preprocessing import inception_preprocessing

from tensorflow.contrib import slim

def process_image(image_string, checkpoints_dir):
    with tf.Graph().as_default():
        image = tf.image.decode_jpeg(image_string, channels=3)
        # Process and apply the pre-trained model on the image   
        processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
        processed_images  = tf.expand_dims(processed_image, 0)

        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(inception_v4_arg_scope()):
            logits, _ = inception_v4(processed_images, num_classes=1001, is_training=False)
        probabilities = tf.nn.softmax(logits)
        
        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, 'inception_v4.ckpt'),
            slim.get_model_variables('InceptionV4'))
        
        with tf.Session() as sess:
            init_fn(sess)
            np_image, probabilities = sess.run([image, probabilities])
            probabilities = probabilities[0, 0:]
            sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]

        plt.figure()
        plt.imshow(np_image.astype(np.uint8))
        plt.axis('off')
        plt.show()

        names = imagenet.create_readable_names_for_imagenet_labels()
        for i in range(5):
            index = sorted_inds[i]
            print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, names[index]))
        
        return (probabilities, names, sorted_inds)
        
checkpoints_dir = './tmp/checkpoints'
# Get image from local
image_size = inception_v4.default_image_size
image_dir = './flickr8k/Flicker8k_Dataset/'
filenames = ['10815824_2997e03d76.jpg', '667626_18933d713e.jpg', '12830823_87d2654e31.jpg', '23445819_3a458716c1.jpg']

results = {}
for filename in filenames:
    image_file = open(image_dir + filename, 'rb')
    image_string = image_file.read()
    image_file.close()

    probabilities, names, sorted_inds = process_image(image_string, checkpoints_dir)
    result = []
    for i in range(3):
        index = sorted_inds[i]
        result.append((probabilities[index], names[index]))
    results[filename] = result


# In[ ]:


# Load image and predict
img_path = 'IMG_0098.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=5)[0])


# ### Conclusion on Interception V4
# As we apply the Interception V4 model to a image, we can get the tags with probabilities describing specific image. In this case, we can use this model to train our actual image captioning model.

# ![image-captioning](https://github.com/Hvass-Labs/TensorFlow-Tutorials/raw/e686db8f90669350087d6e0879d9d9dda390b4e4/images/22_image_captioning_flowchart.png)

# The image shows above demonstrate how to train an image captioning network. For our project, we use inception V4 and LSTM instead of VGG16 and GRU.

# ## Inception V3 Pre-trained Model

# In[2]:


# Imports
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.preprocessing.text import Tokenizer
from keras.layers import BatchNormalization, Dense, RepeatVector, Embedding, Input
from keras.models import Model
from keras import backend as K


# In[2]:


# Model summary
model = InceptionV3(weights='imagenet')

print(model.summary())


# In[3]:


input_tensor = Input(shape=(299, 299, 3))  # this assumes K.image_data_format() == 'channels_last'

model_no_top = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False, pooling='avg')
print(model_no_top.summary())


# In[4]:


# Freeze all convolutional InceptionV3 layers
for layer in model_no_top.layers:
    layer.trainable = False

embedding_size = 300
x = model_no_top.output
x = BatchNormalization()(x)
x = Dense(embedding_size)(x)
transfer_output = x

image_input = model_no_top.input
transfer_model = Model(inputs=image_input,
                             outputs=transfer_output)


# In[5]:


img_size = K.int_shape(image_input)[1:3]
img_size


# In[6]:


transfer_values_size = K.int_shape(transfer_output)[1]
transfer_values_size


# ## Preprocess

# In[7]:


from tqdm import tqdm

def print_progress(count, pbar):
    pbar.update(count)


# In[36]:


from keras.preprocessing.image import list_pictures, load_img, img_to_array

def process_images(data_dir, filenames, batch_size=32):
    # Get the number of total processing images
    num_images = len(filenames)
    pbar = tqdm(total=num_images)
    
    # Initial image batch to store image arrays
    batch_shape = (batch_size, ) + img_size + (3, )
    image_batch = np.zeros(shape=batch_shape, dtype=np.float16)
    
    # Initial transfer values to store outputs from image model
    transfer_shape = (num_images, transfer_values_size)
    transfer_values = np.zeros(shape=transfer_shape, dtype=np.float16)
    
    # Initialize index into the filenames.
    start_index = 0
    
    while start_index < num_images:
        end_index = start_index + batch_size
    
        if end_index > num_images:
            end_index = num_images
            batch_size = end_index - start_index
        
        pbar.update(batch_size)
        # Load all the images in the batch.
        for i, filename in enumerate(filenames[start_index:end_index]):

            # Load and resize the image.
            # This returns the image as a numpy-array.
            img = load_img(filename, target_size=img_size)
            img_array = img_to_array(img)
            
            # Save the image for later use.
            image_batch[i] = img_array

        # Use the pre-trained image-model to process the image.
        # Note that the last batch may have a different size,
        # so we only use the relevant images.
        transfer_values_batch =             transfer_model.predict(image_batch[0:batch_size])

        # Save the transfer-values in the pre-allocated array.
        transfer_values[start_index:end_index] =             transfer_values_batch[0:batch_size]

        # Increase the index for the next loop-iteration.
        start_index = end_index
        
    # Print newline.
    print()

    return transfer_values

def process_image():
    pbar.update()


# In[28]:


def process_images_train(data_dir, filenames_train):
    print("Processing {0} images in training-set ...".format(len(filenames_train)))

    # Path for the cache-file.
    cache_path = os.path.join(data_dir,
                              "transfer_values_train.pkl")

    # If the cache-file already exists then reload it,
    # otherwise process all images and save their transfer-values
    # to the cache-file so it can be reloaded quickly.
    transfer_values = process_images(data_dir, filenames_train)

    return transfer_values


# In[10]:


def process_images_val():
    print("Processing {0} images in validation-set ...".format(len(filenames_val)))

    # Path for the cache-file.
    cache_path = os.path.join(coco.data_dir, "transfer_values_val.pkl")

    # If the cache-file already exists then reload it,
    # otherwise process all images and save their transfer-values
    # to the cache-file so it can be reloaded quickly.
    transfer_values = cache(cache_path=cache_path,
                            fn=process_images,
                            data_dir=coco.val_dir,
                            filenames=filenames_val)

    return transfer_values


# In[ ]:


data_dir = "./flickr8k/Flicker8k_Dataset/"

filenames_train = list_pictures(data_dir)
process_images_train(data_dir, filenames_train)


# In[11]:


get_ipython().run_cell_magic('time', '', 'transfer_values_train = process_images_train()\nprint("dtype:", transfer_values_train.dtype)\nprint("shape:", transfer_values_train.shape)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'transfer_values_val = process_images_val()\nprint("dtype:", transfer_values_val.dtype)\nprint("shape:", transfer_values_val.shape)')


# ## Tokenize
