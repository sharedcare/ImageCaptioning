import numpy as np
import os
import tensorflow as tf
import json

from datasets import imagenet
from nets import inception_v4
from nets.inception_v4 import inception_v4
from nets.inception_v4 import inception_v4_arg_scope
from nets.inception_v4 import inception_v4_base
from preprocessing import inception_preprocessing

from tensorflow.contrib import slim

flags = tf.flags

flags.DEFINE_string('checkpoints_dir', './tmp/checkpoints', '')
flags.DEFINE_string('data_dir', './data', '')
flags.DEFINE_string('output_dir', './output', '')

FLAGS = flags.FLAGS

image_size = inception_v4.default_image_size

def img2tag(file_name):
    with tf.Graph().as_default():
        # Get image
        f = open(file_name, 'rb')
        image_string = f.read()
        f.close()
        image = tf.image.decode_jpeg(image_string, channels=3)

        # Process and apply the pre-trained model on the image   
        processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
        processed_images  = tf.expand_dims(processed_image, 0)

        with slim.arg_scope(inception_v4_arg_scope()):
            logits, _ = inception_v4(processed_images, num_classes=1001, is_training=False)
        probabilities = tf.nn.softmax(logits)

        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(FLAGS.checkpoints_dir, 'inception_v4.ckpt'),
            slim.get_model_variables('InceptionV4'))

        with tf.Session() as sess:
            init_fn(sess)
            np_image, probabilities = sess.run([image, probabilities])
            probabilities = probabilities[0, 0:]
            sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]

        names = imagenet.create_readable_names_for_imagenet_labels()

        results = []

        for i in range(10):
            index = sorted_inds[i]
            # print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, names[index]))
            results.append(names[index].split(', '))

        return [item for sublist in results for item in sublist]

# img2tag('005OCjVRjw1eoevxoji02j30cw0w30v4.jpg')

json_path = os.path.join(FLAGS.data_dir, 'dataset.json')
json_data = json.load(open(json_path))

data = []

for i in range(len(json_data['images'])):
# for i in range(20):
    print("{} / {}".format(i + 1, len(json_data['images'])))

    filename = json_data['images'][i]['filename']
    sentences_tokens = []
    for j in range(len(json_data['images'][i]['sentences'])):
        sentences_tokens.append(json_data['images'][i]['sentences'][j]['tokens'])
    
    tags = img2tag(os.path.join(FLAGS.data_dir, 'imgs/' + filename))

    img_data = {
        'filename': filename,
        'sentences_tokens': sentences_tokens,
        'tags': tags
    }

    data.append(img_data)

    if len(data) >= 200 or i == len(json_data['images']) - 1:

        json_output = json.dumps(data)

        output_path = os.path.join(FLAGS.output_dir, "data_{}.json".format(i))

        f = open(output_path, 'w+')

        f.write(json_output)

        f.close()

        data = []