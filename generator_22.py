import numpy as np
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
import json
from keras.preprocessing.image import list_pictures
from keras.preprocessing.sequence import pad_sequences
from models import ImageCaptioningModel
from callbacks import callback
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import RMSprop
from model_22 import image_caption_model
from keras.preprocessing import sequence as keras_seq
import os
from preprocessing.image_processing import ImagePreprocessor
from keras.models import load_model


def train():
    class TokenizerWrap(Tokenizer):
        """Wrap the Tokenizer-class from Keras with more functionality."""

        def __init__(self, texts, num_words=None):
            """
            :param texts: List of strings with the data-set.
            :param num_words: Max number of words to use.
            """

            Tokenizer.__init__(self, num_words=num_words)

            # Create the vocabulary from the texts.
            self.fit_on_texts(texts)

            # Create inverse lookup from integer-tokens to words.
            self.index_to_word = dict(zip(self.word_index.values(),
                                          self.word_index.keys()))

        def token_to_word(self, token):
            """Lookup a single word from an integer-token."""

            word = " " if token == 0 else self.index_to_word[token]
            return word

        def tokens_to_string(self, tokens):
            """Convert a list of integer-tokens to a string."""

            # Create a list of the individual words.
            words = [self.index_to_word[token]
                     for token in tokens
                     if token != 0]

            # Concatenate the words to a single string
            # with space between all the words.
            text = " ".join(words)

            return text

        def captions_to_tokens(self, captions_listlist):
            """
            Convert a list-of-list with text-captions to
            a list-of-list of integer-tokens.
            """

            # Note that text_to_sequences() takes a list of texts.
            tokens = [self.texts_to_sequences(captions_list)
                      for captions_list in captions_listlist]

            return tokens


    def flatten(captions_listlist):
        captions_list = [caption
                         for captions_list in captions_listlist
                         for caption in captions_list]

        return captions_list


    mark_start = 'ssss '
    mark_end = ' eeee'


    def mark_captions(captions_listlist):
        captions_marked = [[mark_start + caption + mark_end
                            for caption in captions_list]
                           for captions_list in captions_listlist]

        return captions_marked


    fh = open('./flickr8k/dataset.json')
    raw_data = fh.read()
    data = json.loads(raw_data)

    img_groups = {}
    all_caps = []
    for img_group in data['images']:
        img_groups[img_group['filename']] = []

        for sentence in img_group['sentences']:
            img_groups[img_group['filename']].append(sentence['raw'])

            for word in sentence['raw'].strip().lower().split(' '):
                all_caps.append(word)

    image_files = list_pictures('./flickr8k/Flicker8k_Dataset/')

    num_images_train = len(image_files)

    captions_train = []

    for image_file in image_files:
        name = image_file.split('/')[-1]
        captions_train.append(img_groups[name])

    captions_train_marked = mark_captions(captions_train)

    captions_train_flat = flatten(captions_train_marked)

    tokenizer = TokenizerWrap(texts=captions_train_flat)

    token_start = tokenizer.word_index[mark_start.strip()]
    token_end = tokenizer.word_index[mark_end.strip()]

    tokens_train = tokenizer.captions_to_tokens(captions_train_marked)


    def get_random_caption_tokens(idx):
        """
        Given a list of indices for images in the training-set,
        select a token-sequence for a random caption,
        and return a list of all these token-sequences.
        """

        # Initialize an empty list for the results.
        result = []

        # For each of the indices.
        for i in idx:
            # The index i points to an image in the training-set.
            # Each image in the training-set has at least 5 captions
            # which have been converted to tokens in tokens_train.
            # We want to select one of these token-sequences at random.

            # Get a random index for a token-sequence.
            j = np.random.choice(len(tokens_train[i]))

            # Get the j'th token-sequence for image i.
            tokens = tokens_train[i][j]

            # Add this token-sequence to the list of results.
            result.append(tokens)

        return result


    global img_array
    if os.path.exists('./img_array.npy'):
        img_array = np.load('./img_array.npy')
    else:
        print("Preprocessing images...\n")
        image_processor = ImagePreprocessor(is_training=True, img_size=(224, 224))
        img_array = image_processor.process_images(image_files)
        print("\nImages preprocessed.")
        np.save('./img_array', img_array)


    def batch_generator(batch_size):
        """
        Generator function for creating random batches of training-data.

        Note that it selects the data completely randomly for each
        batch, corresponding to sampling of the training-set with
        replacement. This means it is possible to sample the same
        data multiple times within a single epoch - and it is also
        possible that some data is not sampled at all within an epoch.
        However, all the data should be unique within a single batch.
        """

        # Infinite loop.
        while True:
            # Get a list of random indices for images in the training-set.
            idx = np.random.randint(num_images_train, size=batch_size)

            # Get the pre-computed transfer-values for those images.
            # These are the outputs of the pre-trained image-model.
            transfer_values = img_array[idx]

            # For each of the randomly chosen images there are
            # at least 5 captions describing the contents of the image.
            # Select one of those captions at random and get the
            # associated sequence of integer-tokens.
            tokens = get_random_caption_tokens(idx)

            # Count the number of tokens in all these token-sequences.
            num_tokens = [len(t) for t in tokens]

            # Max number of tokens.
            max_tokens = np.max(num_tokens)

            # Pad all the other token-sequences with zeros
            # so they all have the same length and can be
            # input to the neural network as a numpy array.
            tokens_padded = pad_sequences(tokens,
                                          maxlen=max_tokens,
                                          padding='post',
                                          truncating='post')

            # Further prepare the token-sequences.
            # The decoder-part of the neural network
            # will try to map the token-sequences to
            # themselves shifted one time-step.
            decoder_input_data = tokens_padded[:, 0:-1]
            decoder_output_data = tokens_padded[:, 1:]

            captions = keras_seq.pad_sequences(tokens, padding='post')
            captions_extended1 = keras_seq.pad_sequences(captions,
                                                         maxlen=captions.shape[-1] + 1,
                                                         padding='post')
            captions_one_hot = list(map(tokenizer.sequences_to_matrix,
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

            captions_output = pad_sequences(captions_output,
                                            maxlen=max_tokens,
                                            padding='post',
                                            truncating='post')

            decoder_output_data = captions_output

            # Dict for the input-data. Because we have
            # several inputs, we use a named dict to
            # ensure that the data is assigned correctly.
            x_data = \
                {
                    'decoder_input': decoder_input_data,
                    'input_1': transfer_values
                }

            # Dict for the output-data.
            y_data = \
                {
                    'decoder_output': decoder_output_data
                }

            # print(x_data['decoder_input'].shape, y_data['decoder_output'].shape)

            yield (x_data, y_data)


    image_data_path = './flickr8k/Flicker8k_Dataset/'

    caption_path = './flickr8k/dataset.json'

    images = list_pictures(image_data_path)

    num_image = len(images)

    batch_size = 100

    num_seq_per_image = 5

    total_seq = num_image * num_seq_per_image

    steps_per_epoch = total_seq // batch_size

    generator = batch_generator(batch_size=batch_size)

    decoder_model = image_caption_model(word_size=8388)

    decoder_model.compile(optimizer=RMSprop(lr=1e-3),
                          loss='categorical_crossentropy')

    '''
    image_captioning_model = ImageCaptioningModel(rnn_mode='lstm',
                                                  drop_rate=0.1,
                                                  hidden_dim=3,
                                                  rnn_state_size=256,
                                                  embedding_size=512,
                                                  rnn_activation='tanh',
                                                  cnn_model=InceptionV3,
                                                  optimizer=RMSprop,
                                                  initializer='random_uniform',
                                                  learning_rate=0.001,
                                                  reg_l1=0.001,
                                                  reg_l2=0.001,
                                                  num_word=8388,
                                                  is_trainable=False,
                                                  metrics=None,
                                                  loss='categorical_crossentropy')
                                                  
    image_captioning_model.build()
    decoder_model = image_captioning_model.image_captioning_model
    
    '''
    save_path = 'model22.h5'



    decoder_model.fit_generator(generator=generator,
                                steps_per_epoch=steps_per_epoch,
                                epochs=5,
                                callbacks=callback('checkpoint22.h5', './logs/'))

    decoder_model.save(save_path)


def predict(image_path, max_tokens=30):
    model_path = 'model22.h5'

    decoder_model = load_model(model_path)

    preprocessor = ImagePreprocessor(is_training=False)

    if type(image_path) == str:
        image = preprocessor.process_image(image_path)
        image_batch = np.expand_dims(image, axis=0)
    elif type(image_path) == list:
        image_batch = preprocessor.process_batch(image_path)
    else:
        raise ValueError('Input image name is not vaild')

    shape = (1, max_tokens)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int)

    token_int = 2

    output_text = ''
    outputs = []

    count_tokens = 0

    while token_int != 3 and count_tokens < max_tokens:
        decoder_input_data[0, count_tokens] = token_int

        x_data = \
        {
            'input_1': image_batch,
            'decoder_input': decoder_input_data
        }

        decoder_output = decoder_model.predict(x_data)

        token_onehot = decoder_output[0, count_tokens, :]

        token_int = np.argmax(token_onehot)

        outputs.append(token_int)
        count_tokens += 1

    print(outputs)


if __name__ == '__main__':
    # predict(['./flickr8k/Flicker8k_Dataset/3452411712_5b42d2a1b5.jpg'])
    train()