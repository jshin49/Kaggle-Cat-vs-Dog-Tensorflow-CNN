from __future__ import print_function

import numpy as np    		# dealing with arrays
import os              		# dealing with directories
from tqdm import tqdm	   	# percentage bar for tasks
import matplotlib.pyplot as plt

import preprocess_images as pi

TRAIN_DIR = './data/train/'
TEST_DIR = './data/test/'
IMG_DIR = './npys/'
BATCH_DIR = './batches/'
IMG_SIZES = [64, 150, 224]
CHANNELS = 3
PIXEL_DEPTH = 255.0
BATCH_SIZES = [16, 32, 64]

dogs = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
cats = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR) if 'cat' in i]


def load_data(img_size):
    '''
        Read and Preprocess Images into each labels
    '''
    train_dogs = []
    train_cats = []

    if not os.path.exists(IMG_DIR + 'train_dogs' + str(img_size) + '.npy'):
        for dog in tqdm(dogs):
            img = pi.process_image(dog, img_size, PIXEL_DEPTH)
            train_dogs.append([np.array(img), 1])
        np.save(IMG_DIR + 'train_dogs' + str(img_size) + '.npy', train_dogs)
    else:
        train_dogs = np.load(IMG_DIR + 'train_dogs' +
                             str(img_size) + '.npy')

    if not os.path.exists(IMG_DIR + 'train_cats' + str(img_size) + '.npy'):
        for cat in tqdm(cats):
            img = pi.process_image(cat, img_size, PIXEL_DEPTH)
            train_cats.append([np.array(img), 0])
        np.save(IMG_DIR + 'train_cats' + str(img_size) + '.npy', train_cats)
    else:
        train_cats = np.load(IMG_DIR + 'train_cats' +
                             str(img_size) + '.npy')

    np.random.shuffle(train_dogs)
    np.random.shuffle(train_cats)

    return np.array(train_dogs), np.array(train_cats)


def prepare_train_data(train_dogs, train_cats, batch_size):
    '''
        Split data into evenly distributed batches
        return batches
    '''
    print("Generating training batches")
    split = batch_size / 2
    dogs = np.array(np.array_split(train_dogs, 12500 / split))
    cats = np.array(np.array_split(train_cats, 12500 / split))

    np.random.shuffle(dogs)
    np.random.shuffle(cats)

    # Choose mini batch from each
    batches = []
    for dog, cat in zip(dogs, cats):
        batch = np.concatenate([dog, cat])
        np.random.shuffle(batch)
        batches.append(batch)

    return batches


def next_batch(batches):
    return batches.pop(0)


def split_batches(batches):
    '''
    Split batches into 5 units and load them dynamically 
    during training, to save memory
    '''
    batches = np.array_split(batches, 5)
    for i in range(5):
        np.save(BATCH_DIR + 'batch' + str(i) + '.npy', batches[i])


def load_batches(step):
    return np.load(BATCH_DIR + 'batch' + str(step) + '.npy')


def init_data(config):
    train_dogs, train_cats = load_data(config.image_size)
    train_batches = prepare_train_data(
        train_dogs, train_cats, config.batch_size)
    valid_size = int(len(train_batches) * config.valid_size)

    # Split data set
    valid_batches = train_batches[-valid_size:]
    train_batches = train_batches[:-valid_size]
    split_batches(train_batches)

    # Release training data from memory
    train_batches = []

    return valid_batches


if __name__ == '__main__':
    for img_size in IMG_SIZES:
        train_dogs, train_cats = load_data(img_size)
        print("Train Dogs: {}".format(train_dogs.shape))
        print("Train Cats: {}".format(train_cats.shape))
    # train_dogs, train_cats = load_data(64)
    # print("Train Dogs: {}".format(train_dogs.shape))
    # print("Train Cats: {}".format(train_cats.shape))
    # print(train_cats[0].shape)
    # print(train_cats[0][0].shape)
    # print(train_cats[0][1])
    # batches = prepare_train_data(train_dogs, train_cats, 32)
    # print(len(batches))
    # split_batches(batches)

    # plt.imshow(train_dogs[0][0], cmap='gray')
    # plt.show()
    # plt.imshow(train_dogs[1][0], interpolation='nearest')
    # plt.figure()
    # plt.imshow(train_dogs[2][0], interpolation='nearest')
    # plt.figure()
    # plt.imshow(train_cats[0][0], interpolation='nearest')
    # plt.figure()
    # plt.imshow(train_cats[1][0], interpolation='nearest')
    # plt.figure()
    # plt.imshow(train_cats[2][0], interpolation='nearest')
    # plt.figure()
    # prepare_train_data()
