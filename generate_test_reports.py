import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import pandas as pd
import csv
import re
import sys

if (len(sys.argv) != 2):
    print(
        """
        Please specify which model to train
        \"python test_cnn.py DEEP\" or
        \"python test_cnn.py SIMPLE\"
        """)
    exit()

TEST_DIR = 'data/test'

# used for scaling/normalization
IMAGE_SIZE = 224
CHANNELS = 3
pixel_depth = 255.0  # Number of levels per pixel.

TEST_SIZE_ALL = 12500
file_list = os.listdir(TEST_DIR)
ordered_files = sorted(file_list, key=lambda x: (int(re.sub('\D', '', x)), x))
test_images_global = [TEST_DIR + i for i in ordered_files]
# print(test_images_global)

batch_size = 25
patch_size = 5
depth = 16
num_hidden = 64

loading_size = 500

#For tensorflow###########################################################
image_size = IMAGE_SIZE  # redundant
num_labels = 2
num_channels = 3  # rgb

test_labels = np.array(['unknownclass'] * TEST_SIZE_ALL)
test_labels = (test_labels == 'dogs').astype(np.float32)
test_labels = (np.arange(num_labels) == test_labels[
               :, None]).astype(np.float32)


def reformat(dataset):
    dataset = dataset.reshape(
        (-1, image_size, image_size, num_channels)).astype(np.float32)
    return dataset
##########################################################################


def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)  # cv2.IMREAD_GRAYSCALE
    # print(file_path)
    if (img.shape[0] >= img.shape[1]):  # height is greater than width
        resizeto = (IMAGE_SIZE, int(
            round(IMAGE_SIZE * (float(img.shape[1]) / img.shape[0]))))
    else:
        resizeto = (
            int(round(IMAGE_SIZE * (float(img.shape[0]) / img.shape[1]))), IMAGE_SIZE)

    img2 = cv2.resize(img, (resizeto[1], resizeto[
                      0]), interpolation=cv2.INTER_CUBIC)
    img3 = cv2.copyMakeBorder(
        img2, 0, IMAGE_SIZE - img2.shape[0], 0, IMAGE_SIZE - img2.shape[1], cv2.BORDER_CONSTANT, 0)

    return img3[:, :, ::-1]  # turn into rgb format


def prep_data(images):
    count = len(images)
    data = np.ndarray((count, IMAGE_SIZE, IMAGE_SIZE,
                       CHANNELS), dtype=np.float32)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        image_data = np.array(image, dtype=np.float32)
        ##############PREPROCESSING= subtract mean and normalize###############
        image_data[:, :, 0] -= np.mean(image_data[:, :, 0])
        image_data[:, :, 0] = (image_data[:, :, 0].astype(float)) / pixel_depth
        #image_data[:,:,0] /= np.std(image_data[:,:,0], axis=0)
        image_data[:, :, 1] -= np.mean(image_data[:, :, 1])
        image_data[:, :, 1] = (image_data[:, :, 1].astype(float)) / pixel_depth
        #image_data[:,:,1] /= np.std(image_data[:,:,1], axis=0)
        image_data[:, :, 2] -= np.mean(image_data[:, :, 2])
        image_data[:, :, 2] = (image_data[:, :, 2].astype(float)) / pixel_depth
        #image_data[:,:,2] /= np.std(image_data[:,:,2], axis=0)
        #######################################################################

        ##############PREPROCESSING= center each channel to 0 (-0.5,0.5)#######
        #image_data[:,:,0] = (image_data[:,:,0].astype(float) - pixel_depth / 2) / pixel_depth
        #image_data[:,:,1] = (image_data[:,:,1].astype(float) - pixel_depth / 2) / pixel_depth
        #image_data[:,:,2] = (image_data[:,:,2].astype(float) - pixel_depth / 2) / pixel_depth
        #######################################################################

        data[i] = image_data  # image_data.T
        if i % 250 == 0:
            print('Processed {} of {}'.format(i, count))
    return data


def load_test_set(start_index):
    test_images = test_images_global[start_index:start_index + loading_size]

    test_normalized = prep_data(test_images)
    test_dataset = reformat(test_normalized)
    print('Test set', test_dataset.shape, test_labels.shape)
    return test_dataset


graph = tf.Graph()

with graph.as_default():

    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(
        tf.float32, shape=(batch_size, num_labels))
    tf_test_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))

    # variables
    kernel_conv1 = tf.Variable(tf.truncated_normal([3, 3, 3, 32], dtype=tf.float32,
                                                   stddev=1e-1), name='weights_conv1')
    biases_conv1 = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32),
                               trainable=True, name='biases_conv1')
    kernel_conv2 = tf.Variable(tf.truncated_normal([3, 3, 32, 32], dtype=tf.float32,
                                                   stddev=1e-1), name='weights_conv2')
    biases_conv2 = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32),
                               trainable=True, name='biases_conv2')
    kernel_conv3 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], dtype=tf.float32,
                                                   stddev=1e-1), name='weights_conv3')
    biases_conv3 = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                               trainable=True, name='biases_conv3')
    fc1w = tf.Variable(tf.truncated_normal([50176, 64],
                                           dtype=tf.float32,
                                           stddev=1e-1), name='weights')
    fc1b = tf.Variable(tf.constant(1.0, shape=[64], dtype=tf.float32),
                       trainable=True, name='biases')
    fc2w = tf.Variable(tf.truncated_normal([64, 2],
                                           dtype=tf.float32,
                                           stddev=1e-1), name='weights')
    fc2b = tf.Variable(tf.constant(1.0, shape=[2], dtype=tf.float32),
                       trainable=True, name='biases')

    def model(data):
        parameters = []
        with tf.name_scope('conv1_1') as scope:
            conv = tf.nn.conv2d(data, kernel_conv1, [
                                1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases_conv1)
            conv1_1 = tf.nn.relu(out, name=scope)
            parameters += [kernel_conv1, biases_conv1]

        # pool1
        pool1 = tf.nn.max_pool(conv1_1,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        with tf.name_scope('conv2_1') as scope:
            conv = tf.nn.conv2d(pool1, kernel_conv2, [
                                1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases_conv2)
            conv2_1 = tf.nn.relu(out, name=scope)
            parameters += [kernel_conv2, biases_conv2]

        # pool2
        pool2 = tf.nn.max_pool(conv2_1,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        with tf.name_scope('conv3_1') as scope:
            conv = tf.nn.conv2d(pool2, kernel_conv3, [
                                1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases_conv3)
            conv3_1 = tf.nn.relu(out, name=scope)
            parameters += [kernel_conv3, biases_conv3]

        # pool3
        pool3 = tf.nn.max_pool(conv3_1,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(pool3.get_shape()[1:]))
            pool3_flat = tf.reshape(pool3, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool3_flat, fc1w), fc1b)
            fc1 = tf.nn.relu(fc1l)
            parameters += [fc1w, fc1b]

        # fc3
        with tf.name_scope('fc3') as scope:
            fc2l = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
            parameters += [fc2w, fc2b]
        return fc2l

    # Predictions for the training, validation, and test data.
    test_prediction = tf.nn.softmax(model(tf_test_dataset))


with tf.Session(graph=graph) as session:
    #writer= tf.summary.FileWriter('./graphs',session.graph)
    saver = tf.train.Saver()
    saver.restore(session, "./models/model_dropout2/model_224_dropout.ckpt")
    print("Model restored!")

    #################FOR TEST SET##########################
    num_iterations = int(TEST_SIZE_ALL / loading_size)
    num_test_steps = int(loading_size / batch_size)

    for iteration in range(num_iterations):
        test_dataset = load_test_set(loading_size * iteration)
        for step in range(num_test_steps):
            offset = (step * batch_size)
            batch_data = test_dataset[offset:(offset + batch_size), :, :, :]

            feed_dict = {tf_test_dataset: batch_data}
            batch_predictions = session.run(
                test_prediction, feed_dict=feed_dict)
            test_labels[loading_size * iteration + offset:loading_size *
                        iteration + (offset + batch_size)] = batch_predictions

    with open(sys.argv[2], 'wt') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(('id', 'label'))
        line = [0, 0]
        for i in range(len(test_labels)):
            line[0] = i + 1
            line[1] = test_labels[i][1]
            writer.writerow(line)
        print("File written!")
