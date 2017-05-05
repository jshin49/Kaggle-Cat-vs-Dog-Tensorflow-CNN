import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simple_cnn_model import SimpleModel
from deep_cnn_model import DeepModel
from configs import SimpleConfig, DeepConfig
from utils.prepare_data import init_data, init_test_data


TEST_DIR = 'data/test'

# Process test data and create batches in memory
graph = tf.Graph()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess_config = tf.ConfigProto(
    allow_soft_placement=True, log_device_placement=True, gpu_options=gpu_options)
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)
deep_config = DeepConfig()
deep_model = DeepModel(config, sess, graph)
simple_config = SimpleConfig()
simple_model = SimpleModel(config, sess, graph)

deep_model.restore()
print("Deep CNN Model Restored")
simple_model.restore()
print("Simple CNN Model Restored")

test_batches = init_test_data(model.config)
print(len(test_batches))

deep_weight = 0.6
simple_weight = 0.4

# Get predictions and Write them in CSV for submission
with open('results/ensemble.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'label'])

    for test_batch in test_batches:
        images, labels = map(list, zip(*test_batch))
        labels = np.array(labels).reshape(-1, 1)
        deep_pred = np.array(deep_model.test_batch(images, labels))
        simple_pred = np.array(simple_model.test_batch(images, labels))

        for deep, deep_id, simple, simple_id in zip(deep_pred.flatten(), simple_pred.flatten()):
            print(deep_id, deep, simple_id, simple)
            pred =

        # for id, label in zip(labels.flatten(), pred.flatten()):
        #     print(int(id), label)
        #     writer.writerow([int(id), label])
