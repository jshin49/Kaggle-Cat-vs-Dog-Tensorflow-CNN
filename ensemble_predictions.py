import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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


# Process test data and create batches in memory
graph = tf.Graph()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess_config = tf.ConfigProto(
    allow_soft_placement=True, log_device_placement=True, gpu_options=gpu_options)
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)

simple_config = SimpleConfig()
test_batches = init_test_data(simple_config)
print(len(test_batches))

simple_model = SimpleModel(simple_config, sess, graph)
simple_model.restore(True)
print("Simple CNN Model Restored")

graph2 = tf.Graph()
sess2 = tf.Session(graph=graph2, config=sess_config)
deep_config = DeepConfig()
deep_model = DeepModel(deep_config, sess, graph)
deep_model.restore(True)
print("Deep CNN Model Restored")

# Get predictions and Write them in CSV for submission
with open('results/ensemble.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'label'])

    for test_batch in test_batches:
        images, labels = map(list, zip(*test_batch))
        labels = np.array(labels).reshape(-1, 1)

        simple_pred = np.array(simple_model.test_batch(images, labels))
        deep_pred = np.array(deep_model.test_batch(images, labels))

        pred = []

        for deep, simple, id in zip(deep_pred.flatten(), simple_pred.flatten(), labels.flatten()):
            print(deep, id, simple)
# pred.append([deep_weight * deep_pred +
#              simple_weight * simple_pred, deep_id])

# for id, label in zip(labels.flatten(), pred.flatten()):
#     print(int(id), label)
#     writer.writerow([int(id), label])
