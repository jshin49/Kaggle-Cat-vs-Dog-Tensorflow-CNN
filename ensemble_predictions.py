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
from deep_cnn_model2 import DeepModel2
from configs import SimpleConfig, DeepConfig
from utils.prepare_data import init_data, init_test_data


# Process test data and create batches in memory
graph = tf.Graph()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess_config = tf.ConfigProto(
    allow_soft_placement=True, log_device_placement=True, gpu_options=gpu_options)
sess_config.gpu_options.allow_growth = True
sess = tf.Session(graph=graph, config=sess_config)

deep_config = DeepConfig()
simple_config = SimpleConfig()

test_batches = init_test_data(simple_config)
print(len(test_batches))

simple_model = SimpleModel(simple_config, sess, graph)
simple_model.restore('./ckpt_simple/ckpt')
print("Simple CNN Model Restored")

graph2 = tf.Graph()
sess2 = tf.Session(graph=graph2, config=sess_config)
deep_model1 = DeepModel(deep_config, sess2, graph2)
deep_model1.restore('./ckpt_deep1/ckpt')
print("Deep CNN Model 1 Restored")

graph3 = tf.Graph()
sess3 = tf.Session(graph=graph3, config=sess_config)
deep_model2 = DeepModel2(deep_config, sess3, graph3)
deep_model2.restore('./ckpt_deep2/ckpt/')
print("Deep CNN Model 2 Restored")

# Get predictions and Write them in CSV for submission
with open('results/ensemble.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'label'])

    for test_batch in test_batches:
        images, labels = map(list, zip(*test_batch))
        labels = np.array(labels).reshape(-1, 1)

        simple_pred = np.array(simple_model.test_batch(images, labels))
        deep_pred1 = np.array(deep_model1.test_batch(images, labels))
        deep_pred2 = np.array(deep_model2.test_batch(images, labels))

        for deep1, deep2, simple, id in zip(deep_pred1.flatten(), deep_pred2.flatten(), simple_pred.flatten(), labels.flatten()):
            # print(id, deep1, deep2, simple)
            pred = 0.4 * deep1 + 0.3 * deep2 + 0.3 * simple
            # pred = (deep1 + deep2 + simple) / 3
            print(int(id), pred, deep1, deep2, simple)
            writer.writerow([int(id), pred])

# for id, label in zip(labels.flatten(), pred.flatten()):
#     print(int(id), label)
#     writer.writerow([int(id), label])
