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


if (len(sys.argv) != 2):
    print(
        """
        Please specify which model to train
        \"python generate_test_reports.py DEEP\" or
        \"python generate_test_reports.py SIMPLE\"
        """)
    exit()

DEPTH = sys.argv[1]
print("Testing with " + DEPTH + " CNN Model")

TEST_DIR = 'data/test'

# Process test data and create batches in memory
graph = tf.Graph()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess_config = tf.ConfigProto(
    allow_soft_placement=True, log_device_placement=True, gpu_options=gpu_options)
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)
if DEPTH == 'DEEP':
    config = DeepConfig()
    model = DeepModel(config, sess, graph)
else:
    config = SimpleConfig()
    model = SimpleModel(config, sess, graph)

model.sess.run(model.init)
print("\nGlobal Variables Initialized")

model.restore()
print(DEPTH + " CNN Model Restored")

test_batches = init_test_data(model.config)
print(len(test_batches))

# valid_batches = init_data(model.config)

# Get predictions and Write them in CSV for submission
with open('results/' + DEPTH + '.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'label'])

    for test_batch in test_batches:
        images, labels = map(list, zip(*test_batch))
        labels = np.array(labels).reshape(-1, 1)
        pred = np.array(model.test_batch(images, labels))
        print(pred.flatten().shape)
        print(labels.flatten().shape)
        for id, label in zip(labels.flatten(), pred.flatten()):
            print(id, label)
            # writer.writerow([id, label])
