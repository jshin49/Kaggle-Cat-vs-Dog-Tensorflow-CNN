from __future__ import print_function

import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

from simple_cnn_model import SimpleModel
from deep_cnn_model import DeepModel
from configs import SimpleConfig, DeepConfig
from utils.prepare_data import init_data, load_batches


GRAPH_DIR = './graphs/'
if (len(sys.argv) != 2):
    print(
        """
        Please specify which model to train
        \"python train_cnn.py DEEP\" or
        \"python train_cnn.py SIMPLE\"
        """)
    exit()

DEPTH = sys.argv[1]
print("Training with " + DEPTH + " CNN Model")


def train(model, valid_batches):
    # For Plots
    x_steps = []
    y_training_loss = []
    y_training_accuracy = []
    y_valid_loss = []
    y_valid_accuracy = []
    valid_size = len(valid_batches)

    global_step = 0
    for i in range(5):
        print("Loading Batch")
        train_batches = load_batches(i)
        for epoch in range(model.epochs):
            for train_batch in tqdm(train_batches):
                train_batch_images, train_batch_labels = map(
                    list, zip(*train_batch))
                train_batch_images = np.array(train_batch_images)
                train_batch_labels = np.array(
                    train_batch_labels).reshape(-1, 1)
                loss, acc = model.train_eval_batch(
                    train_batch_images, train_batch_labels, True)

                if global_step % 20 == 0:
                    print('\nEpoch: %d, Global Step: %d, Train Batch Loss: %f, Train Batch Acc: %f' % (
                        epoch + 1, global_step, loss, acc))

                    valid_batch = valid_batches[global_step % valid_size]
                    valid_batch_images, valid_batch_labels = map(
                        list, zip(*valid_batch))
                    valid_batch_images = np.array(valid_batch_images)
                    valid_batch_labels = np.array(
                        valid_batch_labels).reshape(-1, 1)
                    val_loss, val_acc = model.eval_batch(
                        valid_batch_images, valid_batch_labels)
                    print('Epoch: %d, Global Step: %d, Valid Batch Loss: %f, Valid Batch Acc: %f' % (
                        epoch + 1, global_step, val_loss, val_acc))

                    x_steps.append(global_step / 50)
                    y_training_loss.append(loss)
                    y_training_accuracy.append(acc)
                    y_valid_loss.append(val_acc)
                    y_valid_accuracy.append(val_acc)

                global_step += 1

            # Save model after every epoch
            print('saving checkpoint')
            model.save(global_step)

            # Output graphs
            plt.clf()
            plt.plot(x_steps, y_training_loss)
            plt.xlabel('Steps (per 20 Global Steps)')
            plt.ylabel('Training Loss')
            plt.savefig(GRAPH_DIR + 'training_loss_simple_cnn.png')

            plt.clf()
            plt.plot(x_steps, y_training_accuracy)
            plt.xlabel('Steps')
            plt.ylabel('Training Accuracy')
            plt.savefig(GRAPH_DIR + 'training_accuracy_simple_cnn.png')

            plt.clf()
            plt.plot(x_steps, y_valid_loss)
            plt.xlabel('Steps')
            plt.ylabel('Validation Loss')
            plt.savefig(GRAPH_DIR + 'validation_loss_simple_cnn.png')

            plt.clf()
            plt.plot(x_steps, y_valid_accuracy)
            plt.xlabel('Steps')
            plt.ylabel('Validation Accuracy')
            plt.savefig(GRAPH_DIR + 'validation_accuracy_simple_cnn.png')


if __name__ == '__main__':
    # Initialize model
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

    valid_batches = init_data(model.config)

    model.sess.run(model.init)
    print("\nGlobal Variables Initialized")

    model.restore()
    train(model, valid_batches)
    print("Training Complete")
