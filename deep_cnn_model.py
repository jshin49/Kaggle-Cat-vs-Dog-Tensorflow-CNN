import sys
import numpy as np
import tensorflow as tf

from configs import DeepConfig
from utils.prepare_data import load_data, prepare_train_data


class DeepModel(object):

    def __init__(self, config, session, graph):
        print("Init Model object")
        self.graph = graph
        self.sess = session
        self.log_path = '/tmp/tensorboard/'
        self.config = config
        self.learning_rate = self.config.lr
        self.batch_size = self.config.batch_size
        self.image_size = self.config.image_size
        self.epochs = self.config.epochs
        sys.stdout.write('<log>Building Graph')
        # build computation graph
        self.build_graph()
        sys.stdout.write('</log>\n')

    def init_model(self, images, training):
        # CNN Model with tf.layers
        initializer = tf.contrib.layers.xavier_initializer()
        regularizer = tf.contrib.layers.l2_regularizer(self.config.l2)

        # Input Layer
        input_layer = tf.reshape(
            images, [-1,
                     self.config.image_size,
                     self.config.image_size,
                     self.config.channels])

        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            use_bias=True,
            bias_initializer=initializer,
            bias_regularizer=regularizer,
            activation=tf.nn.relu)
        conv1 = tf.layers.conv2d(
            inputs=conv1,
            filters=32,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            use_bias=True,
            bias_initializer=initializer,
            bias_regularizer=regularizer,
            activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1, pool_size=[2, 2], strides=(2, 2))

        # Convolutional Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            use_bias=True,
            bias_initializer=initializer,
            bias_regularizer=regularizer,
            activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(
            inputs=conv2,
            filters=64,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            use_bias=True,
            bias_initializer=initializer,
            bias_regularizer=regularizer,
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(
            inputs=conv2, pool_size=[2, 2], strides=(2, 2))

        # Convolutional Layer #3
        conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            use_bias=True,
            bias_initializer=initializer,
            bias_regularizer=regularizer,
            activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(
            inputs=conv3,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            use_bias=True,
            bias_initializer=initializer,
            bias_regularizer=regularizer,
            activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(
            inputs=conv3, pool_size=[2, 2], strides=(2, 2))

        # return pool3

        # Dense Layer
        # Flatten for 64*64 : 8,8,128
        flatten = tf.reshape(pool3, [-1, 8 * 8 * 128])
        # Flatten for 150*150 : 18,18,128
        # flatten = tf.reshape(pool3, [-1, 18 * 18 * 128])
        # Flatten for 224*224 : 28,28,128
        # flatten = tf.reshape(pool4, [-1, 28 * 28 * 128])
        # Dense Layer
        fc1 = tf.layers.dense(
            inputs=flatten,
            units=1024,
            activation=tf.nn.relu,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            use_bias=True,
            bias_initializer=initializer,
            bias_regularizer=regularizer)
        fc1 = tf.layers.dropout(
            inputs=fc1,
            rate=self.config.dropout,
            training=training)
        fc2 = tf.layers.dense(
            inputs=fc1,
            units=64,
            activation=tf.nn.relu,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            use_bias=True,
            bias_initializer=initializer,
            bias_regularizer=regularizer)
        fc2 = tf.layers.dropout(
            inputs=fc2,
            rate=self.config.dropout,
            training=training)

        # One output: Confidence score of being a dog
        logits = tf.layers.dense(inputs=fc2, units=1, activation=tf.nn.sigmoid)

        return logits

    # build the graph
    def build_graph(self):
        with self.graph.as_default():
            with self.sess:
                with tf.device('/gpu:0'):
                    # Input images
                    self.images = tf.placeholder(shape=[None,
                                                        self.config.image_size,
                                                        self.config.image_size,
                                                        self.config.channels],
                                                 dtype=tf.float32,
                                                 name='Images')
                    # self.val_images = tf.placeholder(shape=[None,
                    #                                         self.config.image_size,
                    #                                         self.config.image_size,
                    #                                         self.config.channels],
                    #                                  dtype=tf.float32,
                    #                                  name='Images')

                    # Input labels that represent the real outputs
                    self.labels = tf.placeholder(shape=[None, 1],
                                                 dtype=tf.float32,
                                                 name='Labels')
                    # self.val_labels = tf.placeholder(shape=[None, 1],
                    #                                  dtype=tf.float32,
                    #                                  name='Labels')

                    # Is Training?
                    self.training = tf.placeholder(dtype=tf.bool)

                    self.model = self.init_model(self.images, self.training)
                    # self.preds = tf.nn.sigmoid(self.model)
                    thresholds = tf.fill(
                        [self.config.batch_size], self.config.threshold)
                    self.predictions = tf.greater_equal(
                        self.model, thresholds)
                    correct_prediction = tf.equal(
                        self.predictions, tf.cast(self.labels, tf.bool))
                    self.accuracy = tf.reduce_mean(
                        tf.cast(correct_prediction, tf.float32))
                    self.loss = tf.losses.log_loss(
                        labels=self.labels, predictions=self.model)

                    # Validation
                    # self.val_model = self.init_model(
                    #     self.val_images, self.training)
                    # self.val_predictions = tf.greater_equal(
                    #     self.val_model, thresholds)
                    # val_correct_prediction = tf.equal(
                    #     self.val_predictions, tf.cast(self.val_labels, tf.bool))
                    # self.val_accuracy = tf.reduce_mean(
                    #     tf.cast(correct_prediction, tf.float32))
                    # self.val_loss = tf.losses.log_loss(
                    #     labels=self.labels, predictions=self.model)

                    # self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    #     labels=self.labels, logits=self.model))
                    # self.accuracy = tf.constant(1)
                    # self.loss = tf.constant(1)
                    self.optimizer = tf.train.RMSPropOptimizer(
                        learning_rate=self.learning_rate).minimize(self.loss)

                    # TensorBoard Summary
                    tf.summary.scalar("log_loss", self.loss)
                    tf.summary.scalar("accuracy", self.accuracy)
                    # tf.summary.scalar("val_loss", self.val_loss)
                    # tf.summary.scalar("val_accuracy", self.val_accuracy)
                    self.summary = tf.summary.merge_all()

                    self.init = tf.global_variables_initializer()
                    self.writer = tf.summary.FileWriter(
                        self.log_path, graph=self.sess.graph_def)

                with tf.device('/cpu:0'):
                    self.saver = tf.train.Saver(tf.trainable_variables())

    def generate_feed_dict(self, batch_images, batch_labels, training=False):
        return {
            self.images: batch_images,
            self.labels: batch_labels,
            self.training: training
        }

    def predict(self, batch_images, batch_labels):
        feed_dict = self.generate_feed_dict(batch_images, batch_labels, False)
        pred, loss, acc = self.sess.run(
            [self.model, self.loss, self.accuracy], feed_dict=feed_dict)
        return pred, loss, acc

    def train_eval_batch(self, batch_images, batch_labels, training=True):
        feed_dict = self.generate_feed_dict(
            batch_images, batch_labels, training)
        loss, acc, _ = self.sess.run(
            [self.loss, self.accuracy, self.optimizer], feed_dict=feed_dict)
        return loss, acc

    def eval_batch(self, batch_images, batch_labels, training=False):
        feed_dict = self.generate_feed_dict(
            batch_images, batch_labels, training)
        summary, loss, acc = self.sess.run(
            [self.summary, self.loss, self.accuracy], feed_dict=feed_dict)
        return summary, loss, acc

    def test_batch(self, batch_images, batch_labels, training=False):
        feed_dict = self.generate_feed_dict(
            batch_images, batch_labels, training)
        pred = self.sess.run(
            [self.model], feed_dict=feed_dict)
        return pred

    def save(self, step):
        self.saver.save(self.sess, self.config.ckpt_path +
                        '.ckpt', global_step=step)

    def restore(self):
        # get checkpoint state
        ckpt = tf.train.get_checkpoint_state('./ckpt')
        # restore session
        if ckpt and ckpt.model_checkpoint_path:
            self.sess.run(self.init)
            print("\nGlobal Variables Initialized")
            self.saver = tf.train.import_meta_graph(
                ckpt.model_checkpoint_path + '.meta')
            print("\nRestoring model")
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            self.sess.run(self.init)
            print("\nGlobal Variables Initialized")


if __name__ == '__main__':
    graph = tf.Graph()
    sess_config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=True)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    config = DeepConfig()
    model = DeepModel(config, sess, graph)

    # model.sess.run(model.init)
    # print("\nGlobal Variables Initialized")
    model.restore()

    train_dogs, train_cats = load_data(config.image_size)
    train_batches = prepare_train_data(
        train_dogs, train_cats, config.batch_size)
    # train_batch = next_batch(train_batches)
    batch_images, batch_labels = map(list, zip(*train_batches[0]))
    batch_images = np.array(batch_images)
    batch_labels = np.array(batch_labels).reshape(-1, 1)
    pred, loss, acc = model.predict(batch_images, batch_labels)
    # zeros = np.zeros(
    #     (8, 150, 150, 3), dtype=np.int)
    # pred, loss, acc = model.predict(
    #     zeros, np.array([1, 1, 1, 1, 1, 1, 1, 1]).reshape(-1, 1))
    print(pred)
    print(loss)
    print(acc)
