from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
from datetime import datetime
import os
from six import iteritems
import numpy as np
import my_net
import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
import time
import itertools
import argparse
import sys
from sklearn.model_selection import KFold
import random


def main(args):
    network = importlib.import_module(args.model_def)

    # create log directory and model diectory
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # Write arguments to a text file
    write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))

    # import dataset and divide it into trainset and validationset
    np.random.seed(seed=args.seed)

    paths_pairs = get_paths_pairs(args.data_dir)
    # paths_pairs: Returns a list of paths_pairs shape(?,2)

    nrof_validation_pairs = 100
    validation_set = paths_pairs[0:nrof_validation_pairs]
    train_set = paths_pairs[nrof_validation_pairs:len(paths_pairs)]

    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    if args.pretrained_model:
        print('Pre-trained model: %s' % os.path.expanduser(args.pretrained_model))

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)

        # Placeholders
        learning_rate_placehloder = tf.placeholder(tf.float32, name='learning_rate')
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        is_training_placeholder = tf.placeholder(tf.bool, name='is_trainning')
        lidarscan_paths_placeholder = tf.placeholder(tf.string, shape=(None, 2), name='lidarscan_paths')
        labels_placeholder = tf.placeholder(tf.int64, shape=(None, 2), name='labels')

        # queue operation
        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                              dtypes=[tf.string, tf.int64],
                                              shapes=[(2,), (2,)],
                                              shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([lidarscan_paths_placeholder, labels_placeholder])

        nrof_preprocess_threads = 4
        lidarscans_and_labels_and_coordinates = []
        for _ in range(nrof_preprocess_threads):
            filenames, labels = input_queue.dequeue()  # shape: (2,) (2,) ?
            lidarscans = []
            coordinates = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                lidarscan = tf.decode_raw(file_contents, tf.float32)
                lidarscan = tf.reshape(lidarscan, [2, 32768])  # for model_cnn.py 64 x 512 = 32768
                coordinate_filename = tf.strings.regex_replace(filename, 'velodyne_scan', 'global_ego_data')
                coordinate = tf.decode_raw(tf.read_file(coordinate_filename), tf.float64)[16:18]
                # easting and northing coordinate

                # per lidarscan normalization
                mean = tf.reduce_mean(lidarscan, axis=1, keepdims=True)  # ((distance_mean),(reflection_mean))
                lidarscan -= mean

                _, var = tf.nn.moments(lidarscan, axes=1, keep_dims=True)
                std = tf.sqrt(var)  # ((distance_std),(reflection_std))
                lidarscan /= std

                # reshape the normalized lidarscan to (64, 512, 2) for model_cnn.py
                distance_data = lidarscan[0]
                reflectance_data = lidarscan[1]
                lidarscan = tf.reshape(tf.stack([distance_data, reflectance_data], axis=1), (64, 512, 2))
                lidarscans.append(lidarscan)
                coordinates.append(coordinate)
                # But in CNN the shape should be (height,width,channel) channel can be 2 if only distance and reflectance
                # are considerd as features. But the original width 2000 pix is not appropriate as input of CNN.
                # So the size of a lidarscan is reduced to 64 x 512 with scan_downsample.py

            lidarscans_and_labels_and_coordinates.append([lidarscans, labels, coordinates])

        lidarscan_batch, labels_batch, coordinates_batch = tf.train.batch_join(
            lidarscans_and_labels_and_coordinates, batch_size=batch_size_placeholder,  # ！！！！！batch size??
            shapes=[(64, 512, 2), (), (2,)], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)
        lidarscan_batch = tf.identity(lidarscan_batch, 'lidarscan_batch')
        lidarscan_batch = tf.identity(lidarscan_batch, 'input')
        labels_batch = tf.identity(labels_batch, 'labels_batch')
        coordinates_batch = tf.identity(coordinates_batch, 'coordinates_batch')

        # Build the inference graph
        prelogits = network.inference(lidarscan_batch, args.keep_probability, weight_decay=args.weight_decay,
                                      is_training=is_training_placeholder,
                                      bottleneck_layer_size=args.embedding_size)
        prelogits = tf.reshape(prelogits, [-1, args.embedding_size])
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        # Split embeddings into anchor, positive and negative and calculate triplet loss
        embedding1, embedding2 = tf.unstack(tf.reshape(embeddings, [-1, 2, args.embedding_size]), 2, 1)
        distances = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(embedding1, embedding2)), 1))
        coordinate1, coordinate2 = tf.unstack(tf.reshape(coordinates_batch, [-1, 2, 2]), 2, 1)
        actual_distances = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(coordinate1, coordinate2)), 1))
        distance_loss=tf.reduce_mean(tf.square(tf.subtract(distances,actual_distances)),0)
        learning_rate = learning_rate_placehloder  # 源代码用了exponential_decay; the original code used exponential_decay

        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([distance_loss] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = my_net.train(total_loss, global_step, args.optimizer,
                                learning_rate, args.moving_average_decay, tf.global_variables())

        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()  # not used

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)  # GPU的设置 还没看
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # Initialize variables
        sess.run(tf.global_variables_initializer(), feed_dict={is_training_placeholder: True})
        sess.run(tf.local_variables_initializer(), feed_dict={is_training_placeholder: True})

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)
        # 启动所有graph里的queue runners。batch_join() added a queue runner to the Graph's QUEUE_RUNNER collection

        with sess.as_default():

            if args.pretrained_model:
                print('Restoring pretrained model: %s' % args.pretrained_model)
                saver.restore(sess, os.path.expanduser(args.pretrained_model))

            # Training and validation loop
            epoch = 0
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // args.epoch_size
                # Train
                train(args, args.learning_rate_schedule_file, epoch, train_set, nrof_subdatasets,
                      classes_per_subdataset, sess, enqueue_op,
                      lidarscan_paths_placeholder, labels_placeholder,
                      args.embedding_size, embeddings, coordinates_batch, labels_batch, batch_size_placeholder,
                      is_training_placeholder, learning_rate_placehloder, total_loss, train_op, global_step,
                      summary_writer, pos_dist, neg_dist, regularization_losses)
                # 源代码里的summary_op并没有被sess运行。但是summary_writer里有写sess
                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, model_dir, step, subdir, summary_writer)

                # Evaluate based on validation_set
                evaluate(sess, validation_paths, embeddings, enqueue_op, batch_size_placeholder,
                         lidarscan_paths_placeholder, labels_placeholder, args.embedding_size,
                         labels_batch, learning_rate_placehloder, is_training_placeholder, args.batch_size,
                         summary_writer, step, log_dir, actual_issame, args.nrof_folds)

    return model_dir


def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in iteritems(vars(args)):
            f.write('%s: %s\n' % (key, str(value)))


def get_paths_pairs(data_dir):
    paths_pairs_file = os.path.join(os.path.expanduser(data_dir), 'paths_pairs_file.txt')
    paths_pairs = []
    with open(paths_pairs_file, 'r') as f:
        paths = f.readlines()
        assert (len(paths) % 2 == 0)
        i = 0
        while i < len(paths):
            paths_pair = []
            paths_pair.append(paths[i])
            paths_pair.append(paths[i + 1])
            paths_pairs.append(paths_pair)
            i = i + 2
    random.shuffle(paths_pairs)
    return paths_pairs
