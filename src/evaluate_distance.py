from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
from datetime import datetime
import os
from six import iteritems
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
import time
import argparse
import sys


def main(args):
    network = importlib.import_module(args.model_def)

    # create log directory and model diectory
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)

    # Write arguments to a text file
    write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))

    # import dataset and divide it into trainset and validationset
    np.random.seed(seed=args.seed)

    pairs_paths, pairs_dists = get_paths_pairs(args.data_dir, args.pairs_file)
    # paths_pairs: Returns a list of shuffled paths_pairs shape(?,2)
    pairs_paths = pairs_paths[0:4096]
    pairs_dists = pairs_dists[0:4096]

    print('Log directory: %s' % log_dir)
    if args.pretrained_model:
        print('Pre-trained model: %s' % os.path.expanduser(args.pretrained_model))

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        is_training_placeholder = tf.placeholder(tf.bool, name='is_trainning')
        lidarscan_paths_placeholder = tf.placeholder(tf.string, shape=(None, 2), name='lidarscan_paths')
        dists_placeholder = tf.placeholder(tf.float32, shape=(None, 2), name='distances')
        labels_placeholder = tf.placeholder(tf.int64, shape=(None, 2), name='labels')

        # queue operation
        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                              dtypes=[tf.string, tf.int64, tf.float32],
                                              shapes=[(2,), (2,), (2,)],
                                              shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([lidarscan_paths_placeholder, labels_placeholder, dists_placeholder])

        nrof_preprocess_threads = 4
        lidarscans_and_labels_and_distance = []
        for _ in range(nrof_preprocess_threads):
            filenames, labels, dist = input_queue.dequeue()  # shape: (2,) (2,) (2)?
            lidarscans = []

            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                lidarscan = tf.decode_raw(file_contents, tf.float32)
                lidarscan = tf.reshape(lidarscan, [3, 32768])  # for model_cnn.py 64 x 512 = 32768

                # per lidarscan normalization
                mean = tf.reduce_mean(lidarscan, axis=1, keepdims=True)  # ((distance_mean),(reflection_mean))
                lidarscan -= mean

                _, var = tf.nn.moments(lidarscan, axes=1, keep_dims=True)
                std = tf.sqrt(var)  # ((distance_std),(reflection_std))
                lidarscan /= std

                # reshape the normalized lidarscan to (64, 512, 3) for model_cnn.py
                range_data = lidarscan[0]
                reflection_data = lidarscan[1]
                forward_data = lidarscan[2]
                lidarscan = tf.reshape(tf.stack([range_data, reflection_data, forward_data], axis=1), (64, 512, 3))
                lidarscans.append(lidarscan)

                # But in CNN the shape should be (height,width,channel) channel can be 2 if only distance and reflectance
                # are considerd as features. But the original width 2000 pix is not appropriate as input of CNN.
                # So the size of a lidarscan is reduced to 64 x 512 with scan_downsample.py

            lidarscans_and_labels_and_distance.append([lidarscans, labels, dist])

        lidarscan_batch, labels_batch, dist_batch = tf.train.batch_join(
            lidarscans_and_labels_and_distance, batch_size=batch_size_placeholder,  # ！！！！！batch size??
            shapes=[(64, 512, 3), (), ()], enqueue_many=True,
            capacity=8 * nrof_preprocess_threads * args.pairs_per_batch,
            allow_smaller_final_batch=True)
        lidarscan_batch = tf.identity(lidarscan_batch, 'lidarscan_batch')
        lidarscan_batch = tf.identity(lidarscan_batch, 'input')
        labels_batch = tf.identity(labels_batch, 'labels_batch')
        dist_batch = tf.identity(dist_batch, 'distance_batch')

        # Build the inference graph
        prelogits = network.inference(lidarscan_batch, args.keep_probability, weight_decay=args.weight_decay,
                                      is_training=is_training_placeholder,
                                      bottleneck_layer_size=args.embedding_size)
        prelogits = tf.reshape(prelogits, [-1, args.embedding_size])
        embeddings = prelogits
        # embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        # Split embeddings into anchor, positive and negative and calculate triplet loss
        embedding1, embedding2 = tf.unstack(tf.reshape(embeddings, [-1, 2, args.embedding_size]), 2, 1)
        dist_prediction = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(embedding1, embedding2)), 1))
        dist_batch, _ = tf.unstack(tf.reshape(dist_batch, [-1, 2]), 2, 1)
        dist_loss_batch = tf.abs(tf.subtract(dist_prediction, dist_batch))
        distance_loss = tf.reduce_mean(dist_loss_batch, 0)

        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)  # GPU的设置 还没看
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # # Initialize variables
        # sess.run(tf.global_variables_initializer(), feed_dict={is_training_placeholder: True})
        # sess.run(tf.local_variables_initializer(), feed_dict={is_training_placeholder: True})

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)
        # 启动所有graph里的queue runners。batch_join() added a queue runner to the Graph's QUEUE_RUNNER collection

        with sess.as_default():

            if args.pretrained_model:
                print('Restoring pretrained model: %s' % args.pretrained_model)
                saver.restore(sess, os.path.expanduser(args.pretrained_model))
            evaluate(args, sess, pairs_paths, pairs_dists, embeddings, enqueue_op, batch_size_placeholder,
                     lidarscan_paths_placeholder, labels_placeholder, dists_placeholder, args.embedding_size,
                     labels_batch, is_training_placeholder, summary_writer,
                     log_dir, distance_loss, dist_loss_batch, dist_batch)

    return True


def evaluate(args, sess, validation_set, validation_dist, embeddings, enqueue_op, batch_size_placeholder,
             lidarscan_paths_placeholder, labels_placeholder, dists_placeholder, embedding_size, labels_batch,
             is_training_placeholder, summary_writer, log_dir, distance_loss, dist_loss_batch, dist_batch):
    start_time = time.time()

    # Run forward pass to calculate embeddings
    print('Running forward pass on validation set: ', end='')
    nrof_lidarscans = len(validation_set) * 2
    lidarscan_paths_array = np.array(validation_set)
    labels_array = np.reshape(np.arange(nrof_lidarscans), (-1, 2))
    dists_list = []
    for dist in validation_dist:
        dists_list.append(dist)
        dists_list.append(dist)
    dists_array = np.reshape(np.array(dists_list), (-1, 2))
    assert (lidarscan_paths_array.shape[0] == labels_array.shape[0])
    assert (dists_array.shape[0] == labels_array.shape[0])
    sess.run(enqueue_op,
             feed_dict={lidarscan_paths_placeholder: lidarscan_paths_array,
                        labels_placeholder: labels_array,
                        dists_placeholder: dists_array})
    emb_array = np.zeros((nrof_lidarscans, embedding_size))
    validation_loss = []
    batch_size = args.pairs_per_batch * 2
    nrof_batches = int(np.ceil(nrof_lidarscans / batch_size))
    label_check_array = np.zeros((nrof_lidarscans,))
    dist_err_0_10 = []
    dist_err_10_20 = []
    dist_err_20_30 = []
    dist_err_30_40 = []
    dist_err_40_56 = []
    for i in range(nrof_batches):
        batch_size = min(nrof_lidarscans - i * batch_size, batch_size)
        emb, lab, loss, loss_batch, true_dist = sess.run(
            [embeddings, labels_batch, distance_loss, dist_loss_batch, dist_batch],
            feed_dict={batch_size_placeholder: batch_size,
                       is_training_placeholder: False})
        for k in range(len(true_dist)):
            if true_dist[k] < 10:
                dist_err_0_10.append(loss_batch[k])
            elif true_dist[k] < 20:
                dist_err_10_20.append(loss_batch[k])
            elif true_dist[k] < 30:
                dist_err_20_30.append(loss_batch[k])
            elif true_dist[k] < 40:
                dist_err_30_40.append(loss_batch[k])
            else:
                dist_err_40_56.append(loss_batch[k])

        emb_array[lab, :] = emb
        validation_loss.append(loss)
        print(loss)
        label_check_array[lab] = 1
    assert (np.all(label_check_array == 1))  # check all emb computed
    err_0_10 = np.array(dist_err_0_10).mean()
    err_10_20 = np.array(dist_err_10_20).mean()
    err_20_30 = np.array(dist_err_20_30).mean()
    err_30_40 = np.array(dist_err_30_40).mean()
    err_40_56 = np.array(dist_err_40_56).mean()

    evaluate_time = time.time() - start_time
    print('%.3f seconds' % (evaluate_time))

    print('Validation losses: ', validation_loss)
    validation_loss = np.mean(validation_loss)
    print('Average validation loss: %3.3f' % (validation_loss))
    print('err_0_10: %2.3f\tcount: %d' % (err_0_10, len(dist_err_0_10)))
    print('err_10_20: %2.3f\tcount: %d' % (err_10_20, len(dist_err_10_20)))
    print('err_20_30: %2.3f\tcount: %d' % (err_20_30, len(dist_err_20_30)))
    print('err_30_40: %2.3f\tcount: %d' % (err_30_40, len(dist_err_30_40)))
    print('err_40_56: %2.3f\tcount: %d' % (err_40_56, len(dist_err_40_56)))
    # Add loss to summary
    summary = tf.Summary()
    summary.value.add(tag='evaluate/loss', simple_value=validation_loss)
    summary.value.add(tag='time/evaluate', simple_value=evaluate_time)
    summary_writer.add_summary(summary, 0)
    with open(os.path.join(log_dir, 'evaluate_result.txt'), 'at') as f:
        f.write(str(validation_loss))


def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in iteritems(vars(args)):
            f.write('%s: %s\n' % (key, str(value)))


def get_paths_pairs(data_dir, pairs_file):
    pairs_paths = []
    pairs_dists = []
    with open(pairs_file, 'r') as f:
        pairs = f.readlines()
        for i in range(len(pairs)):
            pair = pairs[i].split('#')[0:2]
            pair = [os.path.join(os.path.expanduser(data_dir), filename) for filename in pair]
            dist = float(pairs[i].split('#')[2])
            pairs_paths.append(pair)
            pairs_dists.append(dist)
    assert (len(pairs_paths) == len(pairs_dists))
    idx = np.arange(len(pairs_dists))
    np.random.shuffle(idx)

    pairs_paths_shuffle = [pairs_paths[i] for i in idx]
    pairs_dists_shuffle = [pairs_dists[i] for i in idx]

    return pairs_paths_shuffle, pairs_dists_shuffle


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs.',
                        default='D:/User/Sisi_Tao/Masterarbeit/logs/evaluation_on_map')
    parser.add_argument('--models_base_dir', type=str,
                        help='Directory where to write trained models and checkpoints.',
                        default='D:/User/Sisi_Tao/Masterarbeit/models/train_distance')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
                        help='Load a pretrained model before training starts.',
                        default='../models/train_distance/20190208-153244/model-20190208-153244.ckpt-9760')
    # default='../models/train_distance/20190207-152345/model-20190207-152345.ckpt-40000'
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory.',
                        default='D:/User/Sisi_Tao/downsampled_data_3channels')
    parser.add_argument('--pairs_file', type=str,
                        help='Path to the pairs_file.',
                        default='D:/User/Sisi_Tao/pairs_based_on_map.txt')
    parser.add_argument('--model_def', type=str,
                        help='Model definition. Points to a module containing the definition of the inference graph.',
                        default='model_SqueezeNet')
    parser.add_argument('--nrof_validation_pairs', type=int,
                        help='Number of pairs for validation', default=4096)
    parser.add_argument('--max_nrof_epochs', type=int,
                        help='Number of epochs to run.', default=1000)
    parser.add_argument('--pairs_per_batch', type=int,
                        help='Number of pairs per batch.', default=128)
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=384)
    parser.add_argument('--keep_probability', type=float,
                        help='Keep probability of dropout for the convolutional layer(s).Keep probability of dropout '
                             'for the fully connected layer is set to 1', default=0.7)
    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.', default=0.001)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=665)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
