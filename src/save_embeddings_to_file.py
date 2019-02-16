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
import json


def main(args):
    network = importlib.import_module(args.model_def)

    # create log directory
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)

    # Write arguments to a text file
    write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))

    np.random.seed(seed=args.seed)
    if args.map_file:
        lidarscan_paths = get_paths(args.data_dir, args.map_file)
    else:
        lidarscan_filenames = [name for name in os.listdir(os.path.expanduser(args.data_dir)) if 'scan' in name]
        lidarscan_paths = [os.path.join(args.data_dir, name) for name in lidarscan_filenames]
    nrof_channel = args.nrof_channel
    print('Log directory: %s' % log_dir)
    if args.pretrained_model:
        print('Pre-trained model: %s' % os.path.expanduser(args.pretrained_model))

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        is_training_placeholder = tf.placeholder(tf.bool, name='is_trainning')
        lidarscan_paths_placeholder = tf.placeholder(tf.string, shape=(None), name='lidarscan_paths')
        labels_placeholder = tf.placeholder(tf.int64, shape=(None), name='labels')

        # queue operation
        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                              dtypes=[tf.string, tf.int64],
                                              shapes=[(), ()],
                                              shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([lidarscan_paths_placeholder, labels_placeholder])

        nrof_preprocess_threads = 4
        lidarscans_and_labels = []
        for _ in range(nrof_preprocess_threads):
            path, labels = input_queue.dequeue()
            file_contents = tf.read_file(path)
            lidarscan = tf.decode_raw(file_contents, tf.float32)
            lidarscan = tf.reshape(lidarscan, [nrof_channel, 32768])  # for model_cnn.py 64 x 512 = 32768

            # per lidarscan normalization
            mean = tf.reduce_mean(lidarscan, axis=1, keepdims=True)  # ((distance_mean),(reflection_mean))
            lidarscan -= mean

            _, var = tf.nn.moments(lidarscan, axes=1, keep_dims=True)
            std = tf.sqrt(var)  # ((distance_std),(reflection_std))
            lidarscan /= std

            # reshape the normalized lidarscan to (64, 512, 3) for model_cnn.py
            if nrof_channel == 2:
                range_data = lidarscan[0]
                reflection_data = lidarscan[1]
                lidarscan = tf.reshape(tf.stack([range_data, reflection_data], axis=1), (64, 512, nrof_channel))
            if nrof_channel == 3:
                range_data = lidarscan[0]
                reflection_data = lidarscan[1]
                forward_data = lidarscan[2]
                lidarscan = tf.reshape(tf.stack([range_data, reflection_data, forward_data], axis=1),
                                       (64, 512, nrof_channel))
            lidarscans_and_labels.append([lidarscan, labels])

        lidarscan_batch, labels_batch = tf.train.batch_join(
            lidarscans_and_labels, batch_size=batch_size_placeholder,  # ！！！！！batch size??
            shapes=[(64, 512, nrof_channel), ()], enqueue_many=False,
            capacity=8 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)
        lidarscan_batch = tf.identity(lidarscan_batch, 'lidarscan_batch')
        lidarscan_batch = tf.identity(lidarscan_batch, 'input')
        labels_batch = tf.identity(labels_batch, 'labels_batch')

        # Build the inference graph
        prelogits = network.inference(lidarscan_batch, args.keep_probability, weight_decay=args.weight_decay,
                                      is_training=is_training_placeholder,
                                      bottleneck_layer_size=args.embedding_size)
        prelogits = tf.reshape(prelogits, [-1, args.embedding_size])
        embeddings = prelogits
        # embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)  # GPU的设置 还没看
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # Initialize variables
        sess.run(tf.global_variables_initializer(), feed_dict={is_training_placeholder: True})
        sess.run(tf.local_variables_initializer(), feed_dict={is_training_placeholder: True})
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)
        # 启动所有graph里的queue runners。batch_join() added a queue runner to the Graph's QUEUE_RUNNER collection

        with sess.as_default():

            if args.pretrained_model:
                print('Restoring pretrained model: %s' % args.pretrained_model)
                saver.restore(sess, os.path.expanduser(args.pretrained_model))
                evaluate(args, sess, lidarscan_paths, embeddings, enqueue_op, batch_size_placeholder,
                         lidarscan_paths_placeholder, labels_placeholder, args.embedding_size,
                         labels_batch, is_training_placeholder)

    return True


def evaluate(args, sess, lidarscan_paths, embeddings, enqueue_op, batch_size_placeholder,
             lidarscan_paths_placeholder, labels_placeholder, embedding_size, labels_batch,
             is_training_placeholder):
    start_time = time.time()

    # Run forward pass to calculate embeddings
    print('Running forward pass on validation set: ', end='')
    nrof_lidarscans = len(lidarscan_paths)
    lidarscan_paths_array = np.array(lidarscan_paths)
    labels_array = np.arange(nrof_lidarscans)
    assert (lidarscan_paths_array.shape[0] == labels_array.shape[0])
    sess.run(enqueue_op,
             feed_dict={lidarscan_paths_placeholder: lidarscan_paths_array,
                        labels_placeholder: labels_array})
    emb_array = np.zeros((nrof_lidarscans, embedding_size))
    batch_size = args.batch_size
    nrof_batches = int(np.ceil(nrof_lidarscans / batch_size))
    label_check_array = np.zeros((nrof_lidarscans,))
    for i in range(nrof_batches):
        batch_size = min(nrof_lidarscans - i * batch_size, batch_size)
        emb, lab = sess.run([embeddings, labels_batch],
                            feed_dict={batch_size_placeholder: batch_size,
                                       is_training_placeholder: False})
        emb_array[lab, :] = emb
        label_check_array[lab] = 1
    assert (np.all(label_check_array == 1))  # check all emb computed
    embeddings_dict = {}
    emb_list = emb_array.tolist()
    print('length of embeddings: ', len(emb_array))
    for idx in labels_array:
        filename = lidarscan_paths[idx].split('/')[-1]
        embeddings_dict[filename] = emb_list[idx]
    with open(args.embeddings_file, 'w') as f:
        embeddings_dict_dump = json.dumps(embeddings_dict)
        f.write(embeddings_dict_dump)

    evaluate_time = time.time() - start_time
    print('%.3f seconds' % (evaluate_time))


def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in iteritems(vars(args)):
            f.write('%s: %s\n' % (key, str(value)))


def get_paths(data_dir, map_file):
    paths = []
    with open(map_file, 'r') as f:
        line = f.readline()
        while line:
            filename = line.split(':')[0].replace('global_ego_data', 'velodyne_scan')
            path = os.path.expanduser(os.path.join(data_dir, filename))
            paths.append(path)
            line = f.readline()
    return paths


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs.',
                        default='D:/User/Sisi_Tao/Masterarbeit/logs/save_embeddings_to_file')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
                        help='Load a pretrained model before training starts.',
                        default='../models/train_distance/20190211-144544/model-20190211-144544.ckpt-560')
    # default='../models/train_distance/20190131-120604/model-20190131-120604.ckpt-2840'
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory.',
                        default='D:/User/Sisi_Tao/downsampled_data')
    parser.add_argument('--nrof_channel', type=int,
                        help='number of channels of dataset',
                        default=2)
    parser.add_argument('--map_file', type=str,
                        help='map file. Should be None if calculate on dataset')
    # default='D:/User/Sisi_Tao/map_file.txt'
    parser.add_argument('--embeddings_file', type=str,
                        help='embeddings file',
                        default='D:/User/Sisi_Tao/embeddings_files/embeddings_file_downsampled_data.txt')
    parser.add_argument('--model_def', type=str,
                        help='Model definition. Points to a module containing the definition of the inference graph.',
                        default='model_SqueezeNet')
    parser.add_argument('--batch_size', type=int,
                        help='Number of pairs per batch.', default=256)
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--keep_probability', type=float,
                        help='Keep probability of dropout for the convolutional layer(s).Keep probability of dropout '
                             'for the fully connected layer is set to 1', default=0.7)
    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.', default=0.001)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=665)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
