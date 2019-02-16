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
import argparse
import sys


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

    pairs_paths, pairs_dists = get_paths_pairs(args.data_dir, args.pairs_file)
    nrof_channel = args.nrof_channel
    # paths_pairs: Returns a list of shuffled paths_pairs shape(?,2)
    map_pairs_paths, map_pairs_dists = get_paths_pairs(args.data_dir, args.map_pairs_file)
    nrof_validation_pairs = args.nrof_validation_pairs
    validation_set = map_pairs_paths[0:nrof_validation_pairs]
    validation_dist = map_pairs_dists[0:nrof_validation_pairs]
    train_set = pairs_paths
    train_dist = pairs_dists

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
                lidarscans.append(lidarscan)

                # But in CNN the shape should be (height,width,channel) channel can be 2 if only distance and reflectance
                # are considerd as features. But the original width 2000 pix is not appropriate as input of CNN.
                # So the size of a lidarscan is reduced to 64 x 512 with scan_downsample.py

            lidarscans_and_labels_and_distance.append([lidarscans, labels, dist])

        lidarscan_batch, labels_batch, dist_batch = tf.train.batch_join(
            lidarscans_and_labels_and_distance, batch_size=batch_size_placeholder,  # ！！！！！batch size??
            shapes=[(64, 512, nrof_channel), (), ()], enqueue_many=True,
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
        distance_loss = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(dist_prediction, dist_batch))), 0)
        learning_rate = tf.train.exponential_decay(learning_rate_placehloder, global_step,
                                                   args.learning_rate_decay_epochs * args.epoch_size,
                                                   args.learning_rate_decay_factor, staircase=True)

        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([distance_loss] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = my_net.train(total_loss, global_step, args.optimizer,
                                learning_rate, args.moving_average_decay, tf.global_variables())

        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

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
                _, nan = train(args, args.learning_rate_schedule_file, epoch, train_set, train_dist, sess, enqueue_op,
                               lidarscan_paths_placeholder, labels_placeholder, dists_placeholder,
                               args.embedding_size, embeddings, labels_batch, batch_size_placeholder,
                               is_training_placeholder, learning_rate_placehloder, total_loss, train_op, global_step,
                               summary_writer, regularization_losses, learning_rate)
                if nan == True:
                    break
                # 源代码里的summary_op并没有被sess运行。但是summary_writer里有写sess
                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, model_dir, step, subdir, summary_writer)

                # Evaluate based on validation_set
                evaluate(args, sess, validation_set, validation_dist, embeddings, enqueue_op, batch_size_placeholder,
                         lidarscan_paths_placeholder, labels_placeholder, dists_placeholder, args.embedding_size,
                         labels_batch, learning_rate_placehloder, is_training_placeholder, summary_writer, step,
                         log_dir, distance_loss)

    return model_dir


def train(args, learning_rate_schedule_file, epoch, dataset, dataset_dist, sess, enqueue_op,
          lidarscan_paths_placeholder, labels_placeholder, dists_placeholder, embedding_size, embeddings,
          labels_batch, batch_size_placeholder, is_training_placeholder, learning_rate_placeholder,
          loss, train_op, global_step, summary_writer, reg_loss, learning_rate):
    if args.learning_rate > 0.0:
        lr = args.learning_rate
    else:
        lr = my_net.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

    # Sample pairs randomly from the dataset
    lidarscan_paths, dists = sample_pairs(dataset, dataset_dist, args.pairs_per_batch, args.epoch_size)
    lidarscan_paths_array = np.array(lidarscan_paths)
    dists_list = []
    for dist in dists:
        dists_list.append(dist)
        dists_list.append(dist)
    dists_array = np.reshape(np.array(dists_list), (-1, 2))
    assert (lidarscan_paths_array.shape[1] == 2)
    nrof_examples = args.pairs_per_batch * args.epoch_size * 2
    labels_array = np.reshape(np.arange(nrof_examples), (-1, 2))
    sess.run(enqueue_op,
             feed_dict={lidarscan_paths_placeholder: lidarscan_paths_array,
                        labels_placeholder: labels_array,
                        dists_placeholder: dists_array})

    print('Running forward pass on sampled datascans and Training: ')
    train_time = 0
    emb_array = np.zeros((nrof_examples, embedding_size))
    # loss_array = np.zeros((nrof_triplets,))  # 不是应该nrof_batches吗,而且这个array好像用不上; not used
    summary = tf.Summary()
    step = 0
    nan = False
    for batch_number in range(args.epoch_size):  # all examples are dequeued
        start_time = time.time()
        feed_dict = {batch_size_placeholder: args.pairs_per_batch * 2, learning_rate_placeholder: lr,
                     is_training_placeholder: True}
        err, reg_err, _, step, emb, lab, lr_decay = sess.run(
            [loss, reg_loss, train_op, global_step, embeddings, labels_batch, learning_rate],
            feed_dict=feed_dict)  # dequeue batch_size examples for training
        if str(err) == 'nan':
            print(lidarscan_paths_array, batch_number + 1)
            print(emb)  # check nan. sometimes loss suddenly becomes nan.
            nan = not nan
            break

        emb_array[lab, :] = emb  # 这里其实用不上，因为train_op里已经计算了 not used
        # loss_array[i] = err # not used
        duration = time.time() - start_time
        reg_err = sum(reg_err)
        print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tReg_loss %2.3f' %
              (epoch, batch_number + 1, args.epoch_size, duration, err, reg_err))
        train_time += duration
        summary.value.add(tag='loss', simple_value=err)
        summary.value.add(tag='learning_rate', simple_value=lr_decay)

    summary.value.add(tag='time/train', simple_value=train_time)
    summary_writer.add_summary(summary, step)
    return step, nan


def evaluate(args, sess, validation_set, validation_dist, embeddings, enqueue_op, batch_size_placeholder,
             lidarscan_paths_placeholder, labels_placeholder, dists_placeholder, embedding_size, labels_batch,
             learning_rate_placeholder, is_training_placeholder, summary_writer, step, log_dir, distance_loss):
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
    for i in range(nrof_batches):
        batch_size = min(nrof_lidarscans - i * batch_size, batch_size)
        emb, lab, loss = sess.run([embeddings, labels_batch, distance_loss],
                                  feed_dict={batch_size_placeholder: batch_size,
                                             learning_rate_placeholder: 0.0,
                                             is_training_placeholder: False})  # 没有用到learning_rate啊？？？
        emb_array[lab, :] = emb
        validation_loss.append(loss)
        label_check_array[lab] = 1
    assert (np.all(label_check_array == 1))  # check all emb computed

    evaluate_time = time.time() - start_time
    print('%.3f seconds' % (evaluate_time))

    print('Validation losses: ', validation_loss)
    validation_loss = np.mean(validation_loss)
    print('Average validation loss: %3.3f' % (validation_loss))
    # Add loss to summary
    summary = tf.Summary()
    summary.value.add(tag='evaluate/loss', simple_value=validation_loss)
    summary.value.add(tag='time/evaluate', simple_value=evaluate_time)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir, 'evaluate_result.txt'), 'at') as f:
        f.write('%d\t%.5f\n' % (step, validation_loss))


def sample_pairs(dataset, dataset_dist, pairs_per_batch, epoch_size):
    nrof_pairs = pairs_per_batch * epoch_size
    idx = np.arange(len(dataset))
    np.random.shuffle(idx)
    lidarscan_paths = []
    dists = []
    for i in range(nrof_pairs):
        lidarscan_paths.append(dataset[idx[i]])
        dists.append(dataset_dist[idx[i]])
    return lidarscan_paths, dists


def save_variables_and_metagraph(sess, saver, model_dir, step, model_name, summary_writer):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)


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

    return pairs_paths, pairs_dists


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs.',
                        default='D:/User/Sisi_Tao/Masterarbeit/logs/train_distance')
    parser.add_argument('--models_base_dir', type=str,
                        help='Directory where to write trained models and checkpoints.',
                        default='D:/User/Sisi_Tao/Masterarbeit/models/train_distance')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
                        help='Load a pretrained model before training starts.',
                        default='../models/train_distance/20190213-121503/model-20190213-121503.ckpt-1320')
    # default='../models/train_distance/20190131-120604/model-20190131-120604.ckpt-2840'
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory.',
                        default='D:/User/Sisi_Tao/rotated_downsampled_data')
    parser.add_argument('--nrof_channel', type=int,
                        help='number of channels of dataset',
                        default=2)
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=384)
    parser.add_argument('--pairs_file', type=str,
                        help='Path to the pairs_file.',
                        default='D:/User/Sisi_Tao/pairs_file_new.txt')
    parser.add_argument('--map_pairs_file', type=str,
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
    parser.add_argument('--epoch_size', type=int,
                        help='Number of batches per epoch.', default=40)
    parser.add_argument('--keep_probability', type=float,
                        help='Keep probability of dropout for the convolutional layer(s).Keep probability of dropout '
                             'for the fully connected layer is set to 1', default=0.7)
    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.', default=0.001)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
                        help='Initial learning rate. If set to a negative value a learning rate ' +
                             'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.0004)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
                        help='Number of epochs between learning rate decay.(not used)', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
                        help='Learning rate decay factor.(not used)', default=0.2)
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=665)
    parser.add_argument('--learning_rate_schedule_file', type=str,
                        help='learning_rate_schedule_file,not used', default=' ')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
