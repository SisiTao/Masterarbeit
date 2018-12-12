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

    data_set, nrof_subdatasets, classes_per_subdataset = my_net.get_dataset(args.data_dir)
    # data_set: Returns a list of LidarClass(dataset_idx, class_name, lidarscan_paths)
    # nrof_subdatasets: How many subdatasets in the dataset
    # classes_per_subdataset: How many classes or places in each subdataset

    validation_set = data_set[0:classes_per_subdataset[0]]
    train_set = data_set[classes_per_subdataset[0]:len(data_set)]
    classes_per_subdataset = classes_per_subdataset[1:nrof_subdatasets]
    nrof_subdatasets -= 1
    # The validation_set uses the lidarscans in the first subdataset

    # prepare the positive and negative pairs and their labels of validation_set
    validation_paths = []
    actual_issame = []

    # pairs contain lidarscans from the same class/place: 4 pairs in each class (the first 4 lidarscans in one class)
    for i in range(len(validation_set)):
        validation_paths += (validation_set[i].lidarscan_paths[0], validation_set[i].lidarscan_paths[1])
        validation_paths += (validation_set[i].lidarscan_paths[1], validation_set[i].lidarscan_paths[2])
        validation_paths += (validation_set[i].lidarscan_paths[2], validation_set[i].lidarscan_paths[3])
        validation_paths += (validation_set[i].lidarscan_paths[3], validation_set[i].lidarscan_paths[0])
        # 8 paths are appended
    actual_issame += [True] * (len(validation_set) * 4)
    # pairs contain lidarscans from the different classes/places: 4 pairs for each class
    for j in range(2):
        idx_pairs = []
        for class0_idx in range(len(validation_set)):  # class0: the anchor class
            class1_idx = class2_idx = 0  # class1 and class2: the negative class
            while class1_idx == class2_idx or class1_idx == class0_idx or class2_idx == class0_idx:  # avoid the same class
                class1_idx, class2_idx = np.random.randint(len(validation_set), size=2)
                if (class1_idx, class0_idx) in idx_pairs:  # avoid repeatment
                    class1_idx = np.random.randint(len(validation_set))
                if (class2_idx, class0_idx) in idx_pairs:
                    class2_idx = np.random.randint(len(validation_set))
            validation_paths += (
                validation_set[class0_idx].lidarscan_paths[j], validation_set[class1_idx].lidarscan_paths[j])
            validation_paths += (
                validation_set[class0_idx].lidarscan_paths[j], validation_set[class2_idx].lidarscan_paths[j])
            # take the first(j=0) or second(j=1) lidarscans of each class as anchor-negative pairs
            idx_pairs += ((class0_idx, class1_idx), (class0_idx, class2_idx))
    actual_issame += [False] * (len(validation_set) * 4)
    assert (len(validation_paths) == (len(actual_issame) * 2))

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
        lidarscan_paths_placeholder = tf.placeholder(tf.string, shape=(None, 3), name='lidarscan_paths')
        labels_placeholder = tf.placeholder(tf.int64, shape=(None, 3), name='labels')

        # queue operation
        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                              dtypes=[tf.string, tf.int64],
                                              shapes=[(3,), (3,)],
                                              shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([lidarscan_paths_placeholder, labels_placeholder])

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)  # GPU的设置 还没看
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        nrof_preprocess_threads = 4
        lidarscans_and_labels_and_coordinates = []
        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue.dequeue()  # shape: (3,) (3,) ?
            lidarscans = []
            coordinates = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                lidarscan = tf.decode_raw(file_contents, tf.float32)
                lidarscan = tf.reshape(lidarscan, [2, 128000])
                coordinate_filename = tf.strings.regex_replace(filename, 'velodyne_scan', 'global_ego_data')
                coordinate = tf.decode_raw(tf.read_file(coordinate_filename), tf.float64)[16:18]
                # easting and northing coordinate

                # per lidarscan normalization
                mean = tf.reduce_mean(lidarscan, axis=1, keepdims=True)  # ((distance_mean),(reflection_mean))
                lidarscan -= mean

                _, var = tf.nn.moments(lidarscan, axes=1, keep_dims=True)
                std = tf.sqrt(var)  # ((distance_std),(reflection_std))
                lidarscan /= std

                # reshape the normalized lidarscan to [dis,ref,dis,ref...]
                distance_data = lidarscan[0, :]
                reflectance_data = lidarscan[1, :]
                lidarscan = tf.reshape(tf.stack([distance_data, reflectance_data], axis=1), (256000,))

                lidarscans.append(lidarscan)
                coordinates.append(coordinate)
                # Here the lidarscans are turned into one-d array with 256000 elements as the input of network.
                # But in CNN the shape should be (height,width,channel) channel can be 2 if only distance and reflection
                # are considerd as features. But the original width 2000 pix is not appropriate as input of CNN.
                # How to turn it to 1024 or 2048 ?
                # 这里进行data的大小剪裁，在卷积网络里应该要修改，比如宽度变为1024或2048，而不是2000
            lidarscans_and_labels_and_coordinates.append([lidarscans, label, coordinates])
        # 4*3 lidarscans,labels,coors after loop

        lidarscan_batch, labels_batch, coordinates_batch = tf.train.batch_join(
            lidarscans_and_labels_and_coordinates, batch_size=batch_size_placeholder,  # ！！！！！batch size??
            shapes=[(256000,), (), (2,)], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)
        lidarscan_batch = tf.identity(lidarscan_batch, 'lidarscan_batch')
        lidarscan_batch = tf.identity(lidarscan_batch, 'input')
        labels_batch = tf.identity(labels_batch, 'labels_batch')
        coordinates_batch = tf.identity(coordinates_batch, 'coordinates_batch')

        # Build the inference graph
        prelogits, reg_term = network.inference(lidarscan_batch, args.keep_probability,
                                                is_training=is_training_placeholder,
                                                bottleneck_layer_size=args.embedding_size,
                                                )  # weight_decay=args.weight_decay deleted

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        # Split embeddings into anchor, positive and negative and calculate triplet loss
        anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1, 3, args.embedding_size]), 3, 1)
        triplet_loss = my_net.triplet_loss(anchor, positive, negative, args.alpha)

        learning_rate = learning_rate_placehloder  # 源代码用了exponential_decay; the original code used exponential_decay

        # Calculate the total losses
        # for model.py
        # regularization_losses = tf.convert_to_tensor(reg_term)  # temporary used for simple network
        # total_loss = tf.add_n([triplet_loss] + [regularization_losses],
        #                       name='total_loss')  # Here the reg_loss is a value, so turn it to a list

        # for model_cnn.py
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([triplet_loss] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = my_net.train(total_loss, global_step, args.optimizer,
                                learning_rate, args.moving_average_decay, tf.global_variables())

        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()  # not used

        # Initialize variables
        sess.run(tf.global_variables_initializer(), feed_dict={is_training_placeholder: True})
        sess.run(tf.local_variables_initializer(), feed_dict={is_training_placeholder: True})

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)  # 启动所有graph里的queue runners。但是没找到？

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
                      summary_writer)
                # 源代码里的summary_op并没有被sess运行。但是summary_writer里有写sess
                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)

                # Evaluate based on validation_set
                evaluate(sess, validation_paths, embeddings, enqueue_op, batch_size_placeholder,
                         lidarscan_paths_placeholder, labels_placeholder, args.embedding_size,
                         labels_batch, learning_rate_placehloder, is_training_placeholder, args.batch_size,
                         summary_writer, step, log_dir, actual_issame, args.nrof_folds)

    return model_dir


def train(args, learning_rate_schedule_file, epoch, dataset, nrof_subdatasets, classes_per_subdataset, sess, enqueue_op,
          lidarscan_paths_placeholder,
          labels_placeholder, embedding_size, embeddings, coordinates_batch, labels_batch, batch_size_placeholder,
          is_training_placeholder,
          learning_rate_placeholder, loss, train_op, global_step, summary_writer):
    batch_number = 0

    if args.learning_rate > 0.0:
        lr = args.learning_rate
    else:
        lr = my_net.get_learning_rate_from_file(learning_rate_schedule_file, epoch)
    while batch_number < args.epoch_size:
        # Sample places(classes) randomly from the dataset (from the same subdataset)
        lidarscan_paths, num_per_class = sample_places(dataset, nrof_subdatasets, classes_per_subdataset,
                                                       args.places_per_batch, args.lidarscans_per_place)

        print('Running forward pass on sampled datascans: ', end='')
        start_time = time.time()
        nrof_examples = args.places_per_batch * args.lidarscans_per_place
        places_per_batch = len(num_per_class)
        # Here the number of places_per_batch is not equal to args.places_per_batch.
        # Because some lidarscans_per_place may less than args.lidarscans_per_place

        labels_array = np.reshape(np.arange(nrof_examples), (-1, 3))
        lidarscan_paths_array = np.reshape(np.expand_dims(np.array(lidarscan_paths), 1),
                                           (-1, 3))  # why not directly use reshape
        sess.run(enqueue_op,
                 feed_dict={lidarscan_paths_placeholder: lidarscan_paths_array, labels_placeholder: labels_array})
        coordinates_array = np.zeros((nrof_examples, 2))
        nrof_batches = int(np.ceil(nrof_examples / args.batch_size))
        for i in range(nrof_batches):
            batch_size = min(nrof_examples - nrof_batches * args.batch_size, args.batch_size)
            coor, lab = sess.run([coordinates_batch, labels_batch],
                                 feed_dict={batch_size_placeholder: batch_size, is_training_placeholder: True,
                                            learning_rate_placeholder: lr})
            coordinates_array[lab, :] = coor  # sort the coordinates according to the label
        print('%.3f' % (time.time() - start_time))

        # Select triplets based on distances
        print('Selecting suitable triplets for training')
        start_time = time.time()
        triplets, nrof_random_negs, nrof_triplets = select_triplets(coordinates_array, num_per_class,
                                                                    lidarscan_paths, places_per_batch, args.alpha)
        selection_time = time.time() - start_time
        print('(nrof_random_negs, nrof_triplets) = (%d, %d): time=%.3f seconds' % (
            nrof_random_negs, nrof_triplets, selection_time))

        # Perform training on the selected triplets
        nrof_batches = int(np.ceil(nrof_triplets * 3 / args.batch_size))
        triplet_paths = list(itertools.chain(*triplets))
        nrof_examples = len(triplet_paths)
        labels_array = np.reshape(np.arange(nrof_examples), (-1, 3))
        triplet_paths_array = np.reshape(np.expand_dims(np.array(triplet_paths), 1), (-1, 3))
        sess.run(enqueue_op,  # input nrof_examples lidarscans in queue
                 feed_dict={lidarscan_paths_placeholder: triplet_paths_array, labels_placeholder: labels_array})
        train_time = 0
        emb_array = np.zeros((nrof_examples, embedding_size))
        # loss_array = np.zeros((nrof_triplets,))  # 不是应该nrof_batches吗,而且这个array好像用不上; not used
        summary = tf.Summary()
        step = 0
        for i in range(nrof_batches):  # all examples are dequeued
            start_time = time.time()
            batch_size = min(nrof_examples - i * args.batch_size, args.batch_size)
            feed_dict = {batch_size_placeholder: batch_size, learning_rate_placeholder: lr,
                         is_training_placeholder: True}
            err, _, step, emb, lab = sess.run([loss, train_op, global_step, embeddings, labels_batch],
                                              feed_dict=feed_dict)  # dequeue batch_size examples for training
            emb_array[lab, :] = emb  # 这里其实用不上，因为train_op里已经计算了 not used
            # loss_array[i] = err # not used
            duration = time.time() - start_time
            print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f' %
                  (epoch, batch_number + 1, args.epoch_size, duration, err))
            batch_number += 1
            i += 1
            train_time += duration
            summary.value.add(tag='loss', simple_value=err)

        summary.value.add(tag='time/selection', simple_value=selection_time)
        summary.value.add(tag='time/train', simple_value=selection_time)
        summary_writer.add_summary(summary, step)
    return step


def evaluate(sess, validation_paths, embeddings, enqueue_op, batch_size_placeholder, lidarscan_paths_placeholder,
             labels_placeholder,
             embedding_size, labels_batch, learning_rate_placeholder, is_training_placeholder, batch_size,
             summary_writer, step, log_dir, actual_issame, nrof_folds):
    start_time = time.time()

    # Run forward pass to calculate embeddings
    print('Running forward pass on validation set: ', end='')
    nrof_lidarscans = len(validation_paths)
    labels_array = np.reshape(np.arange(nrof_lidarscans), (-1, 3))
    lidarscan_paths_array = np.reshape(np.expand_dims(np.array(validation_paths), 1), (-1, 3))
    sess.run(enqueue_op,
             feed_dict={lidarscan_paths_placeholder: lidarscan_paths_array, labels_placeholder: labels_array})
    emb_array = np.zeros((nrof_lidarscans, embedding_size))
    nrof_batches = int(np.ceil(nrof_lidarscans / batch_size))
    for i in range(nrof_batches):
        batch_size = min(nrof_lidarscans - i * batch_size, batch_size)
        emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: nrof_lidarscans,
                                                                   learning_rate_placeholder: 0.0,
                                                                   is_training_placeholder: False})  # 没有用到learning_rate啊？？？
        emb_array[lab, :] = emb  # 省略了 label_check_array
    print('%.3f' % (time.time() - start_time))

    # calculate accuracy (cross-validation: an array of accuracy of different folds )
    accuracy = calculate_accuracy(emb_array, actual_issame, nrof_folds)
    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    evaluate_time = time.time() - start_time
    # Add accuracy to summary
    summary = tf.Summary()
    summary.value.add(tag='evaluate/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag='time/evaluate', simple_value=evaluate_time)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir, 'evaluate_result.txt'), 'at') as f:
        f.write('%d\t%.5f\n' % (step, np.mean(accuracy)))


def calculate_accuracy(embeddings, actual_issame, nrof_folds=10):
    embeddings1 = embeddings[0::2]  # The embeddings of the first lidarscan in each pair
    embeddings2 = embeddings[1::2]  # The embeddings of the second lidarscan in each pair
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])

    nrof_pairs = len(actual_issame)  # 源代码用了actual_issame和embeddings1.shape[0]的最小值，但其实是一样的呀？
    indices = np.arange(nrof_pairs)
    thresholds = np.arange(0, 4, 0.01)  # 这个范围怎么确定的？how to determine this range?
    nrof_thresholds = len(thresholds)

    distance = np.sum(np.square(np.subtract(embeddings1, embeddings2)), 1)
    assert (len(distance) == len(actual_issame))
    accuracy = np.zeros((nrof_folds))

    k_fold = KFold(n_splits=nrof_folds, shuffle=True)
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best threshold for the fold using train_set
        accuracy_train = np.zeros((nrof_thresholds))
        for i, threshold in enumerate(thresholds):
            predict_issame_train = np.less(distance[train_set], threshold)
            tp_train = np.sum(np.logical_and(predict_issame_train, actual_issame[train_set]))
            tn_train = np.sum(
                np.logical_and(np.logical_not(predict_issame_train), np.logical_not(actual_issame[train_set])))
            accuracy_train[i] = float(tp_train + tn_train) / distance[train_set].size
        best_threshold_test = np.argmax(accuracy_train)
        # define two lidarscans with distance smaller than threshold as same
        predict_issame_test = np.less(distance[test_set], thresholds[best_threshold_test])
        tp_test = np.sum(np.logical_and(predict_issame_test, actual_issame[test_set]))
        tn_test = np.sum(np.logical_and(np.logical_not(predict_issame_test), np.logical_not(actual_issame[test_set])))
        accuracy[fold_idx] = float(tp_test + tn_test) / distance[test_set].size
    return accuracy


def select_triplets(coordinates, nrof_lidarscans_per_place, lidarscan_paths, places_per_batch, alpha):
    triplets = []
    start_idx = 0

    for i in range(places_per_batch):
        nrof_lidarscans = int(nrof_lidarscans_per_place[i])  # lidarscans around the same place
        for j in range(1, nrof_lidarscans):
            a_idx = start_idx + j - 1  # anchor
            neg_dists = np.sqrt(np.sum(np.square(coordinates[a_idx] - coordinates), 1))
            neg_dists[start_idx:start_idx + nrof_lidarscans] = np.NaN
            for pair in range(j, nrof_lidarscans):
                p_idx = start_idx + pair  # all the other lidarscans in the same class are combined with anchor as positive pairs
                pos_dist = np.sqrt(np.sum(np.square(coordinates[a_idx] - coordinates[p_idx])))
                all_neg = np.where(neg_dists - pos_dist < alpha)[0]
                # the negative distances should not be too big(hard negative, faster convergence
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs > 0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((lidarscan_paths[a_idx], lidarscan_paths[p_idx], lidarscan_paths[n_idx]))
        start_idx += nrof_lidarscans
    return triplets, nrof_random_negs, len(triplets)


def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in iteritems(vars(args)):
            f.write('%s: %s\n' % (key, str(value)))


def sample_places(dataset, nrof_subdatasets, classes_per_subdataset, places_per_batch, lidarscans_per_place):
    nrof_lidarscans = places_per_batch * lidarscans_per_place

    # Choose a random subdataset among the dataset
    # Because the positive and negative pairs can only selected inside a common subdataset
    dataset_idx_random = np.random.randint(nrof_subdatasets)
    start_idx = 0
    if dataset_idx_random == 0:
        start_idx = 0
    else:
        for d in range(dataset_idx_random):
            start_idx += classes_per_subdataset[d]

    end_idx = start_idx + classes_per_subdataset[dataset_idx_random]
    subdataset_random = dataset[start_idx:end_idx]

    # Sample classes from the subdataset

    nrof_classes = len(subdataset_random)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)

    i = 0
    lidarscan_paths = []
    num_per_class = []

    # Sample lidarscans from these classes until we have enough
    while len(lidarscan_paths) < nrof_lidarscans:
        class_index = class_indices[i]
        nrof_lidarscans_in_class = len(subdataset_random[class_index])
        lidarscan_indices = np.arange(nrof_lidarscans_in_class)
        np.random.shuffle(lidarscan_indices)
        nrof_lidarscans_from_class = min(lidarscans_per_place, nrof_lidarscans_in_class,
                                         nrof_lidarscans - len(lidarscan_paths))
        idx = lidarscan_indices[0:nrof_lidarscans_from_class]
        lidarscan_paths_for_class = [subdataset_random[class_index].lidarscan_paths[j] for j in idx]
        lidarscan_paths += lidarscan_paths_for_class
        num_per_class.append(nrof_lidarscans_from_class)
        i += 1

    return lidarscan_paths, num_per_class


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


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs.', default='../logs/my_net')
    parser.add_argument('--models_base_dir', type=str,
                        help='Directory where to write trained models and checkpoints.', default='../models/my_net')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
                        help='Load a pretrained model before training starts.')
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory containing aligned face patches.',
                        default='../dataset/divided_data_0')
    parser.add_argument('--model_def', type=str,
                        help='Model definition. Points to a module containing the definition of the inference graph.',
                        default='model')
    parser.add_argument('--max_nrof_epochs', type=int,
                        help='Number of epochs to run.', default=500)
    parser.add_argument('--batch_size', type=int,
                        help='Number of lidar_scans to process in a batch.(used in evaluation)', default=50)
    # parser.add_argument('--image_size', type=int,
    #                   help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--places_per_batch', type=int,
                        help='Number of places(or classes) per batch.', default=20)
    parser.add_argument('--lidarscans_per_place', type=int,
                        help='Number of lidarscans per place.', default=10)
    parser.add_argument('--epoch_size', type=int,
                        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--alpha', type=float,
                        help='Positive to negative triplet distance margin.', default=0.2)
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=128)
    # parser.add_argument('--random_crop',
    #                     help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
    #                          'If the size of the images in the data directory is equal to image_size no cropping is performed',
    #                     action='store_true')
    # parser.add_argument('--random_flip',
    #                     help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
                        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
                        help='Initial learning rate. If set to a negative value a learning rate ' +
                             'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
                        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
                        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--learning_rate_schedule_file', type=str,
                        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.',
                        default='data/learning_rate_schedule.txt')
    parser.add_argument('--nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
