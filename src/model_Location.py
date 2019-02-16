from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


def fully_connected(inputs, keep_probability, is_training=True, weight_decay=0.0):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    }

    with slim.arg_scope([slim.fully_connected],
                        weights_initializer=slim.initializers.xavier_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        with tf.variable_scope('location_net', values=[inputs]):
            net = slim.fully_connected(inputs, 15, scope='fc_1')
            net = slim.dropout(net, keep_probability, is_training=is_training)
            net = slim.fully_connected(net, 20, scope='fc_2')
            net = slim.fully_connected(net, 5, scope='fc_3')
            net = slim.fully_connected(net, 2, activation_fn=None, scope='fc_4')
    return net
