from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


def fire_module(inputs,
                squeeze_depth,
                expand_depth,
                reuse=None,
                scope=None,
                outputs_collections=None):
    with tf.variable_scope(scope, 'fire', [inputs], reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d], outputs_collections=None):
            net = squeeze(inputs, squeeze_depth)
            outputs = expand(net, expand_depth)
            return outputs


def squeeze(inputs, squeeze_depth):
    return slim.conv2d(inputs, squeeze_depth, [1, 1], stride=1, scope='squeeze')


def expand(inputs, expand_depth):
    with tf.variable_scope('expand'):
        e1x1 = slim.conv2d(inputs, expand_depth, [1, 1], stride=1, scope='1x1')
        e3x3 = slim.conv2d(inputs, expand_depth, [3, 3], stride=1, scope='3x3')
        outputs = tf.concat([e1x1, e3x3], axis=3)
        return outputs


def inference(lidarscans, keep_probability, is_training=True, bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
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

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=slim.initializers.xavier_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        with tf.variable_scope('squeezenet_new', values=[lidarscans], reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
                # input: 64 x 512 x 2
                # 64 x 512 x 16
                net = slim.conv2d(lidarscans, 16, [2, 2], stride=[1, 1], scope='conv0')
                # 64 x 256 x 32
                net = slim.conv2d(net, 32, [3, 3], stride=[1, 2], scope='conv1')
                # 64 x 128 x 64
                net = slim.conv2d(net, 64, [3, 3], stride=[1, 2], scope='conv2')
                # 32 x 64 x 64
                net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='maxpool2')
                # 32 x 64 x 128
                net = fire_module(net, 16, 64, scope='fire3')
                # 32 x 64 x 128
                net = fire_module(net, 16, 64, scope='fire4')
                # 32 x 64 x 128
                net = fire_module(net, 16, 64, scope='fire5')
                # 16 x 32 x 128
                net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='maxpool5')
                # 16 x 32 x 256
                net = fire_module(net, 32, 128, scope='fire6')
                # 16 x 32 x 256
                net = fire_module(net, 32, 128, scope='fire7')
                # 8 x 16 x 256
                net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='maxpool7')
                # 8 x 16 x 512
                net = fire_module(net, 64, 256, scope='fire8')
                # 8 x 16 x 512
                net = fire_module(net, 64, 256, scope='fire9')
                # 4 x 8 x 512
                net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='maxpool10')
                # 4 x 4 x 1000
                net = slim.conv2d(net, 1000, [3, 3], stride=[1, 2], scope='conv11')
                # 1 x 1 x 1000
                net = slim.avg_pool2d(net, [4, 4], padding='VALID', scope='avgpool12')
                # 1000
                net = tf.squeeze(net, [1, 2], name='logits')
                # embedding_size
                net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, scope='Bottleneck',
                                           reuse=False)
    return net
