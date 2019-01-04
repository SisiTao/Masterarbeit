from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


def cnn_net(inputs,bottleneck_layer_size=128,dropout_keep_prob=0.8, is_training=True, scope='cnn_net', reuse=None):
    with tf.variable_scope(scope, 'cnn_net', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm,slim.dropout],is_training=is_training):
            # 64 x 512 x 16
            net = slim.conv2d(inputs, 16, 3, scope='Conv2d_1_3x3')
            # 64 x 512 x 32
            net = slim.conv2d(net, 32, 3, scope='Conv2d_2_3x3')
            # 32 x 256 x 32
            net = slim.max_pool2d(net, 2, stride=2, padding='SAME', scope='MaxPool_3_2x2')
            # 32 x 128 x 64
            net = slim.conv2d(net, 64, 3, stride=[1, 2], scope='Conv2d_4_3x3')
            net = slim.dropout(net, keep_prob=dropout_keep_prob, is_training=is_training, scope='Dropout_4')
            # 16 x 64 x 64
            net = slim.max_pool2d(net, 3, stride=2, padding='SAME', scope='MaxPool_5_3x3')
            # 16 x 32 x 64
            net = slim.conv2d(net, 64, 3, stride=[1, 2], scope='Conv2d_6_3x3')
            net = slim.dropout(net, keep_prob=dropout_keep_prob, is_training=is_training, scope='Dropout_6')
            # 16 x 16 x 128
            net = slim.conv2d(net, 128, 3, stride=[1, 2], scope='Conv2d_7_3x3')
            net = slim.dropout(net, keep_prob=dropout_keep_prob, is_training=is_training, scope='Dropout_7')
            # 8 x 8 x 256
            net = slim.conv2d(net, 256, 2, stride=2, scope='Conv2d_8_3x3')
            # 256
            net=slim.avg_pool2d(net,8,padding='VALID',scope='AvgPool_9_8x8')
            net=slim.dropout(net,keep_prob=1.0,is_training=is_training,scope='Dropout_9')
            #128
            net=slim.fully_connected(net,bottleneck_layer_size,activation_fn=None,scope='Bottleneck',reuse=False)
    return net

def inference(lidarscans,keep_probability,is_training=True,bottleneck_layer_size=128,weight_decay=0.0,reuse=None):
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
        return cnn_net(lidarscans, is_training=is_training,
              dropout_keep_prob=keep_probability, bottleneck_layer_size=bottleneck_layer_size, reuse=reuse)
