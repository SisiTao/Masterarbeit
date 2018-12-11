from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


def cnn_net(inputs, activation_fn=tf.nn.relu, dropout_keep_prob=0.8, is_training=True, scope=None, reuse=None):
    with tf.variable_scope(scope, 'cnn_net', [inputs], reuse=reuse):
        # 64 x 512 x 16
        net = slim.conv2d(inputs, 16, 3, scope='Conv2d_1_3x3')
        net = activation_fn(net)
        # 64 x 512 x 32
        net = slim.conv2d(net, 32, 3, scope='Conv2d_2_3x3')
        net = activation_fn(net)
        # 32 x 256 x 32
        net = slim.max_pool2d(net, 2, stride=2, padding='SAME', scope='MaxPool_3_2x2')
        # 32 x 128 x 64
        net = slim.conv2d(net, 64, 3, stride=[1, 2], scope='Conv2d_4_3x3')
        net = activation_fn(net)
        net = slim.dropout(net, keep_prob=dropout_keep_prob, is_training=is_training, scope='Dropout_4')
        # 16 x 64 x 64
        net = slim.max_pool2d(net, 3, stride=2, padding='SAME', scope='MaxPool_5_3x3')
        # 16 x 32 x 64
        net = slim.conv2d(net, 64, 3, stride=[1, 2], scope='Conv2d_6_3x3')
        net = slim.dropout(net, keep_prob=dropout_keep_prob, is_training=is_training, scope='Dropout_6')
        # 16 x 16 x 128
        net = slim.conv2d(net, 128, 3, stride=[1, 2], scope='Conv2d_7_3x3')
        net = slim.dropout(net, keep_prob=dropout_keep_prob, is_training=is_training, scope='Dropout_7')
        # 8 x 8 x 128
        net = slim.conv2d(net, 128, 1, stride=2, scope='Conv2d_8_3x3')
        # 128
        net=slim.avg_pool2d(net,8,padding='VALID',scope='MaxPool_9_8x8')
        net=slim.dropout(net,dropout_keep_prob,is_training=is_training,scope='Dropout_9')
        net=slim.fully_connected(net,)
