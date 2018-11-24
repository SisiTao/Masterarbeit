import tensorflow as tf


def add_layer(inputs, in_size, out_size, layername, keep_prob, activation_function=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    wx_plus_b = tf.matmul(inputs, weights) + biases
    wx_plus_b = tf.nn.dropout(wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b, )
    tf.summary.histogram(layername + 'outputs', outputs)
    return outputs, weights


def model_net(inputs, dropout_keep_prob=0.8, is_training=True, bottleneck_layer_size=128):
    if is_training is False:
        dropout_keep_prob = 1.0
    net, weights_1 = add_layer(inputs, 256000, 10000, 'layer1', dropout_keep_prob, activation_function=tf.nn.relu)
    net, weights_2 = add_layer(net, 10000, 500, 'layer2', dropout_keep_prob, activation_function=tf.nn.relu)
    net, weights_3 = add_layer(net, 500, bottleneck_layer_size, 'layer3', dropout_keep_prob, activation_function=None)
    reg_term = 0.1 * (tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2) + tf.nn.l2_loss(weights_3))
    return net, reg_term
