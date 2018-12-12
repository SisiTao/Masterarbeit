import os
import tensorflow as tf


def get_dataset(path):
    dataset = []
    path_exp = os.path.expanduser(path)
    datasets_indices = [path for path in os.listdir(path_exp) if os.path.isdir(os.path.join(path_exp, path))]
    nrof_subdatasets = len(datasets_indices)
    classes_per_subdataset=[]
    for i in range(nrof_subdatasets):
        dataset_name = datasets_indices[i]
        dataset_dir = os.path.join(path_exp, dataset_name)
        classes = [path for path in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, path))]
        classes.sort()
        nrof_classes = len(classes)
        classes_per_subdataset.append(nrof_classes)
        for j in range(nrof_classes):
            class_name = classes[j]
            class_dir = os.path.join(dataset_dir, class_name)
            lidarscans_and_globalegos = os.listdir(class_dir)
            lidarscans = [name for name in lidarscans_and_globalegos if 'scan' in name]
            lidarscan_paths = [os.path.join(class_dir, lid) for lid in lidarscans]
            dataset.append(LidarClass(dataset_name, class_name, lidarscan_paths))

    return dataset,nrof_subdatasets,classes_per_subdataset


class LidarClass():
    """Stores the paths to lidarscans for a given class 其实没有用到class_name"""

    def __init__(self, dataset_name, name, lidarscan_paths):
        self.dataset_name = dataset_name
        self.name = name
        self.lidarscan_paths = lidarscan_paths

    def __len__(self):
        return len(self.lidarscan_paths)


def triplet_loss(anchor, positive, negative, alpha):
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
    return loss


def train(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars,
          log_histograms=True):
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        if optimizer == 'ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer == 'ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer == 'ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer == 'RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer == 'MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')

        grads = opt.compute_gradients(total_loss, update_gradient_vars)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    if log_histograms:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    if log_histograms:
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())  # not used ???????

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readline():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')  # strip 只能删除开头或是结尾的空格或字符
                e = int(par[0])
                if e < epoch:
                    continue
                else:
                    if par[1] == '-':
                        lr = -1
                    else:
                        lr = float(par[1])
                    learning_rate = lr
                    return learning_rate


def _add_loss_summaries(total_loss):
    """Add summaries for losses.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op
