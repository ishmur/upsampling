import tensorflow as tf


class AbstractOptimizer(object):
    def __init__(self):
        self._optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)

    def minimize_loss(self, loss):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = self._optimizer.minimize(loss=loss)
        return train_op
