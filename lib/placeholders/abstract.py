import tensorflow as tf


class AbstractPlaceholders(object):
    def __init__(self):
        self.is_training = tf.placeholder(dtype=tf.bool,
                                          shape=(),
                                          name='is_training')

    def get_train(self):
        return {
            self.is_training: True
        }

    def get_test(self):
        return {
            self.is_training: False
        }
