import tensorflow as tf

from lib.optimizer.abstract import AbstractOptimizer


class AbstractFetches(object):
    def __init__(self, net_inputs, net_output):
        features, classes = net_inputs
        optimizer = self.get_optimizer()

        # training and evaluation ops
        loss = self.get_loss(features, classes, net_output)
        train_op = optimizer.minimize_loss(loss)
        accuracy_op = self.get_accuracy(logits=net_output, classes=classes)
        mean_loss_op = self.get_mean_loss(loss)

        # prepare outputs
        outputs = accuracy_op, mean_loss_op, self.get_probabilities(net_output)
        self.train = train_op, outputs
        self.test = outputs

    @staticmethod
    def get_optimizer():
        return AbstractOptimizer()

    def get_loss(self, features, classes, net_output):
        return self.cross_entropy_loss(logits=net_output,
                                       classes=classes)

    @staticmethod
    def cross_entropy_loss(logits, classes):
        return tf.losses.softmax_cross_entropy(onehot_labels=classes,
                                               logits=logits)

    @staticmethod
    def get_mean_loss(loss):
        # (returns tuple of values: metric value for evaluation, running total since last reset)
        _, mean_loss_op = tf.metrics.mean(values=loss)
        return mean_loss_op

    @staticmethod
    def get_accuracy(logits, classes):
        # (returns tuple of values: metric value for evaluation, running total since last reset)
        _, accuracy_op = tf.metrics.accuracy(labels=tf.argmax(classes, axis=-1),
                                             predictions=tf.argmax(logits, axis=-1))
        return accuracy_op

    @staticmethod
    def get_probabilities(logits):
        return tf.nn.softmax(logits)
