import tensorflow as tf


class AbstractNet(object):
    def __init__(self, inputs, placeholders):
        self.output = self._build_architecture(inputs, placeholders)

    @staticmethod
    def _build_architecture(inputs, placeholders):
        # unpack iterator
        features, classes = inputs

        # build net
        net = tf.layers.dense(inputs=features,
                              units=512,
                              activation=tf.nn.relu)
        net = tf.layers.dropout(inputs=net,
                                rate=0.2,
                                training=placeholders.is_training)
        logits = tf.layers.dense(inputs=net,
                                 units=10)
        return logits
