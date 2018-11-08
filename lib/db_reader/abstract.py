import os

import numpy as np
import tensorflow as tf

from lib.pipelines.abstract import AbstractPipeline

TF_CONFIG_USE_CPU = '/cpu:0'
NUM_CPU_CORES = os.cpu_count()
SHUFFLE_BUFFER_SIZE = int(10e3)
SHUFFLE_RANDOM_SEED = 12345


class AbstractReader(object):
    def __init__(self, batch_size, num_classes):
        self._batch_size = batch_size
        self._num_classes = num_classes

        x_train, y_train, x_test, y_test = self.load_dataset()

        # make datasets
        with tf.device(TF_CONFIG_USE_CPU):
            dataset_train = self.make_dataset(features=x_train,
                                              labels=y_train,
                                              batch_size=batch_size)
            dataset_test = self.make_dataset(features=x_test,
                                             labels=y_test,
                                             batch_size=batch_size)

        # return iterator object and 'placeholders'
        self._pipeline = AbstractPipeline(dataset=dataset_train)  # can be either train or test

        # return initializer ops
        self.train = self._pipeline.bind_dataset(dataset_train)
        self.test = self._pipeline.bind_dataset(dataset_test)

    @staticmethod
    def load_dataset():
        # load dataset to memory (temporary)
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        return x_train, y_train, x_test, y_test

    def make_dataset(self, features, labels, batch_size):

        def generator():
            for features_, labels_ in zip(features, labels):
                yield features_, labels_

        return tf.data.Dataset.from_generator(
            generator=generator,
            output_types=(features.dtype, labels.dtype),
            output_shapes=(features.shape[1:], labels.shape[1:])
        ).shuffle(
            buffer_size=SHUFFLE_BUFFER_SIZE,
            seed=SHUFFLE_RANDOM_SEED
        ).batch(
            batch_size=batch_size
        ).map(
            map_func=self._preprocess_data,
            num_parallel_calls=NUM_CPU_CORES
        ).prefetch(
            buffer_size=1  # prefetch a single batch, because prefetch() is called after batch()
        )

    def get_tensors(self):
        return self._pipeline.get_tensors()

    """
    Preprocessing
    """

    def _preprocess_data(self, features, labels):
        features_ = tf.reshape(features,
                               shape=(-1, np.prod(features.shape[1:])))
        labels_ = tf.one_hot(tf.cast(labels, dtype=tf.uint8),
                             depth=self._num_classes)
        return features_, labels_
