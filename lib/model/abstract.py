import functools

import tensorflow as tf

from lib.db_reader.abstract import AbstractReader
from lib.placeholders.abstract import AbstractPlaceholders
from lib.net.abstract import AbstractNet
from lib.fetches.abstract import AbstractFetches
from lib.utils.timer import Timer


class AbstractModel(object):
    def __init__(self, batch_size, num_epochs, num_classes):
        self.num_epochs = num_epochs
        self.db_reader = AbstractReader(batch_size, num_classes)
        inputs = self.db_reader.get_tensors()
        self.placeholders = AbstractPlaceholders()
        self.net = AbstractNet(inputs=inputs,
                               placeholders=self.placeholders)
        self.fetches = AbstractFetches(inputs,
                                       net_output=self.net.output)

        self._sess = None

    """
    Session management
    """

    @property
    def sess(self):
        return self._sess

    def _check_session(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if type(self.sess) == tf.Session:
                return func(self, *args, **kwargs)
            else:
                raise AttributeError('tf.Session not started.')
        return wrapper

    def __enter__(self):
        self._sess = tf.Session().__enter__()
        self._initialize_variables()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._sess.__exit__(exc_type, exc_val, exc_tb)
        self._sess = None

    """
    Training
    """

    @Timer.time
    @_check_session
    def train(self):
        print('Training...')
        for epoch_idx in range(self.num_epochs):
            self._train_epoch()

    def _train_epoch(self):
        results = None
        self._reset_train_pipeline()
        self._reset_metrics()
        while True:
            try:
                _, results = self._sess.run(self.fetches.train,
                                            feed_dict=self.placeholders.get_train())
            except (tf.errors.OutOfRangeError, StopIteration):
                break

        print('Mean accuracy:\t\t{0:.2f}, mean loss:\t\t{1:.3f}'.format(100 * results[0], results[1]))

    @_check_session
    def evaluate(self):
        self._reset_test_pipeline()
        self._reset_metrics()

        results = None
        print('Evaluating...')
        while True:
            try:
                results = self._sess.run(self.fetches.test,
                                         feed_dict=self.placeholders.get_test())
            except (tf.errors.OutOfRangeError, StopIteration):
                break

        print('Mean accuracy:\t\t{0:.2f}, mean loss:\t\t{1:.3f}'.format(100*results[0], results[1]))

    @_check_session
    def _reset_train_pipeline(self):
        self._sess.run(self.db_reader.train)

    @_check_session
    def _reset_test_pipeline(self):
        self._sess.run(self.db_reader.test)

    @_check_session
    def _reset_metrics(self):
        self._sess.run(tf.local_variables_initializer())

    @_check_session
    def _initialize_variables(self):
        self._sess.run(tf.global_variables_initializer())

    _check_session = staticmethod(_check_session)
