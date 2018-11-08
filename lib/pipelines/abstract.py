import tensorflow as tf


class AbstractPipeline(object):
    def __init__(self, dataset):
        self._iterator = tf.data.Iterator.from_structure(
            output_types=dataset.output_types,
            output_shapes=dataset.output_shapes
        )

    def get_tensors(self):
        return self._iterator.get_next()

    def bind_dataset(self, dataset):
        # returns iterator init op that needs to be run when switching between datasets
        return self._iterator.make_initializer(dataset)
