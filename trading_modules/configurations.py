import tensorflow as tf


def set_random_seed(seed:int=42):
    tf.keras.utils.set_random_seed(seed)