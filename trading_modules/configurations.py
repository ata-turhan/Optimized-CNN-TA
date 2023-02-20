import numpy as np
import tensorflow as tf


def set_random_seed(seed: int = 42):
    tf.keras.utils.set_random_seed(seed)


def shift_predictions(predictions:np.array) -> np.array:
    predictions = np.roll(predictions, 1)
    predictions[0] = 0
    return predictions
