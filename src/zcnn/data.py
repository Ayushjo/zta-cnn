"""CIFAR-10 loading, splitting and augmentation.

The model consumes x/255 in [0, 1]. Normalisation is done here and NOT as a Keras
layer: a Rescaling layer would become a TFLite MUL op, which ztachip does not support
(SW/apps/nn/tf_util.cpp ParseOpcode). With inputs in [0, 1] the converter picks an
input scale of ~1/255, so after the uint8 rewrite the zero point is 0 and raw camera
bytes can be fed to the accelerator unchanged.
"""
import numpy as np
import tensorflow as tf

from . import config as C


def load_cifar10():
    """Returns uint8 arrays: (x_train, y_train), (x_val, y_val), (x_test, y_test)."""
    (x, y), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y, y_test = y.reshape(-1), y_test.reshape(-1)
    idx = np.random.default_rng(C.SEED).permutation(len(x))
    val, tr = idx[:C.VAL_SIZE], idx[C.VAL_SIZE:]
    return (x[tr], y[tr]), (x[val], y[val]), (x_test, y_test)


def to_float(x):
    return x.astype(np.float32) / 255.0


def _augment(img, label):
    img = tf.pad(img, [[4, 4], [4, 4], [0, 0]], mode="REFLECT")
    img = tf.image.random_crop(img, [C.IMG_SIZE, C.IMG_SIZE, 3])
    img = tf.image.random_flip_left_right(img)
    return img, label


def make_datasets(batch_size):
    (xtr, ytr), (xval, yval), (xte, yte) = load_cifar10()
    ytr_oh = tf.one_hot(ytr, C.NUM_CLASSES)
    train = (tf.data.Dataset.from_tensor_slices((to_float(xtr), ytr_oh))
             .shuffle(len(xtr), seed=C.SEED)
             .map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
             .batch(batch_size).prefetch(tf.data.AUTOTUNE))
    val = (tf.data.Dataset.from_tensor_slices((to_float(xval), tf.one_hot(yval, C.NUM_CLASSES)))
           .batch(512).prefetch(tf.data.AUTOTUNE))
    return train, val, (xte, yte)


def representative_dataset(n=C.REP_SAMPLES):
    """Generator for TFLite post-training quantization calibration."""
    (xtr, _), _, _ = load_cifar10()
    idx = np.random.default_rng(C.SEED + 1).choice(len(xtr), n, replace=False)
    xs = to_float(xtr[idx])
    # Make sure the calibrated input range is exactly [0, 1] -> scale 1/255, zp 0 (uint8).
    xs[0, 0, 0, :] = 0.0
    xs[0, 0, 1, :] = 1.0

    def gen():
        for i in range(n):
            yield [xs[i:i + 1]]
    return gen
