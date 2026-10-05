"""Model zoo.

Every `zta_*` model is restricted to what ztachip's TFLite importer can execute
(see ztachip_check.py for the rule list and source citations):
  * CONV_2D / DEPTHWISE_CONV_2D with 1x1 or 3x3 square kernels, dilation 1,
    stride only on 3x3 kernels, fused activation NONE/RELU/RELU6
  * ADD of two same-shape tensors
  * AVERAGE_POOL_2D only as a *global* pool (pool size == feature-map size)
  * RESHAPE only of a [1,1,1,C] tensor
  * no FULLY_CONNECTED, MAX_POOL_2D, SOFTMAX, MEAN, MUL, PAD ...
So: downsampling is a 3x3 stride-2 conv, the classifier is a 1x1 conv on the pooled
1x1 map, and softmax is applied on the host. BatchNorm is folded into the conv by
the TFLite converter.

`chatgpt_baseline` is the textbook CNN (MaxPool + Dense + Softmax) originally
suggested for the project; it is kept as an accuracy reference and as a negative
example for the compatibility checker.
"""
import tensorflow as tf
from tensorflow.keras import layers as L

from . import config as C


def _conv_bn(x, filters, kernel=3, stride=1, act="relu"):
    x = L.Conv2D(filters, kernel, strides=stride, padding="same", use_bias=False)(x)
    x = L.BatchNormalization()(x)
    if act:
        x = L.Activation(act)(x)
    return x


def _head(x, name):
    # Global average pool written as AveragePooling2D: GlobalAveragePooling2D lowers
    # to MEAN, which ztachip does not support.
    h = x.shape[1]
    x = L.AveragePooling2D(pool_size=h)(x)                 # -> [1,1,1,C]
    x = L.Conv2D(C.NUM_CLASSES, 1, use_bias=True, name="classifier")(x)  # "dense" as 1x1 conv
    x = L.Reshape((C.NUM_CLASSES,), name="logits")(x)      # logits; softmax on host
    return x


def zta_plain(width=1.0):
    """VGG-style plain CNN: 32-32-(s2)64-64-(s2)128-128."""
    w = lambda c: max(8, int(round(c * width / 8)) * 8)    # keep channels multiple of 8
    inp = L.Input((C.IMG_SIZE, C.IMG_SIZE, 3))
    x = _conv_bn(inp, w(32))
    x = _conv_bn(x, w(32))
    x = _conv_bn(x, w(64), stride=2)
    x = _conv_bn(x, w(64))
    x = _conv_bn(x, w(128), stride=2)
    x = _conv_bn(x, w(128))
    return tf.keras.Model(inp, _head(x, "zta_plain"), name="zta_plain")


def _res_block(x, filters, stride):
    y = _conv_bn(x, filters, stride=stride)
    y = _conv_bn(y, filters, act=None)
    if stride != 1 or x.shape[-1] != filters:
        # MLPerf Tiny uses a 1x1 stride-2 shortcut; ztachip ignores stride on 1x1
        # convs (conv.m convolution_1x1), so a 3x3 stride-2 conv is used instead.
        x = _conv_bn(x, filters, kernel=3, stride=stride, act=None)
    return L.Activation("relu")(L.Add()([x, y]))


def zta_resnet8():
    """MLPerf Tiny ResNet-8 (arXiv:2106.07597) adapted to ztachip's operator set."""
    inp = L.Input((C.IMG_SIZE, C.IMG_SIZE, 3))
    x = _conv_bn(inp, 16)
    x = _res_block(x, 16, 1)
    x = _res_block(x, 32, 2)
    x = _res_block(x, 64, 2)
    return tf.keras.Model(inp, _head(x, "zta_resnet8"), name="zta_resnet8")


def _dw_sep(x, filters, stride):
    x = L.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False)(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU(6.0)(x)
    x = L.Conv2D(filters, 1, padding="same", use_bias=False)(x)
    x = L.BatchNormalization()(x)
    return L.ReLU(6.0)(x)


def zta_mobile():
    """MobileNet-style depthwise-separable CNN."""
    inp = L.Input((C.IMG_SIZE, C.IMG_SIZE, 3))
    x = _conv_bn(inp, 32, act=None)
    x = L.ReLU(6.0)(x)
    for filters, stride in [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1)]:
        x = _dw_sep(x, filters, stride)
    return tf.keras.Model(inp, _head(x, "zta_mobile"), name="zta_mobile")


def chatgpt_baseline():
    """Conv-ReLU-MaxPool x2 -> Flatten -> Dense -> Softmax. NOT ztachip-compatible."""
    inp = L.Input((C.IMG_SIZE, C.IMG_SIZE, 3))
    x = L.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = L.MaxPooling2D()(x)
    x = L.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = L.MaxPooling2D()(x)
    x = L.Flatten()(x)
    x = L.Dense(128, activation="relu")(x)
    # Linear output for training (softmax is in the loss); a Softmax layer is added
    # at export so the TFLite graph matches the original design.
    x = L.Dense(C.NUM_CLASSES, name="logits")(x)
    return tf.keras.Model(inp, x, name="chatgpt_baseline")


BUILDERS = {
    "zta_plain": zta_plain,
    "zta_resnet8": zta_resnet8,
    "zta_mobile": zta_mobile,
    "chatgpt_baseline": chatgpt_baseline,
}


def build(name, **kw):
    return BUILDERS[name](**kw)


def export_model(model):
    """Model as it is exported to TFLite.

    The batch dimension is fixed to 1: with a dynamic batch, Keras' Reshape lowers to
    SHAPE + STRIDED_SLICE + PACK + RESHAPE, and those ops are unsupported on ztachip.
    The baseline gets its Softmax back so the graph matches the original design.
    """
    inp = L.Input(batch_shape=(1, C.IMG_SIZE, C.IMG_SIZE, 3))
    out = model(inp)
    if model.name == "chatgpt_baseline":
        out = L.Softmax()(out)
    return tf.keras.Model(inp, out, name=model.name)
