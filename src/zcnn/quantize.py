"""Keras -> TFLite conversion in four variants.

  fp32      float reference
  int8_pc   TF2 default full-integer PTQ (int8, per-channel weights) - NOT ztachip-compatible
  int8_pt   int8 with per-tensor weights and int8 I/O (no QUANTIZE ops)
  uint8     int8_pt rewritten to TF1-style uint8 -> the model deployed on ztachip

Usage:  python -m zcnn.quantize [model_name ...]
"""
import contextlib
import io
import sys

import tensorflow as tf

from . import config as C
from .data import representative_dataset
from .models import export_model
from .uint8_rewrite import rewrite_int8_to_uint8

VARIANTS = ["fp32", "int8_pc", "int8_pt", "uint8"]


def convert(model, variant, rep_gen=None):
    model = export_model(model)
    if variant == "uint8":
        return rewrite_int8_to_uint8(convert(model, "int8_pt", rep_gen))
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if variant == "fp32":
        return _quiet(conv.convert)
    rep_gen = rep_gen or representative_dataset()
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if variant == "int8_pc":
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    elif variant == "int8_pt":
        # Per-tensor weights (ztachip reads only scale[0], tf_util.cpp:196-202).
        conv._experimental_disable_per_channel = True
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    else:
        raise ValueError(variant)
    return _quiet(conv.convert)


def _quiet(fn):
    """Keras 3 prints the whole saved-model signature during conversion."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


def tflite_path(name, variant):
    return C.ARTIFACTS / f"{name}_{variant}.tflite"


def quantize_model(name):
    model = tf.keras.models.load_model(C.ARTIFACTS / f"{name}.keras")
    rep = representative_dataset()
    for v in VARIANTS:
        data = convert(model, v, rep)
        tflite_path(name, v).write_bytes(data)
        print(f"  {name}_{v}.tflite  {len(data)/1024:.1f} KiB")


def main(argv):
    for name in (argv or C.MODELS):
        print(f"[quantize] {name}")
        quantize_model(name)


if __name__ == "__main__":
    main(sys.argv[1:])
