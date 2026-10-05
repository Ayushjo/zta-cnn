"""Rewrite a TF2 int8 per-tensor TFLite model into a TF1-style uint8 model.

Why: ztachip only executes asymmetric uint8, per-tensor quantized models
(SW/apps/nn/tf_util.cpp:220-231 computes conv parameters only for UINT8 inputs;
kernels hard-code UINT8 in conv.m / fcn.m). TensorFlow 2 can no longer emit such
models: it produces int8 internally, and `inference_input_type=tf.uint8` only wraps
the int8 graph in QUANTIZE ops, which ztachip does not support.

The rewrite is exact in representation. For an int8 tensor with scale s and zero
point z, real = s * (q - z). Mapping q' = q + 128 and z' = z + 128 gives
real = s * (q' - z'), the same value, now in the uint8 range [0, 255]. Scales are
unchanged, so every requantisation multiplier (s_in * s_w / s_out) is unchanged, and
int32 biases (scale s_in * s_w, zero point 0) need no change either.

Executed outputs are not always bit-identical: TFLite's int8 kernels and its legacy
uint8 kernels round the requantisation step differently, so outputs can differ by
1-2 LSB (measured: max 2 LSB, top-1 agreement 99.7-100 % on random inputs). Accuracy
of the deployed model is therefore always measured on the uint8 model itself.
"""
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema
from tensorflow.lite.tools import flatbuffer_utils as fu

INT8 = schema.TensorType.INT8
UINT8 = schema.TensorType.UINT8
_OPNAMES = {v: k for k, v in schema.BuiltinOperator.__dict__.items() if not k.startswith("_")}


def op_name(model, op):
    oc = model.operatorCodes[op.opcodeIndex]
    return _OPNAMES[max(oc.builtinCode, oc.deprecatedBuiltinCode)]


def rewrite_int8_to_uint8(tflite_bytes: bytes) -> bytes:
    model = fu.convert_bytearray_to_object(bytearray(tflite_bytes))
    if len(model.subgraphs) != 1:
        raise ValueError("expected a single subgraph")
    sg = model.subgraphs[0]

    for op in sg.operators:
        name = op_name(model, op)
        if name in ("QUANTIZE", "DEQUANTIZE"):
            raise ValueError(f"model contains {name}; convert with int8 input/output types")

    done_buffers = set()
    for t in sg.tensors:
        if t.type != INT8:
            continue
        q = t.quantization
        if q is None or q.scale is None or len(q.scale) != 1:
            raise ValueError(f"tensor {t.name!r} is not per-tensor quantized")
        t.type = UINT8
        q.zeroPoint = (np.asarray(q.zeroPoint, dtype=np.int64) + 128).astype(np.int64)
        buf = model.buffers[t.buffer]
        if buf.data is not None and len(buf.data) and t.buffer not in done_buffers:
            raw = np.asarray(buf.data, dtype=np.uint8)
            buf.data = (raw.view(np.int8).astype(np.int16) + 128).astype(np.uint8)
            done_buffers.add(t.buffer)

    return bytes(fu.convert_object_to_bytearray(model))
