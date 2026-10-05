"""Thin wrapper around tf.lite.Interpreter handling (de)quantization of I/O."""
import time
import warnings

import numpy as np
import tensorflow as tf

warnings.filterwarnings("ignore", message=".*tf.lite.Interpreter is deprecated.*")


class TFLiteModel:
    def __init__(self, path_or_bytes, num_threads=None):
        kw = dict(num_threads=num_threads)
        if isinstance(path_or_bytes, (bytes, bytearray)):
            self.it = tf.lite.Interpreter(model_content=bytes(path_or_bytes), **kw)
        else:
            self.it = tf.lite.Interpreter(model_path=str(path_or_bytes), **kw)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]
        self.in_dtype = self.inp["dtype"]
        self.in_scale, self.in_zp = self.inp["quantization"]
        self.out_scale, self.out_zp = self.out["quantization"]

    def quantize_input(self, img_u8):
        """img_u8: HxWx3 uint8 image (raw camera bytes)."""
        x = img_u8[None].astype(np.float32) / 255.0
        if self.in_dtype == np.float32:
            return x
        info = np.iinfo(self.in_dtype)
        q = np.round(x / self.in_scale) + self.in_zp
        return np.clip(q, info.min, info.max).astype(self.in_dtype)

    def raw(self, img_u8):
        self.it.set_tensor(self.inp["index"], self.quantize_input(img_u8))
        self.it.invoke()
        return self.it.get_tensor(self.out["index"])[0]

    def __call__(self, img_u8):
        """Dequantized output (logits for zta models, probabilities for the baseline)."""
        q = self.raw(img_u8)
        if self.out["dtype"] == np.float32:
            return q
        return (q.astype(np.float32) - self.out_zp) * self.out_scale

    def predict(self, imgs_u8):
        return np.stack([self(x) for x in imgs_u8])

    def latency_ms(self, img_u8, warmup=50, runs=500):
        x = self.quantize_input(img_u8)
        for _ in range(warmup):
            self.it.set_tensor(self.inp["index"], x)
            self.it.invoke()
        ts = []
        for _ in range(runs):
            t = time.perf_counter()
            self.it.set_tensor(self.inp["index"], x)
            self.it.invoke()
            ts.append((time.perf_counter() - t) * 1e3)
        return float(np.median(ts)), float(np.percentile(ts, 95))


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def ztachip_top5(raw_u8):
    """Python port of ztachip's NeuralNet::GetTop5 (SW/apps/nn/nn_util.cpp:347-387).

    Packs (value<<16)+index, so ties resolve to the higher index; the displayed
    'confidence' is (value*100)>>8 as in micropython/examples/image_classification.py.
    """
    packed = sorted(((int(v) << 16) + i for i, v in enumerate(raw_u8)), reverse=True)[:5]
    return [(p & 0xFFFF, p >> 16, (( p >> 16) * 100) >> 8) for p in packed]
