"""Static cost model: params, MACs, activation memory, and a theoretical ztachip bound.

The ztachip bound uses only figures published by the ztachip project and is a
*lower bound*, not a measurement:
  * 20 GOPS peak compute   (Documentation/Overview.md, SSD-MobileNet comparison)
  * 1.2 GB/s DDR bandwidth (Documentation/GettingStarted.md, LLM benchmark on Arty A7)
time >= max(2*MACs / 20e9, bytes_moved / 1.2e9), bytes_moved = weights + input + output.
Real on-board latency is measured in Phase II.
"""
import tensorflow as tf

ZTA_PEAK_OPS = 20e9
ZTA_DDR_BYTES_PER_S = 1.2e9


def layer_table(model):
    rows = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            rows += layer_table(layer)
            continue
        cls = type(layer).__name__
        try:
            out = layer.output.shape
            inp = layer.input.shape
        except (AttributeError, ValueError):
            continue
        macs = 0
        if cls == "Conv2D":
            k = layer.kernel_size
            macs = out[1] * out[2] * k[0] * k[1] * inp[-1] * out[-1]
        elif cls == "DepthwiseConv2D":
            k = layer.kernel_size
            macs = out[1] * out[2] * k[0] * k[1] * inp[-1]
        elif cls == "Dense":
            macs = inp[-1] * out[-1]
        if cls in ("InputLayer",):
            continue
        rows.append(dict(name=layer.name, type=cls, output=tuple(out[1:]),
                         params=int(layer.count_params()), macs=int(macs)))
    return rows


def summary(model, tflite_uint8_bytes=None):
    rows = layer_table(model)
    macs = sum(r["macs"] for r in rows)
    params = int(model.count_params())
    act = [int(tf.math.reduce_prod(r["output"])) for r in rows if r["output"]]
    weight_bytes = tflite_uint8_bytes if tflite_uint8_bytes else params      # 1 byte/weight
    in_bytes, out_bytes = 32 * 32 * 3, 10
    t_compute = 2 * macs / ZTA_PEAK_OPS
    t_mem = (weight_bytes + in_bytes + out_bytes) / ZTA_DDR_BYTES_PER_S
    return dict(
        params=params,
        macs=macs,
        peak_activation_bytes_uint8=max(act) if act else 0,
        weight_bytes_uint8=int(weight_bytes),
        zta_bound_ms=1e3 * max(t_compute, t_mem),
        zta_bound_compute_ms=1e3 * t_compute,
        zta_bound_memory_ms=1e3 * t_mem,
        layers=rows,
    )
