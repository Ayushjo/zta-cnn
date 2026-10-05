"""Evaluate every model x variant on the CIFAR-10 test set.

Usage:  python -m zcnn.evaluate [model ...] [--latency-runs N]
Writes reports/metrics.json and artifacts/<model>_preds.npz.
"""
import argparse
import json
import platform
import subprocess
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from . import config as C
from .data import load_cifar10
from .profile import summary
from .quantize import VARIANTS, tflite_path
from .tflite_runner import TFLiteModel, softmax
from .ztachip_check import check


def ece(probs, labels, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e, rel = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc, c = (pred[m] == labels[m]).mean(), conf[m].mean()
            e += m.mean() * abs(acc - c)
            rel.append((float(c), float(acc), int(m.sum())))
    return float(e), rel


def to_probs(name, out):
    # zta models output logits (softmax runs on the host); the baseline outputs probabilities.
    return out if name == "chatgpt_baseline" else softmax(out)


def metrics(probs, y):
    pred = probs.argmax(1)
    top3 = np.argsort(-probs, 1)[:, :3]
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=range(C.NUM_CLASSES), zero_division=0)
    e, rel = ece(probs, y)
    return dict(top1=float((pred == y).mean()), top3=float((top3 == y[:, None]).any(1).mean()),
                precision=p.tolist(), recall=r.tolist(), f1=f.tolist(), ece=e, reliability=rel,
                confusion=confusion_matrix(y, pred, labels=range(C.NUM_CLASSES)).tolist())


def keras_latency(model, img, warmup=50, runs=500):
    fn = tf.function(lambda x: model(x, training=False))
    x = tf.constant(img[None].astype(np.float32) / 255.0)
    for _ in range(warmup):
        fn(x)
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        fn(x).numpy()
        ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts)), float(np.percentile(ts, 95))


def cpu_name():
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        return platform.processor()


def evaluate_model(name, x, y, runs):
    model = tf.keras.models.load_model(C.ARTIFACTS / f"{name}.keras")
    res = dict(variants={}, latency_ms={})
    keras_logits = model.predict(x.astype(np.float32) / 255.0, batch_size=500, verbose=0)
    probs = {"keras": softmax(keras_logits)}      # every trained Keras model outputs logits
    res["variants"]["keras"] = metrics(probs["keras"], y)
    res["latency_ms"]["keras_fp32"] = keras_latency(model, x[0], runs=runs // 2 or 1)

    raw_u8 = None
    for v in VARIANTS:
        path = tflite_path(name, v)
        m = TFLiteModel(path)
        out = m.predict(x)
        probs[v] = to_probs(name, out)
        res["variants"][v] = metrics(probs[v], y)
        res["variants"][v]["agreement_with_keras"] = float((probs[v].argmax(1) == probs["keras"].argmax(1)).mean())
        res["variants"][v]["size_kib"] = path.stat().st_size / 1024
        for th in (1, None):
            res["latency_ms"][f"tflite_{v}_{'1thread' if th else 'default'}"] = \
                TFLiteModel(path, num_threads=th).latency_ms(x[0], runs=runs)
        if v == "uint8":
            raw_u8 = np.stack([m.raw(img) for img in x])

    c = check(tflite_path(name, "uint8"))
    res["ztachip_compatible"] = c.passed
    res["ztachip_check"] = c.results()
    res["profile"] = summary(model, tflite_path(name, "uint8").stat().st_size)
    hist = C.ARTIFACTS / f"{name}_history.json"
    if hist.exists():
        h = json.loads(hist.read_text())
        res["train"] = dict(best_val_accuracy=h["best_val_accuracy"], epochs=len(h["accuracy"]),
                            sec_per_epoch=float(np.mean(h["epoch_seconds"])))
    np.savez_compressed(C.ARTIFACTS / f"{name}_preds.npz", y=y, raw_u8=raw_u8,
                        **{f"probs_{k}": v.astype(np.float32) for k, v in probs.items()})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=C.MODELS)
    ap.add_argument("--latency-runs", type=int, default=500)
    a = ap.parse_args()
    _, _, (x, y) = load_cifar10()
    out_path = C.REPORTS / "metrics.json"
    allm = json.loads(out_path.read_text()) if out_path.exists() else {}
    allm["_env"] = dict(cpu=cpu_name(), tensorflow=tf.__version__, python=platform.python_version(),
                        test_images=int(len(x)))
    for name in a.models:
        if not (C.ARTIFACTS / f"{name}.keras").exists():
            print(f"[eval] skip {name}: not trained")
            continue
        print(f"[eval] {name}")
        allm[name] = evaluate_model(name, x, y, a.latency_runs)
        v = allm[name]["variants"]
        print("   " + "  ".join(f"{k}={v[k]['top1']:.4f}" for k in v))
        out_path.write_text(json.dumps(allm, indent=1))


if __name__ == "__main__":
    main()
