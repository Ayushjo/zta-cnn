"""Robustness of FP32 vs UINT8 models to camera-like corruptions.

Corruptions are generated locally in the spirit of CIFAR-10-C (Hendrycks & Dietterich,
2019), at 5 severities each, on a fixed 2,000-image test subset:
gaussian noise, gaussian blur, brightness, contrast, JPEG compression.

Usage:  python -m zcnn.robustness [model ...]
Writes reports/robustness.json and reports/figures/robustness.png
"""
import io
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter

from . import config as C
from .data import load_cifar10
from .quantize import tflite_path
from .tflite_runner import TFLiteModel

SEVERITIES = [1, 2, 3, 4, 5]


def corrupt(img, kind, s, rng):
    f = img.astype(np.float32)
    if kind == "gaussian_noise":
        f = f + rng.normal(0, [8, 16, 24, 32, 48][s - 1], f.shape)
    elif kind == "brightness":
        f = f + [20, 40, 60, 80, 100][s - 1]
    elif kind == "contrast":
        m = f.mean((0, 1), keepdims=True)
        f = (f - m) * [0.75, 0.6, 0.45, 0.3, 0.2][s - 1] + m
    elif kind == "gaussian_blur":
        return np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur([0.5, 0.75, 1.0, 1.25, 1.5][s - 1])))
    elif kind == "jpeg":
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "JPEG", quality=[60, 40, 25, 15, 8][s - 1])
        return np.asarray(Image.open(buf))
    return np.clip(f, 0, 255).astype(np.uint8)


KINDS = ["gaussian_noise", "gaussian_blur", "brightness", "contrast", "jpeg"]


def main(models=None):
    models = models or [C.DEPLOY_MODEL]
    _, _, (x, y) = load_cifar10()
    idx = np.random.default_rng(11).choice(len(x), 2000, replace=False)
    x, y = x[idx], y[idx]
    out = {}
    for name in models:
        keras = tf.keras.models.load_model(C.ARTIFACTS / f"{name}.keras")
        u8 = TFLiteModel(tflite_path(name, "uint8"))
        res = {"clean": {}}
        res["clean"]["fp32"] = float((keras.predict(x / 255.0, verbose=0).argmax(1) == y).mean())
        res["clean"]["uint8"] = float((u8.predict(x).argmax(1) == y).mean())
        for kind in KINDS:
            res[kind] = {"fp32": [], "uint8": []}
            for s in SEVERITIES:
                rng = np.random.default_rng(s)
                xc = np.stack([corrupt(im, kind, s, rng) for im in x])
                res[kind]["fp32"].append(float((keras.predict(xc / 255.0, verbose=0).argmax(1) == y).mean()))
                res[kind]["uint8"].append(float((u8.predict(xc).argmax(1) == y).mean()))
            print(f"[robustness] {name} {kind}: fp32 {res[kind]['fp32']} uint8 {res[kind]['uint8']}")
        out[name] = res
    (C.REPORTS / "robustness.json").write_text(json.dumps(out, indent=1))

    name = models[0]
    r = out[name]
    fig, axes = plt.subplots(1, len(KINDS), figsize=(16, 3.2), sharey=True)
    for ax, kind in zip(axes, KINDS):
        ax.plot([0] + SEVERITIES, [r["clean"]["fp32"]] + r[kind]["fp32"], "o-", label="FP32")
        ax.plot([0] + SEVERITIES, [r["clean"]["uint8"]] + r[kind]["uint8"], "s--", label="UINT8 (ztachip)")
        ax.set(title=kind.replace("_", " "), xlabel="severity", ylim=(0, 1))
    axes[0].set_ylabel("accuracy")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{name}: robustness to camera-like corruptions (2,000 test images)")
    fig.savefig(C.FIGURES / "robustness.png", dpi=130, bbox_inches="tight")

    ex = x[0]
    fig, axes = plt.subplots(len(KINDS), 6, figsize=(7, 6))
    for row, kind in zip(axes, KINDS):
        row[0].imshow(ex); row[0].set_ylabel(kind.replace("_", "\n"), fontsize=7)
        for s in SEVERITIES:
            row[s].imshow(corrupt(ex, kind, s, np.random.default_rng(s)))
        for a in row:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Corruption examples (severity 0-5)")
    fig.savefig(C.FIGURES / "corruption_examples.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
