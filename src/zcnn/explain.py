"""Grad-CAM (Selvaraju et al. 2017; keras.io/examples/vision/grad_cam) for the FP32 model.

Usage:  python -m zcnn.explain [model]
Writes reports/figures/gradcam_<model>.png
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from . import config as C
from .data import load_cifar10


def last_feature_layer(model):
    """Output of the layer feeding the global average pool."""
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.AveragePooling2D):
            return model.layers[i - 1]
    raise ValueError("no AveragePooling2D head found")


def gradcam(model, imgs_u8):
    feat = last_feature_layer(model)
    grad_model = tf.keras.Model(model.inputs, [feat.output, model.output])
    x = tf.constant(imgs_u8.astype(np.float32) / 255.0)
    with tf.GradientTape() as tape:
        fmap, logits = grad_model(x, training=False)
        pred = tf.argmax(logits, 1)
        score = tf.gather(logits, pred, batch_dims=1)
    grads = tape.gradient(score, fmap)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * fmap, -1))
    cam = cam / (tf.reduce_max(cam, axis=(1, 2), keepdims=True) + 1e-8)
    cam = tf.image.resize(cam[..., None], imgs_u8.shape[1:3], "bilinear")[..., 0]
    return cam.numpy(), pred.numpy()


def main(name=C.DEPLOY_MODEL):
    model = tf.keras.models.load_model(C.ARTIFACTS / f"{name}.keras")
    _, _, (x, y) = load_cifar10()
    rng = np.random.default_rng(7)
    idx = rng.choice(len(x), 12, replace=False)
    cam, pred = gradcam(model, x[idx])
    fig, axes = plt.subplots(2, 12, figsize=(16, 3.4))
    for k, i in enumerate(idx):
        axes[0, k].imshow(x[i])
        axes[0, k].set_title(f"true {C.CLASS_NAMES[y[i]]}", fontsize=7)
        axes[1, k].imshow(x[i])
        axes[1, k].imshow(cam[k], cmap="jet", alpha=.45)
        axes[1, k].set_title(f"pred {C.CLASS_NAMES[pred[k]]}", fontsize=7,
                             color="green" if pred[k] == y[i] else "red")
        for a in axes[:, k]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Grad-CAM: where the {name} model looks (last 8x8 feature map, upsampled)")
    fig.savefig(C.FIGURES / f"gradcam_{name}.png", dpi=130, bbox_inches="tight")
    print(f"[explain] wrote gradcam_{name}.png")


if __name__ == "__main__":
    main(*sys.argv[1:])
