"""Train a model from the zoo on CIFAR-10.

Usage:  python -m zcnn.train zta_plain [--epochs N]
Writes artifacts/<name>.keras (best val accuracy) and artifacts/<name>_history.json.
"""
import argparse
import json
import time

import numpy as np
import tensorflow as tf

from . import config as C
from .data import make_datasets
from .models import build


def train(name, epochs=None):
    hp = dict(C.TRAIN)
    if epochs:
        hp["epochs"] = epochs
    tf.keras.utils.set_random_seed(C.SEED)
    train_ds, val_ds, _ = make_datasets(hp["batch_size"])
    steps = int(train_ds.cardinality()) * hp["epochs"]

    model = build(name)
    sched = tf.keras.optimizers.schedules.CosineDecay(hp["lr"], decay_steps=steps, alpha=0.01)
    opt = tf.keras.optimizers.SGD(sched, momentum=hp["momentum"], nesterov=True,
                                  weight_decay=hp["weight_decay"])
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True,
                                                               label_smoothing=hp["label_smoothing"]),
                  metrics=["accuracy"])

    ckpt = C.ARTIFACTS / f"{name}.keras"
    epoch_times = []

    class Timer(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.t = time.time()

        def on_epoch_end(self, epoch, logs=None):
            epoch_times.append(time.time() - self.t)

    cbs = [tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_accuracy", save_best_only=True),
           tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=hp["early_stop_patience"]),
           Timer()]
    print(f"[train] {name}: {model.count_params():,} params, {hp['epochs']} epochs")
    hist = model.fit(train_ds, validation_data=val_ds, epochs=hp["epochs"], callbacks=cbs, verbose=2)

    h = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    h.update(params=int(model.count_params()), hyperparams=hp,
             epoch_seconds=epoch_times, best_val_accuracy=float(np.max(h["val_accuracy"])))
    (C.ARTIFACTS / f"{name}_history.json").write_text(json.dumps(h, indent=1))
    print(f"[train] {name}: best val acc {h['best_val_accuracy']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=C.MODELS)
    ap.add_argument("--epochs", type=int)
    a = ap.parse_args()
    for m in a.models:
        train(m, a.epochs)


if __name__ == "__main__":
    main()
