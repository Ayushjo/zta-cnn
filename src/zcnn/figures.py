"""Generate every figure and the results table from reports/metrics.json.

Usage:  python -m zcnn.figures
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config as C
from .data import load_cifar10

LABEL = {"keras": "Keras FP32", "fp32": "TFLite FP32", "int8_pc": "INT8 per-channel (TF default)",
         "int8_pt": "INT8 per-tensor", "uint8": "UINT8 (ztachip)"}
NICE = {"zta_plain": "ZTA-Plain", "zta_resnet8": "ZTA-ResNet8", "zta_mobile": "ZTA-Mobile",
        "chatgpt_baseline": "Baseline CNN (MaxPool+Dense)"}
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})


def _save(fig, name):
    fig.savefig(C.FIGURES / f"{name}.png")
    plt.close(fig)


def training_curves(models):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for m in models:
        p = C.ARTIFACTS / f"{m}_history.json"
        if not p.exists():
            continue
        h = json.loads(p.read_text())
        ep = np.arange(1, len(h["accuracy"]) + 1)
        l, = axes[0].plot(ep, h["val_accuracy"], label=f"{NICE[m]} val")
        axes[0].plot(ep, h["accuracy"], "--", color=l.get_color(), alpha=.6, label=f"{NICE[m]} train")
        axes[1].plot(ep, h["val_loss"], color=l.get_color(), label=f"{NICE[m]} val")
        axes[1].plot(ep, h["loss"], "--", color=l.get_color(), alpha=.6)
    axes[0].set(title="Accuracy (train dashed, train set augmented)", xlabel="epoch", ylabel="accuracy")
    axes[1].set(title="Loss (label-smoothed CE)", xlabel="epoch", ylabel="loss")
    axes[0].legend(fontsize=7)
    _save(fig, "training_curves")


def confusion(name, M, variant):
    cm = np.array(M[name]["variants"][variant]["confusion"])
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10), C.CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(10), C.CLASS_NAMES)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > .5 else "black")
    ax.set(xlabel="predicted", ylabel="true",
           title=f"{NICE[name]} - {LABEL[variant]}  (acc {M[name]['variants'][variant]['top1']:.2%})")
    fig.colorbar(im, ax=ax, fraction=.046)
    _save(fig, f"confusion_{name}_{variant}")


def per_class(name, M):
    v = M[name]["variants"]
    x = np.arange(10)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(x - .2, v["keras"]["f1"], .4, label="FP32")
    ax.bar(x + .2, v["uint8"]["f1"], .4, label="UINT8 (ztachip)")
    ax.set_xticks(x, C.CLASS_NAMES)
    ax.set(ylabel="F1 score", ylim=(0, 1), title=f"{NICE[name]}: per-class F1, FP32 vs UINT8")
    ax.legend()
    _save(fig, f"per_class_{name}")


def quant_comparison(models, M):
    vs = ["keras", "int8_pc", "int8_pt", "uint8"]
    fig, ax = plt.subplots(figsize=(10, 4))
    w = .8 / len(vs)
    x = np.arange(len(models))
    for k, v in enumerate(vs):
        acc = [M[m]["variants"][v]["top1"] * 100 for m in models]
        bars = ax.bar(x + (k - 1.5) * w, acc, w, label=LABEL[v])
        ax.bar_label(bars, fmt="%.1f", fontsize=6, padding=1)
    ax.set_xticks(x, [NICE[m] + ("" if M[m]["ztachip_compatible"] else "\n(not ztachip-compatible)") for m in models])
    ax.set(ylabel="test accuracy (%)", ylim=(0, 100), title="Accuracy after quantization (CIFAR-10 test, 10,000 images)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    _save(fig, "quantization_accuracy")


def pareto(models, M):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for m in models:
        p = M[m]["profile"]
        acc = M[m]["variants"]["uint8"]["top1"] * 100
        ax.scatter(p["macs"] / 1e6, acc, s=p["params"] / 1500, alpha=.7,
                   marker="o" if M[m]["ztachip_compatible"] else "x")
        ax.annotate(f"{NICE[m]}\n{p['params']/1e3:.0f}k params", (p["macs"] / 1e6, acc),
                    fontsize=7, xytext=(6, -4), textcoords="offset points")
    ax.set(xlabel="MACs per image (millions)", ylabel="UINT8 test accuracy (%)",
           title="Accuracy vs compute (marker size = params)")
    _save(fig, "accuracy_vs_macs")


def reliability(name, M):
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    for v in ("keras", "uint8"):
        rel = M[name]["variants"][v]["reliability"]
        c, a, _ = zip(*rel)
        ax.plot(c, a, "o-", label=f"{LABEL[v]} (ECE {M[name]['variants'][v]['ece']:.3f})")
    ax.set(xlabel="confidence", ylabel="accuracy", title=f"{NICE[name]}: calibration")
    ax.legend(fontsize=8)
    _save(fig, f"reliability_{name}")


def latency(models, M):
    keys = [("keras_fp32", "Keras FP32"), ("tflite_fp32_1thread", "TFLite FP32"),
            ("tflite_int8_pt_1thread", "TFLite INT8"), ("tflite_uint8_1thread", "TFLite UINT8")]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    w = .8 / len(keys)
    x = np.arange(len(models))
    for k, (key, lab) in enumerate(keys):
        vals = [M[m]["latency_ms"][key][0] for m in models]
        bars = ax.bar(x + (k - 1.5) * w, vals, w, label=lab)
        ax.bar_label(bars, fmt="%.2f", fontsize=6, padding=1)
    ax.set_xticks(x, [NICE[m] for m in models])
    ax.set(ylabel="median latency (ms, batch 1)", title=f"CPU inference latency - {M['_env']['cpu']} (TFLite 1 thread)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.text(0.01, -0.06, "TFLite's XNNPACK backend accelerates FP32/INT8 but not legacy UINT8 kernels, so CPU UINT8 "
             "latency says nothing about ztachip; it is the format ztachip executes in hardware.", fontsize=7)
    _save(fig, "cpu_latency")


def sample_predictions(name):
    p = np.load(C.ARTIFACTS / f"{name}_preds.npz")
    _, _, (x, y) = load_cifar10()
    pu = p["probs_uint8"]
    rng = np.random.default_rng(3)
    idx = rng.choice(len(x), 24, replace=False)
    fig, axes = plt.subplots(3, 8, figsize=(12, 5.2))
    for ax, i in zip(axes.flat, idx):
        ax.imshow(x[i])
        k = pu[i].argmax()
        ax.set_title(f"{C.CLASS_NAMES[k]} {pu[i, k]:.0%}", fontsize=8, color="green" if k == y[i] else "red")
        ax.set_xlabel(f"true: {C.CLASS_NAMES[y[i]]}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{NICE[name]} UINT8 predictions on random test images (green = correct)")
    _save(fig, f"samples_{name}")

    wrong = np.where(pu.argmax(1) != y)[0]
    conf = pu[wrong].max(1)
    worst = wrong[np.argsort(-conf)[:16]]
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.6))
    for ax, i in zip(axes.flat, worst):
        ax.imshow(x[i])
        k = pu[i].argmax()
        ax.set_title(f"pred {C.CLASS_NAMES[k]} {pu[i, k]:.0%}", fontsize=7, color="red")
        ax.set_xlabel(f"true {C.CLASS_NAMES[y[i]]}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{NICE[name]}: most confident mistakes")
    _save(fig, f"mistakes_{name}")


def results_markdown(models, M):
    env = M["_env"]
    L = ["# Results", "",
         f"Auto-generated by `python -m zcnn.figures`. CPU: {env['cpu']}, TensorFlow {env['tensorflow']}, "
         f"Python {env['python']}, test set: {env['test_images']} CIFAR-10 images.", "",
         "## Accuracy by quantization variant (top-1, %)", "",
         "| Model | Params | MACs | Keras FP32 | TFLite FP32 | INT8 per-channel | INT8 per-tensor | **UINT8 (ztachip)** | UINT8 agrees w/ FP32 | ztachip-compatible |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        v, p = M[m]["variants"], M[m]["profile"]
        L.append(f"| {NICE[m]} | {p['params']:,} | {p['macs']/1e6:.1f} M | {v['keras']['top1']*100:.2f} | "
                 f"{v['fp32']['top1']*100:.2f} | {v['int8_pc']['top1']*100:.2f} | {v['int8_pt']['top1']*100:.2f} | "
                 f"**{v['uint8']['top1']*100:.2f}** | {v['uint8']['agreement_with_keras']*100:.1f} % | "
                 f"{'yes' if M[m]['ztachip_compatible'] else 'NO'} |")
    L += ["", "## Model size and latency", "",
          "| Model | FP32 .tflite | UINT8 .tflite | Keras FP32 ms | TFLite FP32 ms (1T) | TFLite UINT8 ms (1T) | TFLite UINT8 ms (all threads) | ztachip theoretical lower bound ms |",
          "|---|---|---|---|---|---|---|---|"]
    for m in models:
        v, lt, p = M[m]["variants"], M[m]["latency_ms"], M[m]["profile"]
        L.append(f"| {NICE[m]} | {v['fp32']['size_kib']:.0f} KiB | {v['uint8']['size_kib']:.0f} KiB | "
                 f"{lt['keras_fp32'][0]:.3f} | {lt['tflite_fp32_1thread'][0]:.3f} | {lt['tflite_uint8_1thread'][0]:.3f} | "
                 f"{lt['tflite_uint8_default'][0]:.3f} | {p['zta_bound_ms']:.3f} |")
    L += ["", "Latency = median of 500 batch-1 runs after 50 warm-up runs. The ztachip bound is "
          "max(2*MACs/20 GOPS, bytes/1.2 GB/s) from figures published by the ztachip project; it is "
          "**not a measurement**. On-board latency is measured in Phase II. TFLite UINT8 is slow on the CPU "
          "because XNNPACK accelerates only FP32/INT8; legacy UINT8 kernels run the reference path. That is "
          "a CPU-library effect, not a property of the model.", "",
          "## Calibration and top-3", "", "| Model | Top-3 FP32 | Top-3 UINT8 | ECE FP32 | ECE UINT8 |", "|---|---|---|---|---|"]
    for m in models:
        v = M[m]["variants"]
        L.append(f"| {NICE[m]} | {v['keras']['top3']*100:.2f} | {v['uint8']['top3']*100:.2f} | "
                 f"{v['keras']['ece']:.4f} | {v['uint8']['ece']:.4f} |")
    L += ["", "## ztachip compatibility check (UINT8 models)", ""]
    for m in models:
        fails = [r for r in M[m]["ztachip_check"] if r["status"] != "PASS"]
        L.append(f"- **{NICE[m]}**: " + ("all rules pass" if not fails else
                 "; ".join(f"{r['status']} {r['rule']}: {', '.join(r['issues'][:3])}" for r in fails)))
    if "train" in M[models[0]]:
        L += ["", "## Training", "", "| Model | Epochs | Best val acc | s/epoch |", "|---|---|---|---|"]
        for m in models:
            t = M[m].get("train")
            if t:
                L.append(f"| {NICE[m]} | {t['epochs']} | {t['best_val_accuracy']*100:.2f} | {t['sec_per_epoch']:.1f} |")
    (C.REPORTS / "results.md").write_text("\n".join(L) + "\n")


def main():
    M = json.loads((C.REPORTS / "metrics.json").read_text())
    models = [m for m in C.MODELS if m in M]
    training_curves(models)
    quant_comparison(models, M)
    pareto(models, M)
    latency(models, M)
    for m in models:
        confusion(m, M, "keras")
        confusion(m, M, "uint8")
        per_class(m, M)
        reliability(m, M)
    sample_predictions(C.DEPLOY_MODEL)
    results_markdown(models, M)
    print(f"[figures] wrote {len(list(C.FIGURES.glob('*.png')))} figures and reports/results.md")


if __name__ == "__main__":
    main()
