# ZTA-CNN

**A CIFAR-10 image classifier co-designed to run on [ztachip](https://github.com/ztachip/ztachip), an open-source RISC-V AI accelerator for FPGAs, plus the toolchain that proves a model will run on it before it reaches the board.**

Minor project (ECE), Phase I: software. Phase II, deployment on a ZedBoard, follows in December 2026.

**Live site:** https://ayushjo.github.io/zta-cnn/ (results, checker and a precomputed demo; the webcam demo runs locally)

---

## Why this project exists

ztachip runs TensorFlow Lite models on a low-cost FPGA, but when we read its TFLite importer (`SW/apps/nn/`) we found it supports only a narrow subset:

- **7 operators:** convolution, depthwise convolution, add, global average pool, concatenation, logistic, reshape. There is no MaxPool, no Dense and no Softmax.
- **Kernels:** 1×1 and 3×3 only, on square feature maps.
- **Number format:** only TF1-style **uint8 with one scale per tensor**. TensorFlow 2 can no longer produce this format.

An unsupported model does not fail with an error. An unknown layer reaches `assert(0)`, and on this bare-metal system `_exit()` is an infinite loop, so the board just freezes. Other unsupported features compute wrong numbers silently. The textbook CNN and TensorFlow's default int8 export both hit these traps.

So we designed the model around the hardware, and built tools to check the result.

## Results

CIFAR-10 test set, 10,000 images never used in training. UINT8 is the exact file that goes to ztachip.

| Model | Params | FP32 | **UINT8 (ztachip)** | Quantization cost | Size | Runs on ztachip |
|---|---|---|---|---|---|---|
| **ZTA-Plain** (deployed) | 290k | 88.38% | **88.30%** | 0.08 pp | 291 KiB | yes |
| ZTA-Mobile | 140k | 86.42% | 86.18% | 0.24 pp | 152 KiB | yes |
| ZTA-ResNet8 | 99k | 83.39% | 83.23% | 0.16 pp | 108 KiB | yes |
| Baseline CNN (MaxPool + Dense) | 269k | 76.57% | 76.61% | — | 268 KiB | **no** |

- **Accuracy:** the deployed model is above the MLPerf Tiny CIFAR-10 target of 85%, and agrees with FP32 on 98.2% of images.
- **Robustness:** under noise, blur, brightness, contrast and JPEG corruption, FP32 and UINT8 stay within 1.5 points of each other at every severity.
- **Grad-CAM:** the model attends to the object, not the background.

Full tables are in [`reports/results.md`](reports/results.md), and all figures are in [`reports/figures/`](reports/figures/).

## What we built

| Part | What it does |
|---|---|
| **Hardware-aware models** ([`src/zcnn/models.py`](src/zcnn/models.py)) | Built only from operators ztachip executes: 3×3 stride-2 convolutions instead of MaxPool, a 1×1 convolution on the pooled map instead of Dense, softmax on the host, and a fixed batch size so no SHAPE/PACK operators appear. |
| **int8 → uint8 rewriter** ([`src/zcnn/uint8_rewrite.py`](src/zcnn/uint8_rewrite.py)) | Converts a TF2 per-tensor int8 model into the uint8 format ztachip reads: q → q+128 and zp → zp+128, so every real value stays identical. |
| **Compatibility checker** ([`src/zcnn/ztachip_check.py`](src/zcnn/ztachip_check.py)) | 21 rules, each traced to the line of ztachip C++ that imposes it. It accepts ztachip's own MobileNet as the positive control, and rejects the textbook CNN and TF's default export. |
| **Evaluation** ([`src/zcnn/evaluate.py`](src/zcnn/evaluate.py) and friends) | Accuracy per quantization stage, confusion matrices, calibration, CPU latency, a cost model, Grad-CAM and corruption robustness. |
| **Phase II package** ([`src/zcnn/export_hw.py`](src/zcnn/export_hw.py)) | Model, labels, 10 golden test vectors in ztachip's own test format, and an on-board `test_cifar10()`. |
| **Showcase site** ([`server/`](server/), [`web/`](web/)) | Live demo (webcam, upload, test images) of FP32 next to the deployed UINT8 model, the checker with ztachip source views, interactive results, and a built-in presentation mode. |

## Quick start

Requirements: macOS or Linux, [uv](https://docs.astral.sh/uv/), Node 22+ with pnpm. Use Python 3.12; TensorFlow has no wheels for 3.14 yet.

```bash
git clone --recursive https://github.com/Ayushjo/zta-cnn.git
cd zta-cnn
make setup          # Python 3.12 + TensorFlow 2.21 via uv
make test           # toolchain and API tests
make web-install    # frontend dependencies
make web            # http://localhost:8411  (slides at /present)
```

The trained models and results are committed, so the site works straight away.

**Reproducing everything from scratch** takes about 3–4 hours on an Apple M4 CPU:

```bash
make train quantize eval explain robustness figures export
```

**Checking any TFLite model against ztachip:**

```bash
uv run python -m zcnn.ztachip_check path/to/model.tflite
```

## Repository layout

```
src/zcnn/          toolchain: data, models, training, quantization, rewriter, checker, evaluation
server/            FastAPI backend for the site + static export for GitHub Pages
web/               React + Vite + Tailwind frontend (Calm design system)
tests/             toolchain tests (no training needed)
artifacts/         trained models (.keras, .tflite) and the Phase II package in artifacts/hw/
reports/           metrics.json, robustness.json, results.md, figures/
docs/              architecture and Phase II notes
third_party/       ztachip, pinned as a git submodule (read-only reference)
```

## Honest limitations

- **No FPGA numbers yet.** ztachip's neural-network path runs neither on a PC nor in its RTL simulator, so latency here is measured on a laptop CPU. The "ztachip bound" is a theoretical lower bound from ztachip's published 20 GOPS and 1.2 GB/s, not a measurement.
- **The rewrite is exact in representation, but executed outputs can differ by 1–2 LSB.** TFLite's int8 and uint8 kernels round differently. All UINT8 accuracies are measured on the uint8 file itself.
- **Training length differs.** ZTA-Plain trained for 40 epochs and the other models for 25.
- **ZedBoard fit is unverified.** ztachip's reference board is the Arty A7-100T, so synthesis is the first Phase II task.

## Phase II (December)

1. Port ztachip to the ZedBoard: PS DDR through S_AXI_HP, 64-bit bus, re-pinned I/O.
2. Put the model into ztachip's firmware and run the golden-vector test on the board.
3. Measure on-board accuracy, latency, power and FPGA resources against the ZedBoard's ARM CPU.

Details are in [`docs/architecture.md`](docs/architecture.md).

## Team

- *Name 1*
- *Name 2*
- *Name 3*

Guide: *Name*

## Credits

- **ztachip** by Vuong Nguyen (Apache-2.0) is the accelerator: RTL, compiler and runtime. It is included as a read-only submodule; we did not write it.
- **CIFAR-10:** Krizhevsky, 2009.
- **MLPerf Tiny:** Banbury et al., 2021, the source of the ResNet-8 reference and the 85% target.
- **Grad-CAM:** Selvaraju et al., 2017.
