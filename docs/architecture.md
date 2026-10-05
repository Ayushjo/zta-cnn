## System overview

```
 PHASE I  (done - software, laptop)                                PHASE II  (December - hardware)
 ─────────────────────────────────────────────────────────         ──────────────────────────────────────

 CIFAR-10 ──► Hardware-aware CNN ──► PTQ int8 per-tensor ──► int8→uint8 ──► ztachip ──► cifar10_zta.tflite ──► ztachip on FPGA
 (50k/10k)    (only ztachip ops)     (TF 2.21 converter)     rewriter       checker     + labels + golden     (ZedBoard / Arty A7)
                    │                                            (ours)      (ours)       vectors (ours)              │
                    ▼                                                                                                 ▼
          accuracy, confusion, calibration, Grad-CAM, robustness, CPU latency            on-board accuracy, latency, power,
                                                                                         LUT/FF/BRAM/DSP vs CPU baseline
```

## Why the model looks the way it does

ztachip's TFLite engine (`SW/apps/nn/`) only executes a subset of TFLite. We read its
source and turned every limitation into a design rule (and a checker rule):

| ztachip limitation (source)                                   | Design decision                                   |
|---------------------------------------------------------------|---------------------------------------------------|
| No MAX_POOL_2D (`tf_util.cpp:74-100`)                         | Downsample with 3×3 stride-2 convolutions          |
| No FULLY_CONNECTED                                            | Classifier = 1×1 conv on the pooled 1×1 map        |
| No SOFTMAX                                                    | Softmax on the host; ztachip ranks raw uint8 logits|
| No MEAN (Keras GlobalAveragePooling2D)                        | `AveragePooling2D(pool_size=8)`                    |
| No MUL (Keras Rescaling)                                      | Input scale 1/255 baked into quantization          |
| AVERAGE_POOL must be global (`fcn.m:111-152`)                 | Exactly one pool, at the end                       |
| Kernels 1×1 / 3×3 only; 1×1 ignores stride (`conv.m:738-745`) | 3×3 convs; ResNet shortcut is 3×3 stride 2         |
| uint8 per-tensor only (`tf_util.cpp:196-231`)                 | Per-tensor PTQ + our int8→uint8 rewriter           |
| Dynamic-batch Reshape → SHAPE/STRIDED_SLICE/PACK ops          | Export with fixed batch = 1                        |

## Model zoo

| Model        | Structure                                                              | ztachip ops used                       |
|--------------|------------------------------------------------------------------------|----------------------------------------|
| ZTA-Plain    | 6× [3×3 conv-BN-ReLU] 32-32-64↓-64-128↓-128 → AvgPool(8) → 1×1 conv(10) | CONV_2D, AVERAGE_POOL_2D, RESHAPE       |
| ZTA-ResNet8  | MLPerf-Tiny ResNet-8 adapted (16/32/64, residual ADD)                   | + ADD                                  |
| ZTA-Mobile   | MobileNet-style depthwise-separable blocks                              | + DEPTHWISE_CONV_2D                    |
| Baseline CNN | Conv-MaxPool ×2 → Dense → Softmax (the textbook design)                 | MAX_POOL_2D, FULLY_CONNECTED, SOFTMAX ✗ |

## Proposed Phase II hardware (ZedBoard, Zynq-7020)

```
 ┌──────────────────────────── ZedBoard (XC7Z020) ─────────────────────────────┐
 │  PS (ARM Cortex-A9)                     PL (FPGA fabric)                    │
 │  ┌──────────────┐   S_AXI_HP (64-bit)   ┌───────────────────────────────┐   │
 │  │ DDR3 512 MB  │◄─────────────────────►│ ztachip SoC                    │   │
 │  │ controller   │                       │  VexRiscv RV32IM (control)     │   │
 │  └──────────────┘                       │  tensor engine: 4 pcores ×     │   │
 │  FSBL / ps7_init configures DDR         │  16 threads, VECTOR_WIDTH 8    │   │
 │                                         │  runs cifar10_zta.tflite        │   │
 │                                         └──────┬─────────────┬──────────┘   │
 │                                          camera (Pmod)   VGA (Pmod) / UART  │
 └─────────────────────────────────────────────────────────────────────────────┘
```

Porting steps (from ztachip's porting guide and our reading of `HW/`; ztachip paths below are inside `third_party/ztachip/`):
1. Replace the MIG DDR controller in `HW/examples/GHRD/main.v` with the PS7 S_AXI_HP port.
2. Set `exmem_data_width_c = 64` in `HW/src/config.vhd` (HP ports are 64-bit; reference design uses 128).
3. Drop the Arty-specific Ethernet IP, re-pin the XDC for ZedBoard Pmods, clock from FCLK/MMCM.
4. Rebase `SW/linker.ld` RAM to the PS-DDR address window; keep `NUM_PCORE=4`.
5. Week 1: synthesis to check fit (Z-7020 has ~84 % of the A7-100T's LUTs; ztachip's utilisation is unpublished).
   If an Arty A7-100T is available, use ztachip's reference flow directly.

## What is ours vs. ztachip

| ztachip (Vuong Nguyen, Apache-2.0)            | This project                                                         |
|-----------------------------------------------|----------------------------------------------------------------------|
| RTL accelerator, RISC-V SoC, compiler, runtime| Hardware-aware CNN design for ztachip's operator set                  |
| TFLite execution engine                       | Training, evaluation, calibration, robustness, explainability        |
| MobileNet / SSD / LLM demos                   | int8→uint8 TFLite rewriter (TF2 can no longer emit uint8 models)      |
|                                               | ztachip compatibility checker (21 rules traced to ztachip source)    |
|                                               | Phase II package: model, labels, golden vectors, on-board test        |
|                                               | ZedBoard port (Phase II)                                              |

## Timeline

| Phase | When | Deliverable |
|---|---|---|
| I   | Oct 4-6   | Software toolchain, trained + quantized models, compatibility proof, CPU baseline, demo |
| II  | Oct-Nov   | QAT (recover per-tensor accuracy), width sweep, ZedBoard synthesis + port |
| III | Nov       | Firmware integration (model in `SW/fs`, `test_cifar10`), camera pipeline (640×480→32×32 resize chain) |
| IV  | Dec       | On-board golden test, accuracy/latency/power/resources vs CPU, final report |
