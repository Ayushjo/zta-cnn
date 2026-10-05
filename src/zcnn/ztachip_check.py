"""Static compatibility checker: will this .tflite run correctly on ztachip?

ztachip's TFLite importer does not report unsupported models: unknown ops hit
`assert(0)` and the bare-metal `_exit` spins forever, and several unsupported
features (int8, per-channel scales, 5x5 kernels, dilation, ...) are silently
computed wrong. Each rule below encodes one constraint found by reading the
ztachip sources; `ref` points at the code that imposes it (paths relative to the
ztachip repo root).

Usage:  python -m zcnn.ztachip_check model.tflite [--labels labels.txt]
        python -m zcnn.ztachip_check ztachip-mobilenet     # positive control
"""
import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tensorflow.lite.python import schema_py_generated as schema
from tensorflow.lite.tools import flatbuffer_utils as fu

from .uint8_rewrite import op_name

TT = schema.TensorType
_TTNAMES = {v: k for k, v in TT.__dict__.items() if not k.startswith("_")}
AF = schema.ActivationFunctionType
_AFNAMES = {v: k for k, v in AF.__dict__.items() if not k.startswith("_")}
PAD = schema.Padding

SUPPORTED_OPS = {"ADD", "AVERAGE_POOL_2D", "CONV_2D", "DEPTHWISE_CONV_2D",
                 "CONCATENATION", "LOGISTIC", "RESHAPE"}
SUPPORTED_CUSTOM = {"TFLite_Detection_PostProcess"}
NN = "SW/apps/nn/"


@dataclass
class Rule:
    id: str
    title: str
    ref: str
    severity: str = "error"          # error -> FAIL, warning -> WARN
    issues: list = field(default_factory=list)

    @property
    def status(self):
        if not self.issues:
            return "PASS"
        return "FAIL" if self.severity == "error" else "WARN"


RULES = [
    ("schema", "TFLite schema version 3", NN + "tf.cpp:95-97"),
    ("subgraph", "Exactly one subgraph", NN + "nn.cpp:83"),
    ("ops", "Only supported operators (unknown op -> assert(0), board hangs)", NN + "tf_util.cpp:74-100, tf.cpp:400-401"),
    ("dtype", "Activations/weights uint8, biases int32", NN + "tf_util.cpp:220-231, kernels/conv.m:86-89"),
    ("per_tensor", "Per-tensor quantization (only scale[0]/zero_point[0] read)", NN + "tf_util.cpp:196-202"),
    ("kernel", "Conv kernels 1x1 or 3x3, square", NN + "kernels/conv.m:738-745, kernels/conv.p:44-75"),
    ("stride", "stride_h == stride_w; 1x1 convs must use stride 1", NN + "nn_conv2d.cpp:130-167, kernels/conv.m:200-428"),
    ("dilation", "Dilation factor 1 (dilation is ignored)", NN + "nn_conv2d.cpp:122-172"),
    ("depthwise", "Depthwise: depth_multiplier 1, 3x3 kernel", NN + "nn_conv2d.cpp:59-60"),
    ("activation", "Fused activation NONE / RELU / RELU6", NN + "tf_util.cpp:54-72"),
    ("bias", "Every conv has a bias tensor", NN + "tf.cpp:218"),
    ("square", "Square feature maps (H == W)", NN + "nn_conv2d.cpp:44-48"),
    ("bias_range", "Bias minus activation offset fits ztachip's 2x14-bit split", NN + "nn_conv2d.cpp:358-378"),
    ("avgpool", "AVERAGE_POOL_2D is global with identical in/out quantization", NN + "kernels/fcn.m:111-152, nn_poolavg.cpp"),
    ("add", "ADD has 2 inputs of identical shape (no broadcast)", NN + "tf.cpp:319-337"),
    ("concat", "CONCATENATION: <16 inputs, inputs x outer_size <= 8", NN + "tf.cpp:231-249, nn_concat.cpp"),
    ("reshape", "RESHAPE only of spatially 1x1 tensors (pass-through, no data move)", NN + "nn_reshape.cpp, nn.cpp:100-116"),
    ("input", "Input is uint8 [1,H,H,3]", NN + "nn.cpp:276-289"),
    ("output", "Output tensor is uint8 (read by GetTop5)", NN + "nn_util.cpp:347-387"),
    ("channels", "Conv output channels multiple of 8 (weights padded to VECTOR_WIDTH)", "SW/base/zta.h:60-62"),
    ("labels", "Label file has one line per output class", NN + "nn_util.cpp:323-343"),
]
WARNING_RULES = {"channels"}


def _quantize_multiplier(m):
    """TFLite QuantizeMultiplier (same as ztachip tf_util.cpp)."""
    if m == 0:
        return 0, 0
    q, e = math.frexp(m)
    qf = int(round(q * (1 << 31)))
    if qf == (1 << 31):
        qf //= 2
        e += 1
    if e < -31:
        return 0, 0
    return qf, e


def _act_range_uint8(act, scale, zp):
    qmin, qmax = 0, 255
    if act == AF.RELU:
        return max(qmin, zp), qmax
    if act == AF.RELU6:
        return max(qmin, zp), min(qmax, zp + int(round(6.0 / scale)))
    return qmin, qmax


class Checker:
    def __init__(self, path, labels=None):
        self.path = Path(path)
        self.model = fu.read_model(str(path))
        self.labels = labels
        self.rules = {rid: Rule(rid, title, ref, "warning" if rid in WARNING_RULES else "error")
                      for rid, title, ref in RULES}
        self.ops = []
        self.info = {}

    def fail(self, rid, msg):
        r = self.rules[rid]
        if len(r.issues) < 12:
            r.issues.append(msg)
        elif len(r.issues) == 12:
            r.issues.append("... (more)")

    # helpers
    def T(self, i):
        return self.sg.tensors[i]

    def shape(self, i):
        return [int(s) for s in self.T(i).shape] if self.T(i).shape is not None else []

    def qp(self, i):
        q = self.T(i).quantization
        if q is None or q.scale is None or len(q.scale) == 0:
            return None, None
        return [float(s) for s in q.scale], [int(z) for z in q.zeroPoint]

    def run(self):
        m = self.model
        if m.version != 3:
            self.fail("schema", f"version {m.version}")
        if len(m.subgraphs) != 1:
            self.fail("subgraph", f"{len(m.subgraphs)} subgraphs")
        self.sg = sg = m.subgraphs[0]

        for k, op in enumerate(sg.operators):
            name = op_name(m, op)
            if name == "CUSTOM":
                name = "CUSTOM:" + bytes(m.operatorCodes[op.opcodeIndex].customCode).decode()
            self.ops.append(name)
            tag = f"op{k} {name}"
            if name not in SUPPORTED_OPS and name.removeprefix("CUSTOM:") not in SUPPORTED_CUSTOM:
                self.fail("ops", tag)
                continue
            if name in ("CONV_2D", "DEPTHWISE_CONV_2D"):
                self._conv(op, name, tag)
            elif name == "AVERAGE_POOL_2D":
                self._avgpool(op, tag)
            elif name == "ADD":
                ins = list(op.inputs)
                if len(ins) != 2 or self.shape(ins[0]) != self.shape(ins[1]):
                    self.fail("add", f"{tag}: inputs {[self.shape(i) for i in ins]}")
                self._activation(op, tag)
            elif name == "CONCATENATION":
                ins = list(op.inputs)
                shp = self.shape(ins[0])
                axis = op.builtinOptions.axis % len(shp)
                outer = int(np.prod(shp[:axis])) if axis else 1
                if len(ins) >= 16 or len(ins) * outer > 8:
                    self.fail("concat", f"{tag}: {len(ins)} inputs, outer size {outer}")
            elif name == "RESHAPE":
                s = self.shape(op.inputs[0])
                if len(s) == 4 and (s[1] != 1 or s[2] != 1):
                    self.fail("reshape", f"{tag}: input {s} is not spatially 1x1")

        self._tensors()
        self._io()
        return self

    def _activation(self, op, tag):
        opts = op.builtinOptions
        act = getattr(opts, "fusedActivationFunction", AF.NONE)
        if act not in (AF.NONE, AF.RELU, AF.RELU6):
            self.fail("activation", f"{tag}: {_AFNAMES.get(act, act)}")
        return act

    def _conv(self, op, name, tag):
        o = op.builtinOptions
        ins = list(op.inputs)
        fshape = self.shape(ins[1])          # conv: [O,kh,kw,I], depthwise: [1,kh,kw,O]
        kh, kw = fshape[1], fshape[2]
        ishape, oshape = self.shape(ins[0]), self.shape(op.outputs[0])
        if kh != kw or kh not in (1, 3):
            self.fail("kernel", f"{tag}: kernel {kh}x{kw}")
        if o.strideH != o.strideW:
            self.fail("stride", f"{tag}: stride {o.strideH}x{o.strideW}")
        if kh == 1 and o.strideW != 1:
            self.fail("stride", f"{tag}: 1x1 conv with stride {o.strideW}")
        if o.dilationHFactor != 1 or o.dilationWFactor != 1:
            self.fail("dilation", f"{tag}: dilation {o.dilationHFactor}x{o.dilationWFactor}")
        if name == "DEPTHWISE_CONV_2D":
            if o.depthMultiplier not in (0, 1) or oshape[3] != ishape[3]:
                self.fail("depthwise", f"{tag}: multiplier {o.depthMultiplier}")
            if kh != 3:
                self.fail("depthwise", f"{tag}: {kh}x{kw} depthwise")
        if ishape[1] != ishape[2] or oshape[1] != oshape[2]:
            self.fail("square", f"{tag}: {ishape} -> {oshape}")
        act = self._activation(op, tag)
        if len(ins) < 3 or ins[2] < 0:
            self.fail("bias", tag)
            return
        is_head = op is self.sg.operators[-1] or (
            len(self.sg.operators) >= 2 and op is self.sg.operators[-2]
            and op_name(self.model, self.sg.operators[-1]) in ("RESHAPE", "SOFTMAX", "LOGISTIC"))
        if oshape[3] % 8 and not is_head:
            self.fail("channels", f"{tag}: {oshape[3]} output channels")
        self._bias_range(ins, op.outputs[0], act, tag)

    def _bias_range(self, ins, out, act, tag):
        (si, _), (sw, _), (so, zo) = self.qp(ins[0]), self.qp(ins[1]), self.qp(out)
        if si is None or sw is None or so is None or self.T(ins[0]).type != TT.UINT8:
            return                                     # dtype rule already reports this
        mult, e = _quantize_multiplier(si[0] * sw[0] / so[0])
        if mult == 0:
            self.fail("bias_range", f"{tag}: zero requant multiplier")
            return
        amin, _ = _act_range_uint8(act, so[0], zo[0])
        act_bias = ((amin - zo[0]) * (1 << (31 - e))) // mult      # nn_conv2d.cpp:47
        bias = np.frombuffer(bytes(self.model.buffers[self.T(ins[2]).buffer].data), np.int32)
        v = bias.astype(np.int64) - act_bias
        rng = 1 << 14                                             # 1 << (DATA_BIT_WIDTH-2)
        hi = np.trunc(v / rng)
        if np.any(np.abs(hi) >= rng):
            self.fail("bias_range", f"{tag}: max |bias-offset| = {int(np.abs(v).max())} >= 2^28")

    def _avgpool(self, op, tag):
        o = op.builtinOptions
        ishape, oshape = self.shape(op.inputs[0]), self.shape(op.outputs[0])
        if not (o.filterHeight == ishape[1] and o.filterWidth == ishape[2] and oshape[1] == oshape[2] == 1):
            self.fail("avgpool", f"{tag}: pool {o.filterHeight}x{o.filterWidth} on {ishape} (must be global)")
        if self.qp(op.inputs[0]) != self.qp(op.outputs[0]):
            self.fail("avgpool", f"{tag}: input/output quantization differ")

    def _tensors(self):
        bias_ids = set()
        for op, name in zip(self.sg.operators, self.ops):
            if name in ("CONV_2D", "DEPTHWISE_CONV_2D") and len(op.inputs) > 2:
                bias_ids.add(op.inputs[2])
        used = {i for op in self.sg.operators for i in list(op.inputs) + list(op.outputs) if i >= 0}
        for i in sorted(used):
            t = self.T(i)
            nm = t.name.decode() if isinstance(t.name, bytes) else t.name
            want = TT.INT32 if i in bias_ids else TT.UINT8
            if t.type != want and not (t.type == TT.INT32 and self._is_reshape_shape(i)):
                self.fail("dtype", f"{nm}: {_TTNAMES.get(t.type)} (want {_TTNAMES[want]})")
            scale, _ = self.qp(i)
            if scale is not None and len(scale) != 1:
                self.fail("per_tensor", f"{nm}: {len(scale)} scales (per-channel)")

    def _is_reshape_shape(self, i):
        return any(name == "RESHAPE" and len(op.inputs) > 1 and op.inputs[1] == i
                   for op, name in zip(self.sg.operators, self.ops))

    def _io(self):
        sg = self.sg
        ii, oi = sg.inputs[0], sg.outputs[0]
        s, t = self.shape(ii), self.T(ii)
        scale, zp = self.qp(ii)
        self.info["input"] = dict(shape=s, dtype=_TTNAMES.get(t.type), scale=scale and scale[0], zero_point=zp and zp[0])
        if t.type != TT.UINT8 or len(s) != 4 or s[0] != 1 or s[1] != s[2] or s[3] != 3:
            self.fail("input", f"{_TTNAMES.get(t.type)} {s}")
        os_, ot = self.shape(oi), self.T(oi)
        oscale, ozp = self.qp(oi)
        self.info["output"] = dict(shape=os_, dtype=_TTNAMES.get(ot.type), scale=oscale and oscale[0], zero_point=ozp and ozp[0])
        if ot.type != TT.UINT8:
            self.fail("output", f"{_TTNAMES.get(ot.type)}")
        n_classes = os_[-1] if os_ else 0
        self.info["classes"] = n_classes
        if self.labels is not None:
            lines = [l for l in Path(self.labels).read_text().splitlines()]
            if len(lines) < n_classes:
                self.fail("labels", f"{len(lines)} labels for {n_classes} outputs")
        # Normalisation note: ztachip feeds raw camera bytes 0..255 (no preprocessing).
        if scale and zp is not None:
            self.info["camera_bytes_direct"] = abs(scale[0] - 1 / 255) < 1e-4 and zp[0] == 0

    # reporting
    @property
    def passed(self):
        return all(r.status != "FAIL" for r in self.rules.values())

    def results(self):
        return [dict(id=r.id, rule=r.title, status=r.status, issues=r.issues, ref=r.ref)
                for r in self.rules.values()
                if not (r.id == "labels" and self.labels is None)]

    def report(self):
        from collections import Counter
        lines = [f"ztachip compatibility: {self.path.name}",
                 f"  ops: {dict(Counter(self.ops))}",
                 f"  input: {self.info.get('input')}",
                 f"  output: {self.info.get('output')}"]
        if "camera_bytes_direct" in self.info:
            lines.append("  raw camera bytes can be fed directly: "
                         + ("yes (scale 1/255, zp 0)" if self.info["camera_bytes_direct"] else "no - needs input rescaling"))
        for r in self.results():
            lines.append(f"  [{r['status']:4}] {r['rule']}   ({r['ref']})")
            for msg in r["issues"]:
                lines.append(f"           - {msg}")
        lines.append("RESULT: " + ("COMPATIBLE" if self.passed else "NOT COMPATIBLE"))
        return "\n".join(lines)


def check(path, labels=None):
    return Checker(path, labels).run()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", nargs="+")
    ap.add_argument("--labels")
    a = ap.parse_args(argv)
    ok = True
    for p in a.model:
        if p == "ztachip-mobilenet":                 # alias for the model shipped with ztachip
            from .config import ZTACHIP_MOBILENET
            p = ZTACHIP_MOBILENET
        c = check(p, a.labels)
        print(c.report(), end="\n\n")
        ok &= c.passed
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
