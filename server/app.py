"""FastAPI backend for the showcase site.

Run:  uv run uvicorn server.app:app --port 8411
Serves /api/*, /static/figures/* and the built React app from web/dist.
"""
import base64
import io
import json
import re
import tempfile
import threading
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from zcnn import config as C
from zcnn.preprocess import center_crop_resize
from zcnn.quantize import tflite_path
from zcnn.tflite_runner import TFLiteModel, softmax, ztachip_top5
from zcnn.ztachip_check import RULES, check

DIST = C.ROOT / "web" / "dist"
MAX_TFLITE_BYTES = 30 * 1024 * 1024
ZTA_MODELS = ["zta_plain", "zta_resnet8", "zta_mobile"]
NICE = {"zta_plain": "ZTA-Plain", "zta_resnet8": "ZTA-ResNet8", "zta_mobile": "ZTA-Mobile",
        "chatgpt_baseline": "Baseline CNN"}

app = FastAPI(title="ZTA-CNN showcase")
_lock = threading.Lock()          # TF Lite interpreters are not thread-safe


# ---------------------------------------------------------------- helpers
def _json_file(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _png_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _available(name):
    return (C.ARTIFACTS / f"{name}.keras").exists() and tflite_path(name, "uint8").exists()


@lru_cache(maxsize=8)
def _load(name, mtime):
    import tensorflow as tf
    keras = tf.keras.models.load_model(C.ARTIFACTS / f"{name}.keras")
    return keras, TFLiteModel(tflite_path(name, "uint8"))


def models_for(name):
    if name not in ZTA_MODELS or not _available(name):
        raise HTTPException(404, f"model {name!r} not available")
    mtime = max((C.ARTIFACTS / f"{name}.keras").stat().st_mtime, tflite_path(name, "uint8").stat().st_mtime)
    return _load(name, mtime)


@lru_cache(maxsize=1)
def _test_set():
    from zcnn.data import load_cifar10
    _, _, (x, y) = load_cifar10()
    return x, y


def _gradcam_overlay(keras, img):
    import matplotlib.cm as cm
    from zcnn.explain import gradcam
    cam, _ = gradcam(keras, img[None])
    big = np.asarray(Image.fromarray(img).resize((256, 256), Image.NEAREST)).astype(np.float32)
    heat = np.asarray(Image.fromarray((cam[0] * 255).astype(np.uint8)).resize((256, 256), Image.BILINEAR)) / 255.0
    color = cm.jet(heat)[..., :3] * 255
    return _png_b64((0.55 * big + 0.45 * color).clip(0, 255).astype(np.uint8))


def _predict(name, img, want_gradcam):
    keras, u8 = models_for(name)
    with _lock:
        t = time.perf_counter()
        logits = keras(img[None].astype(np.float32) / 255.0, training=False).numpy()[0]
        t32 = (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        raw = u8.raw(img)
        tu8 = (time.perf_counter() - t) * 1e3
        cam = _gradcam_overlay(keras, img) if want_gradcam else None
    p32 = softmax(logits)
    pu8 = softmax((raw.astype(np.float32) - u8.out_zp) * u8.out_scale)
    return dict(
        model=name,
        preview=_png_b64(img),
        fp32=dict(probs=p32.round(5).tolist(), top=int(p32.argmax()), ms=round(t32, 3)),
        uint8=dict(probs=pu8.round(5).tolist(), top=int(pu8.argmax()), ms=round(tu8, 3),
                   raw=raw.astype(int).tolist(), out_scale=float(u8.out_scale), out_zero_point=int(u8.out_zp)),
        agree=bool(p32.argmax() == pu8.argmax()),
        ztachip_top5=[dict(index=i, label=C.CLASS_NAMES[i], raw=v, pct=pct) for i, v, pct in ztachip_top5(raw)],
        gradcam=cam,
    )


# ---------------------------------------------------------------- API
@app.get("/api/meta")
def meta():
    import platform
    M = _json_file(C.REPORTS / "metrics.json") or {}
    return dict(
        classes=C.CLASS_NAMES,
        deploy_model=C.DEPLOY_MODEL,
        models=[dict(id=m, name=NICE[m], available=_available(m), ztachip=m != "chatgpt_baseline") for m in C.MODELS],
        env=M.get("_env", dict(python=platform.python_version())),
        has_metrics=bool(M), has_robustness=(C.REPORTS / "robustness.json").exists(),
        has_golden=(C.ARTIFACTS / "hw" / "golden.json").exists(),
        rules=len(RULES),
        figures=sorted(p.name for p in C.FIGURES.glob("*.png")),
    )


@app.get("/api/metrics")
def metrics():
    return _json_file(C.REPORTS / "metrics.json") or JSONResponse({"available": False}, 404)


@app.get("/api/robustness")
def robustness():
    return _json_file(C.REPORTS / "robustness.json") or JSONResponse({"available": False}, 404)


@app.get("/api/history/{name}")
def history(name: str):
    if name not in C.MODELS:
        raise HTTPException(404)
    return _json_file(C.ARTIFACTS / f"{name}_history.json") or JSONResponse({"available": False}, 404)


@app.get("/api/golden")
def golden():
    g = _json_file(C.ARTIFACTS / "hw" / "golden.json")
    if not g:
        return JSONResponse({"available": False}, 404)
    x, _ = _test_set()
    for v in g["vectors"]:
        v["image"] = _png_b64(x[v["index"]])
    return g


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), model: str = Form(C.DEPLOY_MODEL), gradcam: bool = Form(False)):
    try:
        img = center_crop_resize(Image.open(io.BytesIO(await file.read())))
    except Exception:
        raise HTTPException(400, "could not read image")
    return _predict(model, img, gradcam)


@app.get("/api/samples")
def samples(n: int = 12, seed: int = 0):
    x, y = _test_set()
    idx = np.random.default_rng(seed).choice(len(x), min(max(n, 1), 48), replace=False)
    return [dict(index=int(i), label=C.CLASS_NAMES[y[i]], image=_png_b64(x[i])) for i in idx]


@app.get("/api/predict_index/{index}")
def predict_index(index: int, model: str = C.DEPLOY_MODEL, gradcam: bool = False):
    x, y = _test_set()
    if not 0 <= index < len(x):
        raise HTTPException(404)
    out = _predict(model, x[index], gradcam)
    out["true"] = int(y[index])
    return out


def _check_payload(path, label):
    c = check(path)
    return dict(name=label, compatible=c.passed, ops=dict(Counter(c.ops)), info=c.info, rules=c.results())


def _check_samples():
    s = {"ztachip_mobilenet": ("ztachip's MobileNet v2 (ships with ztachip)", C.ZTACHIP_MOBILENET)}
    for m in C.MODELS:
        if tflite_path(m, "uint8").exists():
            s[f"{m}_uint8"] = (f"{NICE[m]} - UINT8 (our pipeline)", tflite_path(m, "uint8"))
        if tflite_path(m, "int8_pc").exists():
            s[f"{m}_int8_pc"] = (f"{NICE[m]} - INT8 per-channel (TF default)", tflite_path(m, "int8_pc"))
    return s


@app.get("/api/check/samples")
def check_samples():
    return [dict(id=k, name=v[0]) for k, v in _check_samples().items()]


@app.get("/api/check/{sample}")
def check_sample(sample: str):
    s = _check_samples()
    if sample not in s:
        raise HTTPException(404)
    return _check_payload(s[sample][1], s[sample][0])


@app.post("/api/check")
async def check_upload(file: UploadFile = File(...)):
    data = await file.read(MAX_TFLITE_BYTES + 1)
    if len(data) > MAX_TFLITE_BYTES:
        raise HTTPException(413, "file too large")
    with tempfile.NamedTemporaryFile(suffix=".tflite") as f:
        f.write(data)
        f.flush()
        try:
            return _check_payload(f.name, file.filename or "uploaded.tflite")
        except Exception as e:
            raise HTTPException(400, f"not a readable TFLite model: {e}")


_REF_PART = re.compile(r"^(?P<path>[\w./-]+?)(?::(?P<a>\d+)(?:-(?P<b>\d+))?)?$")


@app.get("/api/source")
def source(ref: str):
    """Return ztachip source lines behind a checker rule, e.g. 'SW/apps/nn/tf.cpp:95-97, nn.cpp:83'."""
    root = (C.REPO / "SW").resolve()
    snippets, last_dir = [], ""
    for part in [p.strip() for p in ref.split(",") if p.strip()][:6]:
        m = _REF_PART.match(part)
        if not m:
            raise HTTPException(400, f"bad ref {part!r}")
        rel = m["path"]
        if not rel.startswith("SW/"):
            # Relative to the previous file's folder or one of its parents
            # ("SW/apps/nn/kernels/fcn.m:1, nn_poolavg.cpp" -> SW/apps/nn/nn_poolavg.cpp).
            base = last_dir.rstrip("/").split("/")
            cands = ["/".join(base[:k] + [rel]) for k in range(len(base), 0, -1)]
            rel = next((c for c in cands if (C.REPO / c).is_file()), cands[0] if cands else rel)
        last_dir = rel.rsplit("/", 1)[0] + "/"
        path = (C.REPO / rel).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise HTTPException(404, f"not found: {rel}")
        lines = path.read_text(errors="replace").splitlines()
        a = int(m["a"]) if m["a"] else 1
        b = int(m["b"]) if m["b"] else (a if m["a"] else min(len(lines), 40))
        lo, hi = max(1, a - 4), min(len(lines), b + 4)
        snippets.append(dict(file=rel, start=lo, highlight=[a, b],
                             lines=lines[lo - 1:hi][:120]))
    return snippets


@app.get("/api/model/{name}/layers")
def layers(name: str):
    M = _json_file(C.REPORTS / "metrics.json") or {}
    if name in M:
        return M[name]["profile"]
    if name in C.MODELS and (C.ARTIFACTS / f"{name}.keras").exists():
        from zcnn.profile import summary
        import tensorflow as tf
        return summary(tf.keras.models.load_model(C.ARTIFACTS / f"{name}.keras"))
    raise HTTPException(404)


@app.get("/api/model/{name}/tensors")
def tensors(name: str):
    from tensorflow.lite.tools import flatbuffer_utils as fu
    from zcnn.uint8_rewrite import INT8, op_name
    from zcnn.ztachip_check import _TTNAMES
    a_p, b_p = tflite_path(name, "int8_pt"), tflite_path(name, "uint8")
    if name not in C.MODELS or not (a_p.exists() and b_p.exists()):
        raise HTTPException(404)
    A, B = fu.read_model(str(a_p)), fu.read_model(str(b_p))
    rows = []
    for ta, tb in zip(A.subgraphs[0].tensors, B.subgraphs[0].tensors):
        qa, qb = ta.quantization, tb.quantization
        if qa is None or qa.scale is None or len(qa.scale) == 0 or ta.type != INT8:
            continue                                   # int32 biases are not rewritten
        nm = ta.name.decode() if isinstance(ta.name, bytes) else ta.name
        rows.append(dict(name=nm, shape=[int(s) for s in ta.shape],
                         int8=dict(dtype=_TTNAMES[ta.type], scale=float(qa.scale[0]), zero_point=int(qa.zeroPoint[0])),
                         uint8=dict(dtype=_TTNAMES[tb.type], scale=float(qb.scale[0]), zero_point=int(qb.zeroPoint[0]))))
    ops = [op_name(B, op) for op in B.subgraphs[0].operators]
    return dict(tensors=rows, ops=ops)


# ---------------------------------------------------------------- static + SPA
app.mount("/static/figures", StaticFiles(directory=C.FIGURES), name="figures")
if (DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/", include_in_schema=False)
@app.get("/{full_path:path}", include_in_schema=False)
def spa(full_path: str = ""):
    if full_path.startswith("api/"):
        raise HTTPException(404)
    f = (DIST / full_path).resolve()
    if full_path and f.is_file() and f.is_relative_to(DIST.resolve()):
        return FileResponse(f)
    index = DIST / "index.html"
    if index.exists():
        return FileResponse(index)
    return HTMLResponse("<p>Frontend not built. Run <code>make web-build</code>.</p>", 503)
