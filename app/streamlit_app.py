"""Demo UI.  Run:  uv run streamlit run app/streamlit_app.py"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from zcnn import config as C                                   # noqa: E402
from zcnn.quantize import tflite_path                          # noqa: E402
from zcnn.preprocess import center_crop_resize                 # noqa: E402
from zcnn.tflite_runner import TFLiteModel, softmax, ztachip_top5  # noqa: E402
from zcnn.ztachip_check import check                           # noqa: E402

st.set_page_config(page_title="CIFAR-10 on ztachip", layout="wide")
NICE = {"zta_plain": "ZTA-Plain", "zta_resnet8": "ZTA-ResNet8", "zta_mobile": "ZTA-Mobile",
        "chatgpt_baseline": "Baseline CNN (MaxPool+Dense)"}


@st.cache_resource
def load_models(name):
    import tensorflow as tf
    keras = tf.keras.models.load_model(C.ARTIFACTS / f"{name}.keras")
    return keras, TFLiteModel(tflite_path(name, "uint8"))


@st.cache_data
def metrics():
    p = C.REPORTS / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


st.title("Hardware-aware CIFAR-10 CNN for the ztachip FPGA accelerator")
st.caption("Phase I (software): model co-designed for ztachip's operator set, quantized to the uint8 "
           "format its TFLite engine executes, and verified with our compatibility checker. "
           "ztachip itself is an open-source accelerator by Vuong Nguyen (Apache-2.0); "
           "on-FPGA deployment is Phase II.")

trained = [m for m in C.MODELS if (C.ARTIFACTS / f"{m}.keras").exists() and tflite_path(m, "uint8").exists()]
tab1, tab2, tab3, tab4 = st.tabs(["Live classify", "Results", "ztachip checker", "Architecture"])

with tab1:
    if not trained:
        st.warning("No trained models yet - run `make train quantize`.")
    else:
        zta_models = [m for m in trained if m != "chatgpt_baseline"]
        name = st.selectbox("Model", zta_models, format_func=NICE.get,
                            index=zta_models.index(C.DEPLOY_MODEL) if C.DEPLOY_MODEL in zta_models else 0)
        src = st.radio("Input", ["Upload image", "Camera", "Random CIFAR-10 test image"], horizontal=True)
        img = None
        true = None
        if src == "Upload image":
            f = st.file_uploader("Image", type=["png", "jpg", "jpeg", "bmp", "webp"])
            if f:
                img = center_crop_resize(Image.open(f))
        elif src == "Camera":
            f = st.camera_input("Take a picture")
            if f:
                img = center_crop_resize(Image.open(f))
        else:
            from zcnn.data import load_cifar10
            if st.button("New image") or "rand_idx" not in st.session_state:
                st.session_state.rand_idx = int(np.random.randint(10000))
            _, _, (xt, yt) = st.cache_data(load_cifar10)()
            img, true = xt[st.session_state.rand_idx], C.CLASS_NAMES[yt[st.session_state.rand_idx]]

        if img is not None:
            keras, u8 = load_models(name)
            t = time.perf_counter()
            p32 = softmax(keras.predict(img[None] / 255.0, verbose=0)[0])
            t32 = (time.perf_counter() - t) * 1e3
            t = time.perf_counter()
            raw = u8.raw(img)
            tu8 = (time.perf_counter() - t) * 1e3
            pu8 = softmax((raw.astype(np.float32) - u8.out_zp) * u8.out_scale)

            c1, c2, c3 = st.columns([1, 2, 2])
            with c1:
                st.image(img, caption="What the model sees (32x32)", width=160)
                if true:
                    st.write(f"True label: **{true}**")
            for col, title, p, ms in ((c2, "FP32 (Keras, laptop CPU)", p32, t32),
                                      (c3, "UINT8 (model deployed on ztachip, run in TFLite)", pu8, tu8)):
                with col:
                    k = int(p.argmax())
                    st.subheader(title)
                    st.metric("Prediction", C.CLASS_NAMES[k])
                    st.write(f"confidence **{p[k]:.1%}**")
                    st.bar_chart(pd.DataFrame({"probability": p}, index=C.CLASS_NAMES), height=220)
                    st.caption(f"{ms:.2f} ms incl. Python overhead")
            st.markdown("**On-board view** - what ztachip's `GetTop5` would print for this uint8 output "
                        "(raw uint8 codes, `(v*100)>>8`, as in `micropython/examples/image_classification.py`):")
            st.code("\n".join(f"{C.CLASS_NAMES[i]:<11} raw={v:3d}  0.{pct:02d}" for i, v, pct in ztachip_top5(raw)))

with tab2:
    M = metrics()
    if not M:
        st.info("Run `make eval figures` to populate results.")
    else:
        rows = []
        for m in [m for m in C.MODELS if m in M]:
            v, p = M[m]["variants"], M[m]["profile"]
            rows.append({"model": NICE[m], "params": p["params"], "MMACs": round(p["macs"] / 1e6, 1),
                         "FP32 %": round(v["keras"]["top1"] * 100, 2),
                         "INT8 per-ch %": round(v["int8_pc"]["top1"] * 100, 2),
                         "UINT8 ztachip %": round(v["uint8"]["top1"] * 100, 2),
                         "UINT8 KiB": round(v["uint8"]["size_kib"], 1),
                         "TFLite UINT8 ms": round(M[m]["latency_ms"]["tflite_uint8_1thread"][0], 3),
                         "ztachip-compatible": "yes" if M[m]["ztachip_compatible"] else "NO"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(f"CIFAR-10 test set (10,000 images). CPU: {M['_env']['cpu']}.")
        figs = sorted(C.FIGURES.glob("*.png"))
        order = ["training_curves", "quantization_accuracy", "accuracy_vs_macs", "cpu_latency",
                 f"confusion_{C.DEPLOY_MODEL}_uint8", f"per_class_{C.DEPLOY_MODEL}", f"samples_{C.DEPLOY_MODEL}",
                 f"mistakes_{C.DEPLOY_MODEL}", f"gradcam_{C.DEPLOY_MODEL}", "robustness", f"reliability_{C.DEPLOY_MODEL}"]
        for stem in order:
            f = C.FIGURES / f"{stem}.png"
            if f.exists():
                st.image(str(f), caption=stem.replace("_", " "))
        with st.expander("All figures"):
            for f in figs:
                st.image(str(f), caption=f.stem)

with tab3:
    st.markdown("Upload any `.tflite`. Each rule comes from reading ztachip's TFLite importer; "
                "unsupported models otherwise **hang the board** (`assert(0)`) or compute garbage silently.")
    samples = {"ztachip's own MobileNet v2 (positive control)": C.ZTACHIP_MOBILENET}
    for m in trained:
        samples[f"{NICE[m]} - UINT8"] = tflite_path(m, "uint8")
        samples[f"{NICE[m]} - INT8 per-channel (TF default)"] = tflite_path(m, "int8_pc")
    choice = st.selectbox("Sample model", ["(upload)"] + list(samples))
    path = None
    if choice == "(upload)":
        f = st.file_uploader("TFLite model", type=["tflite"])
        if f:
            tmp = Path(tempfile.mkdtemp()) / f.name
            tmp.write_bytes(f.getvalue())
            path = tmp
    else:
        path = samples[choice]
    if path:
        c = check(path)
        (st.success if c.passed else st.error)("COMPATIBLE with ztachip" if c.passed else "NOT compatible with ztachip")
        from collections import Counter
        st.write("Operators:", dict(Counter(c.ops)))
        st.write("Input:", c.info["input"], " Output:", c.info["output"])
        df = pd.DataFrame([{"status": r["status"], "rule": r["rule"],
                            "issues": "; ".join(r["issues"][:4]), "ztachip source": r["ref"]} for r in c.results()])
        st.dataframe(df, hide_index=True, use_container_width=True)

with tab4:
    doc = C.ROOT / "docs" / "architecture.md"
    if doc.exists():
        st.markdown(doc.read_text())
    M = metrics()
    if C.DEPLOY_MODEL in M:
        st.subheader(f"{NICE[C.DEPLOY_MODEL]} layer table")
        st.dataframe(pd.DataFrame(M[C.DEPLOY_MODEL]["profile"]["layers"]), hide_index=True, use_container_width=True)
