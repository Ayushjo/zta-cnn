"""API tests. Prediction tests run only if the deploy model has been trained."""
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from server.app import _available, app
from zcnn import config as C
from zcnn.tflite_runner import ztachip_top5

client = TestClient(app)
needs_model = pytest.mark.skipif(not _available(C.DEPLOY_MODEL), reason="deploy model not trained")


def test_meta():
    r = client.get("/api/meta").json()
    assert r["classes"] == C.CLASS_NAMES and r["rules"] >= 20


def test_check_ztachip_mobilenet_passes():
    r = client.get("/api/check/ztachip_mobilenet").json()
    assert r["compatible"] is True
    assert r["ops"]["CONV_2D"] == 36


@needs_model
def test_check_tf_default_int8_fails():
    r = client.get(f"/api/check/{C.DEPLOY_MODEL}_int8_pc").json()
    assert r["compatible"] is False


def test_check_upload_rejects_garbage():
    r = client.post("/api/check", files={"file": ("x.tflite", b"not a model", "application/octet-stream")})
    assert r.status_code == 400


def test_source_returns_ztachip_code():
    r = client.get("/api/source", params={"ref": "SW/apps/nn/tf_util.cpp:74-100, tf.cpp:400-401"}).json()
    assert r[0]["file"] == "SW/apps/nn/tf_util.cpp" and r[1]["file"] == "SW/apps/nn/tf.cpp"
    assert any("ParseOpcode" in line for line in r[0]["lines"])


def test_source_relative_parts_resolve_against_parent_folders():
    r = client.get("/api/source", params={"ref": "SW/apps/nn/kernels/fcn.m:111-152, nn_poolavg.cpp"}).json()
    assert [s["file"] for s in r] == ["SW/apps/nn/kernels/fcn.m", "SW/apps/nn/nn_poolavg.cpp"]
    r = client.get("/api/source", params={"ref": "SW/apps/nn/kernels/conv.m:738-745, kernels/conv.p:44-75"}).json()
    assert r[1]["file"] == "SW/apps/nn/kernels/conv.p"


@pytest.mark.parametrize("ref", ["SW/../../etc/passwd", "../README.md", "/etc/passwd", "SW/apps/nn/../../../LICENSE.md"])
def test_source_rejects_traversal(ref):
    assert client.get("/api/source", params={"ref": ref}).status_code in (400, 404)


@needs_model
def test_predict_upload():
    img = Image.fromarray(np.random.default_rng(0).integers(0, 256, (120, 90, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    r = client.post("/api/predict", files={"file": ("a.png", buf.getvalue(), "image/png")},
                    data={"model": C.DEPLOY_MODEL, "gradcam": "true"}).json()
    assert len(r["fp32"]["probs"]) == 10 and len(r["uint8"]["probs"]) == 10
    assert r["gradcam"].startswith("data:image/png")
    top = ztachip_top5(np.array(r["uint8"]["raw"], np.uint8))
    assert [t["index"] for t in r["ztachip_top5"]] == [t[0] for t in top]


@needs_model
def test_predict_index_and_samples():
    s = client.get("/api/samples", params={"n": 4}).json()
    assert len(s) == 4
    r = client.get(f"/api/predict_index/{s[0]['index']}").json()
    assert C.CLASS_NAMES[r["true"]] == s[0]["label"]


@pytest.mark.parametrize("path", ["/", "/demo", "/present"])
def test_spa_fallback(path):
    r = client.get(path)
    assert r.status_code in (200, 503) and "html" in r.headers["content-type"]
    assert client.get("/api/does-not-exist").status_code == 404
