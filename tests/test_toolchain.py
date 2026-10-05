"""Toolchain tests that need no trained weights (untrained models, random calibration)."""
import numpy as np
import pytest

from zcnn import config as C
from zcnn.models import build
from zcnn.quantize import convert
from zcnn.tflite_runner import TFLiteModel, ztachip_top5
from zcnn.uint8_rewrite import INT8, UINT8
from zcnn.ztachip_check import check


def rep():
    rng = np.random.default_rng(0)
    for _ in range(16):
        x = rng.random((1, 32, 32, 3), dtype=np.float32)
        x[0, 0, 0], x[0, 0, 1] = 0.0, 1.0
        yield [x]


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    d = tmp_path_factory.mktemp("tfl")
    out = {}
    for name in C.MODELS:
        m = build(name)
        for v in ("int8_pc", "int8_pt", "uint8"):
            p = d / f"{name}_{v}.tflite"
            p.write_bytes(convert(m, v, rep))
            out[name, v] = p
    return out


def test_checker_accepts_ztachip_mobilenet():
    """Positive control: the model ztachip ships with must pass every rule."""
    c = check(C.ZTACHIP_MOBILENET)
    assert c.passed, c.report()


@pytest.mark.parametrize("name", ["zta_plain", "zta_resnet8", "zta_mobile"])
def test_zta_models_are_compatible(converted, name):
    c = check(converted[name, "uint8"])
    assert c.passed, c.report()
    assert set(c.ops) <= {"CONV_2D", "DEPTHWISE_CONV_2D", "ADD", "AVERAGE_POOL_2D", "RESHAPE"}
    assert c.info["camera_bytes_direct"], c.info["input"]


def test_baseline_is_rejected(converted):
    c = check(converted["chatgpt_baseline", "uint8"])
    bad = {r["id"] for r in c.results() if r["status"] == "FAIL"}
    assert {"ops", "reshape"} <= bad
    assert any("MAX_POOL_2D" in i for i in c.rules["ops"].issues)
    assert any("FULLY_CONNECTED" in i for i in c.rules["ops"].issues)


def test_tf_default_int8_is_rejected(converted):
    c = check(converted["zta_plain", "int8_pc"])
    bad = {r["id"] for r in c.results() if r["status"] == "FAIL"}
    assert {"dtype", "per_tensor", "input", "output"} <= bad


def test_uint8_rewrite_parameters_are_exact(converted):
    """Every int8 tensor maps to uint8 with identical scale and zero point + 128."""
    from tensorflow.lite.tools import flatbuffer_utils as fu
    a = fu.read_model(str(converted["zta_plain", "int8_pt"])).subgraphs[0].tensors
    b = fu.read_model(str(converted["zta_plain", "uint8"])).subgraphs[0].tensors
    for ta, tb in zip(a, b):
        if ta.quantization is not None and ta.quantization.scale is not None and ta.type == INT8:
            assert tb.type == UINT8
            np.testing.assert_array_equal(ta.quantization.scale, tb.quantization.scale)
            np.testing.assert_array_equal(np.asarray(ta.quantization.zeroPoint) + 128, tb.quantization.zeroPoint)


@pytest.mark.parametrize("name", ["zta_plain", "zta_resnet8", "zta_mobile"])
def test_uint8_matches_int8_within_rounding(converted, name):
    """The rewrite is exact in representation, but TFLite's int8 and legacy uint8 kernels
    round requantisation slightly differently, so outputs may differ by a few LSB."""
    i8, u8 = TFLiteModel(converted[name, "int8_pt"]), TFLiteModel(converted[name, "uint8"])
    rng = np.random.default_rng(1)
    for _ in range(20):
        img = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        a = i8.raw(img).astype(np.int16) + 128
        b = u8.raw(img).astype(np.int16)
        assert np.abs(a - b).max() <= 3


def test_uint8_input_is_raw_camera_bytes(converted):
    m = TFLiteModel(converted["zta_plain", "uint8"])
    img = np.arange(32 * 32 * 3, dtype=np.uint32).reshape(32, 32, 3).astype(np.uint8)
    np.testing.assert_array_equal(m.quantize_input(img)[0], img)


def test_top5_matches_ztachip_semantics():
    raw = np.array([5, 200, 7, 200, 1, 0, 9, 3, 100, 2], np.uint8)
    top = ztachip_top5(raw)
    assert [t[0] for t in top] == [3, 1, 8, 6, 2]          # ties -> higher index first
    assert top[0][2] == (200 * 100) >> 8
