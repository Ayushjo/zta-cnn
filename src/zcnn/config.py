"""Single source of truth for paths, seeds and hyper-parameters."""
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[2]          # project root


def _find_ztachip():
    """ztachip checkout: $ZTACHIP_DIR, the third_party/ztachip submodule, or the parent folder."""
    for p in (os.environ.get("ZTACHIP_DIR"), ROOT / "third_party" / "ztachip", ROOT.parent):
        if p and (Path(p) / "SW" / "apps" / "nn").is_dir():
            return Path(p).resolve()
    return ROOT / "third_party" / "ztachip"


REPO = _find_ztachip()                              # ztachip sources (read-only)
_OUT = Path(os.environ.get("ZCNN_OUT", ROOT))      # override for smoke tests
ARTIFACTS = _OUT / "artifacts"
REPORTS = _OUT / "reports"
FIGURES = REPORTS / "figures"
for _d in (ARTIFACTS, REPORTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# Reference model shipped with ztachip; known to run on the accelerator.
ZTACHIP_MOBILENET = REPO / "SW" / "fs" / "mobilenet_v2_1_0_224_quant.tflite"

SEED = 42
NUM_CLASSES = 10
IMG_SIZE = 32
VAL_SIZE = 5000
REP_SAMPLES = 500           # calibration images for post-training quantization

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

TRAIN = dict(
    batch_size=128,
    epochs=40,
    lr=0.05,
    momentum=0.9,
    weight_decay=5e-4,
    label_smoothing=0.1,
    early_stop_patience=10,
)

# Main model to deploy on ztachip.
DEPLOY_MODEL = "zta_plain"
MODELS = ["zta_plain", "zta_resnet8", "zta_mobile", "chatgpt_baseline"]
