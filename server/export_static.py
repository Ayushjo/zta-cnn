"""Freeze every API response the site needs into JSON files for static hosting.

Usage:  uv run python -m server.export_static OUT_DIR
Writes OUT_DIR/static-api/** and OUT_DIR/figures/*.png. Live upload and webcam
prediction are not exported; the static site points visitors to the local server.
"""
import json
import re
import shutil
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from server.app import ZTA_MODELS, _available, app
from zcnn import config as C
from zcnn.ztachip_check import RULES

SEEDS = (1, 2, 3)
N_SAMPLES = 16


def slug(ref: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", ref).strip("_")


def source_refs():
    """Every ztachip source reference the site can open: checker rules + refs written in the UI."""
    refs = {r for _, _, r in RULES}
    for f in (C.ROOT / "web" / "src").rglob("*.tsx"):
        refs |= set(re.findall(r"['\"](SW/[^'\"]+)['\"]", f.read_text()))
    return sorted(r for r in refs if not r.endswith("/"))      # skip folder names in prose


def main(out):
    out = Path(out)
    api = out / "static-api"
    if api.exists():
        shutil.rmtree(api)
    client = TestClient(app)

    def save(path, rel):
        r = client.get(path)
        if r.status_code != 200:
            print(f"  skip {path} ({r.status_code})")
            return None
        f = api / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(r.json(), separators=(",", ":")))
        return r.json()

    for name in ("meta", "metrics", "robustness", "golden"):
        save(f"/api/{name}", f"{name}.json")
    for m in C.MODELS:
        save(f"/api/history/{m}", f"history/{m}.json")
        save(f"/api/model/{m}/layers", f"model/{m}/layers.json")
        save(f"/api/model/{m}/tensors", f"model/{m}/tensors.json")

    samples = save("/api/check/samples", "check/samples.json") or []
    for s in samples:
        save(f"/api/check/{s['id']}", f"check/{s['id']}.json")

    for ref in source_refs():
        save(f"/api/source?ref={ref}", f"source/{slug(ref)}.json")

    models = [m for m in ZTA_MODELS if _available(m)]
    n_pred = 0
    for seed in SEEDS:
        imgs = save(f"/api/samples?n={N_SAMPLES}&seed={seed}", f"samples-{seed}.json") or []
        for s in imgs:
            for m in models:
                save(f"/api/predict_index/{s['index']}?model={m}&gradcam=true", f"predict/{m}/{s['index']}.json")
                n_pred += 1

    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    for p in C.FIGURES.glob("*.png"):
        shutil.copy(p, figs / p.name)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"[export_static] {n_pred} predictions, {len(source_refs())} source refs, {size:.1f} MB -> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build/static")
