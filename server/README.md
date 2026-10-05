# server

FastAPI backend for the showcase site. It imports the `zcnn` toolchain directly, so the site shows exactly what the evaluation measured.

```bash
uv run uvicorn server.app:app --port 8411      # or: make web
```

It serves `/api/*`, the figures under `/static/figures/*`, and the built frontend from `web/dist` (any unknown path returns `index.html`).

| Endpoint | Returns |
|---|---|
| `GET /api/meta` | classes, available models, environment |
| `GET /api/metrics`, `/api/robustness`, `/api/history/{model}` | evaluation outputs from `reports/` and `artifacts/` |
| `POST /api/predict` | FP32 and UINT8 predictions for an uploaded image, with an optional Grad-CAM overlay |
| `GET /api/samples`, `/api/predict_index/{i}` | CIFAR-10 test images and their predictions |
| `GET /api/check/{sample}`, `POST /api/check` | ztachip compatibility report for a sample or an uploaded `.tflite` |
| `GET /api/source?ref=…` | the ztachip source lines behind a rule (restricted to `third_party/ztachip/SW`) |
| `GET /api/model/{name}/layers`, `/tensors` | per-layer cost and int8-vs-uint8 quantization parameters |
| `GET /api/golden` | Phase II golden test vectors |

`export_static.py` calls every GET endpoint and saves the answers as JSON files for the GitHub Pages copy (`make pages-build`). Live upload and webcam prediction are not exported.

Tests: `uv run pytest server`.
