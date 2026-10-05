RUN = PYTHONWARNINGS=ignore uv run python -m

PAGES_BASE ?= /zta-cnn/

.PHONY: pages-build setup test train quantize check eval figures explain robustness export demo all clean web web-install web-build web-dev

PORT ?= 8411

setup:            ## create Python 3.12 venv with pinned deps
	uv sync --python 3.12

test:             ## toolchain tests (no training needed)
	PYTHONWARNINGS=ignore uv run pytest -q

train:            ## train all models (~1-2 h on Apple M4 CPU)
	$(RUN) zcnn.train

quantize:         ## FP32 / INT8-per-channel / INT8-per-tensor / UINT8 TFLite per model
	$(RUN) zcnn.quantize

check:            ## ztachip compatibility report for every UINT8 model + ztachip's MobileNet
	-$(RUN) zcnn.ztachip_check ztachip-mobilenet artifacts/*_uint8.tflite

eval:             ## accuracy, confusion, calibration, CPU latency -> reports/metrics.json
	$(RUN) zcnn.evaluate

explain:
	$(RUN) zcnn.explain

robustness:
	$(RUN) zcnn.robustness

figures:          ## all plots + reports/results.md
	$(RUN) zcnn.figures

export:           ## Phase II package: model, labels, golden vectors, on-board test
	$(RUN) zcnn.export_hw

demo:             ## old Streamlit UI (fallback)
	uv run streamlit run app/streamlit_app.py

web-install:      ## install frontend dependencies
	cd web && pnpm install

web-build:        ## build the site; bundles a snapshot of the results as an offline fallback
	mkdir -p web/src/data && rm -f web/src/data/*.json
	-cp reports/metrics.json web/src/data/metrics.json
	-cp reports/robustness.json web/src/data/robustness.json
	cd web && pnpm build

web: web-build    ## build + serve the showcase site on http://localhost:$(PORT)
	PYTHONWARNINGS=ignore TF_CPP_MIN_LOG_LEVEL=3 uv run uvicorn server.app:app --port $(PORT)

pages-build:      ## static copy for GitHub Pages in build/site (precomputed API, no backend)
	PYTHONWARNINGS=ignore TF_CPP_MIN_LOG_LEVEL=3 uv run python -m server.export_static build/static
	rm -rf build/site && mkdir -p build/site
	cd web && VITE_STATIC=1 VITE_BASE=$(PAGES_BASE) pnpm vite build --outDir ../build/site --emptyOutDir
	cp -R build/static/static-api build/static/figures build/site/
	cp build/site/index.html build/site/404.html
	touch build/site/.nojekyll

web-dev:          ## backend with reload + Vite dev server (http://localhost:5173)
	PYTHONWARNINGS=ignore TF_CPP_MIN_LOG_LEVEL=3 uv run uvicorn server.app:app --port $(PORT) --reload & cd web && pnpm dev

all: test train quantize eval explain robustness figures export check

clean:
	rm -rf artifacts/*.tflite artifacts/*.npz reports/figures/*.png reports/*.json reports/results.md artifacts/hw
