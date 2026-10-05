# web

The showcase site: React 19, Vite 8, Tailwind CSS 4, Recharts and Motion, styled with the "Calm" design system. That means a warm paper background, ink text, one violet accent, hairlines instead of cards, mono labels, safety colours shown only as a dot plus a word, and a seven-colour data palette used only in charts.

```bash
pnpm install
pnpm dev            # http://localhost:5173, proxies /api to the backend on :8411
pnpm build          # type-check + bundle into dist/
```

## Pages

| Route | Content |
|---|---|
| `/` | Overview: the problem, approach, contributions, credit |
| `/demo` | Live classification (test set, upload, webcam), FP32 next to UINT8, Grad-CAM, the on-board GetTop5 view |
| `/design` | Each ztachip limitation and the design decision it forced (with source), model zoo, compute treemap |
| `/quantization` | The four stages, why the uint8 rewrite is exact, tensor inspector |
| `/checker` | The compatibility checker with links to ztachip source |
| `/results` | Accuracy, training, errors, calibration, latency, robustness, Grad-CAM |
| `/hardware` | ZedBoard plan, porting steps, golden vectors, timeline |
| `/present` | Full-screen slides (←/→, `F` for full screen, `#N` jumps to slide N) |

Team names for the title slide are set in `src/content.ts`.

## Static build

`VITE_STATIC=1` makes the app read pre-exported JSON instead of calling the backend (`src/lib/api.ts`), and `VITE_BASE` sets the base path. `make pages-build` at the repo root produces the GitHub Pages copy in `build/site`.
