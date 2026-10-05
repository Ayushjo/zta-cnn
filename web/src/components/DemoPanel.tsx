import { Camera, ImageUp, Shuffle } from 'lucide-react'
import { motion } from 'motion/react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { STATIC, getJSON, postForm, useApi } from '@/lib/api'
import { MODEL_NAMES, VARIANT_COLOR, ZTA_MODELS } from '@/lib/format'
import type { Meta, Prediction, Sample } from '@/lib/types'
import { Empty, Loading, MonoLabel, Notice, SafetyLabel, Select, Tabs } from './ui'

type Mode = 'gallery' | 'upload' | 'webcam'
type Src = { kind: 'index'; index: number } | { kind: 'blob'; blob: Blob }

export function ProbBars({ pred, classes }: { pred: Prediction; classes: string[] }) {
  return (
    <div>
      <div className="mb-2 grid grid-cols-[96px_1fr_56px] gap-3">
        <MonoLabel>Class</MonoLabel>
        <div className="flex gap-4">
          <span className="mono-label flex items-center gap-1.5"><span className="h-[3px] w-3 rounded-full" style={{ background: VARIANT_COLOR.keras }} />FP32</span>
          <span className="mono-label flex items-center gap-1.5"><span className="h-[3px] w-3 rounded-full" style={{ background: VARIANT_COLOR.uint8 }} />UINT8 · ztachip</span>
        </div>
        <MonoLabel className="text-right">UINT8</MonoLabel>
      </div>
      {classes.map((c, i) => {
        const top = i === pred.uint8.top
        return (
          <div key={c} className="grid h-[30px] grid-cols-[96px_1fr_56px] items-center gap-3 border-b border-line/70 last:border-0">
            <div className={`truncate text-body ${top ? 'font-medium text-ink' : 'text-ink-2'}`}>{c}</div>
            <div className="space-y-[3px]">
              {(['fp32', 'uint8'] as const).map((v) => (
                <div key={v} className="h-[4px] overflow-hidden rounded-full bg-line">
                  <motion.div className="h-full rounded-full" style={{ background: VARIANT_COLOR[v === 'fp32' ? 'keras' : 'uint8'] }}
                    initial={false} animate={{ width: `${pred[v].probs[i] * 100}%` }} transition={{ duration: 0.25, ease: 'easeOut' }} />
                </div>
              ))}
            </div>
            <div className={`figure-sm text-right ${top ? 'text-ink' : 'text-ink-2'}`}>{(pred.uint8.probs[i] * 100).toFixed(1)}%</div>
          </div>
        )
      })}
    </div>
  )
}

export function OnboardConsole({ pred }: { pred: Prediction }) {
  return (
    <div className="rounded-[10px] border border-line px-3.5 py-3">
      <MonoLabel>On-board view · ztachip GetTop5() · score = (v·100) &gt;&gt; 8</MonoLabel>
      <pre className="mt-2 text-[12.5px] leading-[1.7] text-ink-2">
        {pred.ztachip_top5.map((t, k) => (
          <div key={t.index} className={k === 0 ? 'text-ink' : ''}>
            {`${k + 1}  ${t.label.padEnd(11)} raw ${String(t.raw).padStart(3)}   0.${String(t.pct).padStart(2, '0')}`}
          </div>
        ))}
      </pre>
    </div>
  )
}

function ResultView({ pred, classes, gradcam }: { pred: Prediction; classes: string[]; gradcam: boolean }) {
  const label = classes[pred.uint8.top]
  const correct = pred.true === undefined ? null : pred.true === pred.uint8.top
  return (
    <div className="grid gap-7 md:grid-cols-[176px_1fr]">
      <div className="space-y-4">
        <div>
          <img src={pred.preview} alt="32×32 model input" className="pixelated aspect-square w-full rounded-md" />
          <MonoLabel className="mt-1.5">Model input · 32×32</MonoLabel>
        </div>
        {gradcam && pred.gradcam && (
          <div>
            <img src={pred.gradcam} alt="Grad-CAM heatmap" className="aspect-square w-full rounded-md" />
            <MonoLabel className="mt-1.5">Grad-CAM · where it looked</MonoLabel>
          </div>
        )}
      </div>
      <div className="min-w-0 space-y-6">
        <div>
          <MonoLabel>Prediction · UINT8 model</MonoLabel>
          <div className="mt-1 flex flex-wrap items-baseline gap-x-3">
            <motion.span key={label} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.15 }}
              className="text-[32px] font-semibold capitalize leading-tight tracking-[-0.01em] text-ink">{label}</motion.span>
            <span className="figure text-ink-2">{(pred.uint8.probs[pred.uint8.top] * 100).toFixed(1)}%</span>
          </div>
          <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1">
            {correct !== null && (
              <SafetyLabel tone={correct ? 'safe' : 'danger'}>{correct ? 'Correct' : `Wrong · true class is ${classes[pred.true!]}`}</SafetyLabel>
            )}
            <SafetyLabel tone="neutral">{pred.agree ? 'FP32 agrees' : `FP32 says ${classes[pred.fp32.top]}`}</SafetyLabel>
          </div>
        </div>
        <ProbBars pred={pred} classes={classes} />
        <OnboardConsole pred={pred} />
        <div className="figure-sm text-ink-3">
          FP32 {pred.fp32.ms.toFixed(2)} ms  ·  UINT8 {pred.uint8.ms.toFixed(2)} ms  ·  laptop CPU, incl. Python overhead
        </div>
      </div>
    </div>
  )
}

function Webcam({ onFrame, busy }: { onFrame: (b: Blob) => void; busy: boolean }) {
  const video = useRef<HTMLVideoElement>(null)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [live, setLive] = useState(true)

  const start = async () => {
    try {
      setStream(await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'environment' }, audio: false }))
      setErr(null)
    } catch (e) {
      setErr(`Camera unavailable: ${(e as Error).message}`)
    }
  }
  const stop = () => { stream?.getTracks().forEach((t) => t.stop()); setStream(null) }
  useEffect(() => { if (video.current && stream) video.current.srcObject = stream }, [stream])
  useEffect(() => () => stream?.getTracks().forEach((t) => t.stop()), [stream])

  const grab = useCallback(() => {
    const v = video.current
    if (!v || !v.videoWidth) return
    const s = Math.min(v.videoWidth, v.videoHeight)
    const c = document.createElement('canvas')
    c.width = c.height = 224
    c.getContext('2d')!.drawImage(v, (v.videoWidth - s) / 2, (v.videoHeight - s) / 2, s, s, 0, 0, 224, 224)
    c.toBlob((b) => b && onFrame(b), 'image/jpeg', 0.9)
  }, [onFrame])

  // Next frame goes out as soon as the previous prediction returns.
  useEffect(() => {
    if (!stream || !live || busy) return
    const t = setTimeout(grab, 120)
    return () => clearTimeout(t)
  }, [stream, live, busy, grab])

  if (!stream)
    return (
      <div className="flex aspect-[4/3] flex-col items-center justify-center gap-3 rounded-[10px] border border-dashed border-line px-6 text-center">
        <Camera size={20} strokeWidth={1.5} className="text-ink-3" />
        <button className="btn btn-primary" onClick={start}>Start Camera</button>
        <p className="text-secondary text-ink-2">Frames go only to the backend on this laptop.</p>
        {err && <p className="text-secondary text-danger">{err}</p>}
      </div>
    )
  return (
    <div className="space-y-3">
      <div className="relative overflow-hidden rounded-[10px] bg-black">
        <video ref={video} autoPlay playsInline muted className="aspect-[4/3] w-full object-cover" />
        <div className="pointer-events-none absolute inset-y-0 left-1/2 aspect-square -translate-x-1/2 border border-white/80 shadow-[0_0_0_9999px_rgba(0,0,0,0.3)]" />
      </div>
      <div className="flex items-center gap-1">
        {live ? <SafetyLabel tone="danger" className="mr-2">Live</SafetyLabel> : <SafetyLabel tone="neutral" className="mr-2">Paused</SafetyLabel>}
        <button className="btn btn-quiet" onClick={() => setLive(!live)}>{live ? 'Pause' : 'Resume'}</button>
        {!live && <button className="btn btn-quiet" onClick={grab}>Capture Frame</button>}
        <button className="btn btn-quiet ml-auto" onClick={stop}>Stop Camera</button>
      </div>
      <p className="text-secondary text-ink-2">Only the centre square is classified. Objects work best when they fill it.</p>
    </div>
  )
}

export default function DemoPanel({ compact = false, autoRun = false }: { compact?: boolean; autoRun?: boolean }) {
  const { data: meta } = useApi<Meta>('/api/meta')
  const [model, setModel] = useState('zta_plain')
  const [mode, setMode] = useState<Mode>('gallery')
  const [gradcam, setGradcam] = useState(false)
  const [seed, setSeed] = useState(1)
  const [samples, setSamples] = useState<Sample[] | null>(null)
  const [pred, setPred] = useState<Prediction | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [selected, setSelected] = useState<number | null>(null)
  const last = useRef<Src | null>(null)

  const available = meta?.models.filter((m) => m.ztachip && m.available).map((m) => m.id) ?? []
  useEffect(() => {
    if (meta && !available.includes(model) && available.length) setModel(available.includes(meta.deploy_model) ? meta.deploy_model : available[0])
  }, [meta]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    getJSON<Sample[]>(`/api/samples?n=${compact ? 8 : 16}&seed=${seed}`).then((s) => setSamples(s.slice(0, compact ? 8 : 16))).catch(() => setSamples([]))
  }, [seed, compact])

  const run = useCallback(async (src: Src, m = model, g = gradcam) => {
    last.current = src
    setBusy(true)
    try {
      let p: Prediction
      if (src.kind === 'index') {
        // Grad-CAM is always computed so toggling it needs no new request (and works on the static site).
        p = await getJSON<Prediction>(`/api/predict_index/${src.index}?model=${m}&gradcam=true`)
      } else {
        const f = new FormData()
        f.append('file', src.blob, 'frame.jpg')
        f.append('model', m)
        f.append('gradcam', String(g))
        p = await postForm<Prediction>('/api/predict', f)
      }
      setPred(p)
      setErr(null)
    } catch (e) {
      setErr(`Prediction failed (${(e as Error).message}). Check that the backend is running.`)
    } finally {
      setBusy(false)
    }
  }, [model, gradcam])

  // In Present mode start with a classified image instead of an empty panel.
  useEffect(() => {
    if (autoRun && samples?.length && !last.current && available.length) {
      setSelected(samples[0].index)
      run({ kind: 'index', index: samples[0].index })
    }
  }, [samples, available.length]) // eslint-disable-line react-hooks/exhaustive-deps

  // Re-run the current input when the model or Grad-CAM changes (not for live frames).
  useEffect(() => {
    if (last.current && mode !== 'webcam') run(last.current, model, gradcam)
  }, [model, gradcam]) // eslint-disable-line react-hooks/exhaustive-deps

  const onFile = (f: File | undefined) => f && run({ kind: 'blob', blob: f })
  const classes = meta?.classes ?? []

  if (meta && !available.length)
    return <Empty title="No trained model yet">Run <code className="inline-code">make train quantize</code>, then reload.</Empty>

  return (
    <div className={`grid gap-0 ${compact ? 'lg:grid-cols-[360px_1fr]' : 'lg:grid-cols-[380px_1fr]'}`}>
      <div className="space-y-5 lg:border-r lg:border-line lg:pr-7">
        <Tabs value={mode} onChange={setMode} options={[
          { value: 'gallery', label: 'Test set' },
          { value: 'upload', label: 'Upload' },
          { value: 'webcam', label: 'Webcam' },
        ]} />
        <div className="flex flex-wrap items-center justify-between gap-2">
          <Select label="Model" value={model} onChange={setModel}
            options={(available.length ? available : ZTA_MODELS).map((m) => ({ value: m, label: MODEL_NAMES[m] }))} />
          <button className="chip" data-on={gradcam} onClick={() => setGradcam(!gradcam)} aria-pressed={gradcam}>Grad-CAM</button>
        </div>

        {mode === 'gallery' && (
          <div>
            <div className="mb-2 flex items-center justify-between">
              <span className="text-secondary text-ink-2">CIFAR-10 test images, unseen in training</span>
              <button className="btn btn-quiet !h-6 !text-secondary" onClick={() => setSeed(seed + 1)}><Shuffle size={12} /> Shuffle</button>
            </div>
            {!samples ? <Loading /> : (
              <div className="grid grid-cols-4 gap-1.5">
                {samples.map((s) => (
                  <button key={s.index} title={s.label} onClick={() => { setSelected(s.index); run({ kind: 'index', index: s.index }) }}
                    className={`overflow-hidden rounded-md transition-shadow ${selected === s.index ? 'shadow-[0_0_0_2px_var(--color-accent)]' : 'hover:shadow-[0_0_0_2px_var(--color-line)]'}`}>
                    <img src={s.image} alt={s.label} className="pixelated aspect-square w-full" />
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
        {STATIC && mode !== 'gallery' && (
          <Notice title="Runs on the presenter’s laptop">
            Upload and webcam classify new images live, which needs the local backend (<code className="inline-code">make web</code>). This online copy shows precomputed predictions for the test set.
          </Notice>
        )}
        {!STATIC && mode === 'upload' && (
          <label onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); onFile(e.dataTransfer.files[0]) }}
            className="flex aspect-[4/3] cursor-pointer flex-col items-center justify-center gap-2 rounded-[10px] border border-dashed border-line px-6 text-center transition-colors hover:border-ink-3">
            <ImageUp size={20} strokeWidth={1.5} className="text-ink-3" />
            <span className="text-body font-medium text-ink">Drop a photo, or click to choose</span>
            <span className="text-secondary text-ink-2">Centre-cropped and resized to 32×32 on the backend, as for the camera.</span>
            <input type="file" accept="image/*" className="hidden" onChange={(e) => onFile(e.target.files?.[0])} />
          </label>
        )}
        {!STATIC && mode === 'webcam' && <Webcam busy={busy} onFrame={(b) => run({ kind: 'blob', blob: b })} />}
      </div>

      <div className="min-h-[420px] pt-8 lg:pl-8 lg:pt-1">
        {err && <p className="mb-4 text-secondary text-danger">{err}</p>}
        {pred ? <ResultView pred={pred} classes={classes} gradcam={gradcam} /> : (
          <div className="flex h-full min-h-[360px] flex-col items-center justify-center text-center">
            <div className="text-body font-medium text-ink">Select an image</div>
            <p className="mt-1 max-w-sm text-secondary text-ink-2">Each prediction runs the FP32 model and the exact UINT8 file deployed on ztachip, side by side.</p>
            {busy && <Loading label="Classifying" />}
          </div>
        )}
      </div>
    </div>
  )
}
