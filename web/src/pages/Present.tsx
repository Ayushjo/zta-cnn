import { ChevronLeft, ChevronRight, Expand, X } from 'lucide-react'
import { AnimatePresence, motion } from 'motion/react'
import { useCallback, useEffect, useLayoutEffect, useState, type ReactNode } from 'react'
import { useNavigate } from 'react-router'
import CheckerView from '@/components/CheckerView'
import DemoPanel from '@/components/DemoPanel'
import { Logo } from '@/components/Layout'
import Pipeline from '@/components/Pipeline'
import ZedBoard from '@/components/ZedBoard'
import { AccuracyByVariant, ConfusionMatrix, RobustnessChart } from '@/components/charts'
import { SafetyLabel, SourceButton } from '@/components/ui'
import { PROJECT } from '@/content'
import { figureUrl, useApi, useMetrics } from '@/lib/api'
import { ALL_MODELS, MODEL_NAMES } from '@/lib/format'
import type { Meta, Robustness } from '@/lib/types'
import { DESIGN_RULES } from './Design'
import { PORT_STEPS, TIMELINE } from './Hardware'
import { CREDITS, UNSUPPORTED } from './Overview'
import { STEPS, STEP_VARIANT } from './Quantization'
import { ResultsTable } from './Results'

const W = 1600, H = 900

// Slide-scale Calm type: label 15 mono, title 48, body 25, secondary 20.
const Label = ({ children, className = '' }: { children: ReactNode; className?: string }) => (
  <div className={`font-mono text-[15px] font-medium uppercase tracking-[0.06em] text-ink-3 ${className}`}>{children}</div>
)

function Slide({ n, total, kicker, title, children }: { n: number; total: number; kicker: string; title: string; children: ReactNode }) {
  return (
    <div className="flex h-full flex-col px-[104px] pb-[84px] pt-[72px]">
      <Label>{String(n).padStart(2, '0')} / {total} · {kicker}</Label>
      <h2 className="mt-4 max-w-[1300px] text-[48px] font-semibold leading-[1.12] tracking-[-0.015em] text-ink">{title}</h2>
      <div className="mt-6 border-t border-line" />
      <div className="mt-9 min-h-0 flex-1">{children}</div>
    </div>
  )
}

const Points = ({ items }: { items: ReactNode[] }) => (
  <ul>
    {items.map((it, i) => (
      <li key={i} className="grid grid-cols-[56px_1fr] items-baseline border-b border-line/70 py-[18px] text-[25px] leading-snug text-ink-2 last:border-0 [&_strong]:font-medium [&_strong]:text-ink">
        <span className="font-mono text-[15px] text-ink-3">{String(i + 1).padStart(2, '0')}</span>
        <span>{it}</span>
      </li>
    ))}
  </ul>
)

function useSlides() {
  const { data: M } = useMetrics()
  const { data: meta } = useApi<Meta>('/api/meta')
  const { data: R } = useApi<Robustness>('/api/robustness', 'robustness')
  const dm = meta?.deploy_model ?? 'zta_plain'
  const m = M?.[dm]
  const models = M ? ALL_MODELS.filter((x) => x in M) : []
  const classes = meta?.classes ?? []
  const acc = m ? `${(m.variants.uint8.top1 * 100).toFixed(1)}%` : '—'

  const body: { kicker: string; title: string; node: ReactNode }[] = [
    {
      kicker: 'Motivation', title: 'AI at the edge needs hardware the model is designed for',
      node: <Points items={[
        <>Classifying one 32×32 image takes <strong>millions of multiply-accumulates</strong>; cameras, drones and sensors cannot carry a GPU.</>,
        <>FPGAs run many MAC units in parallel at low power and can be reprogrammed as models change.</>,
        <>An accelerator only runs the operators and number formats it was built for, so <strong>the model must be designed for the hardware</strong>.</>,
        <>Our goal: an image classifier that provably runs on a real open-source FPGA accelerator, and the toolchain that proves it.</>,
      ]} />,
    },
    {
      kicker: 'Platform', title: 'ztachip, an open-source FPGA AI accelerator',
      node: (
        <div className="grid grid-cols-[1.15fr_1fr] gap-20">
          <Points items={[
            <>By Vuong Nguyen, Apache-2.0: RTL, compiler and runtime are open.</>,
            <>VexRiscv RISC-V CPU plus a tensor engine of 4 cores × 16 threads; reference board Arty A7.</>,
            <>Runs quantized <strong>TensorFlow Lite</strong> models such as MobileNet and SSD.</>,
            <>It is our target, <strong>not our work</strong>. We built everything needed to put our own model on it.</>,
          ]} />
          <div>
            <Label className="mb-3">Its TFLite engine cannot run</Label>
            <ul className="grid grid-cols-2 gap-x-8">
              {UNSUPPORTED.map((x) => (
                <li key={x} className="border-b border-line/70 py-3"><SafetyLabel tone="danger" className="!text-[19px]"><span className="font-mono">{x}</span></SafetyLabel></li>
              ))}
            </ul>
          </div>
        </div>
      ),
    },
    {
      kicker: 'Key finding', title: 'A normal CNN would silently hang the board',
      node: (
        <div className="grid grid-cols-[1.2fr_1fr] gap-20">
          <Points items={[
            <>An unknown operator reaches <code className="text-[21px] text-ink">assert(0)</code> in the importer.</>,
            <>On this bare-metal system <code className="text-[21px] text-ink">_exit()</code> is an infinite loop: no message, the board freezes.</>,
            <>int8 or per-channel models do not freeze; they run and compute <strong>wrong numbers</strong>.</>,
            <>The textbook CNN and TensorFlow’s default export both fall into these traps.</>,
          ]} />
          <div>
            <Label className="mb-3">Evidence in ztachip</Label>
            {[['Unknown operator → assert(0)', 'SW/apps/nn/tf.cpp:398-404'], ['Operator whitelist', 'SW/apps/nn/tf_util.cpp:74-100'], ['_exit() loops forever', 'SW/base/newlib.c:119-124'], ['Per-channel scales: error ignored', 'SW/apps/nn/tf_util.cpp:196-202']].map(([t, r]) => (
              <div key={r} className="flex items-baseline justify-between gap-4 border-b border-line/70 py-4">
                <span className="text-[21px] text-ink">{t}</span>
                <span className="text-[16px]"><SourceButton ref_={r} label={r.split('/').pop()!} /></span>
              </div>
            ))}
          </div>
        </div>
      ),
    },
    {
      kicker: 'Design', title: 'Every layer choice comes from ztachip’s code',
      node: (
        <table className="w-full">
          <thead><tr className="border-b border-line"><th className="pb-3 text-left"><Label>ztachip limitation</Label></th><th className="pb-3 text-left"><Label>Our decision</Label></th></tr></thead>
          <tbody>{DESIGN_RULES.slice(0, 7).map((r) => (
            <tr key={r.limit} className="border-b border-line/70 last:border-0">
              <td className="w-[38%] py-[15px] pr-8 text-[21px] font-medium text-ink">{r.limit}</td>
              <td className="py-[15px] text-[20px] text-ink-2">{r.decision}</td>
            </tr>
          ))}</tbody>
        </table>
      ),
    },
    {
      kicker: 'Approach', title: 'From training to a board-ready file',
      node: (
        <div className="space-y-14">
          <Pipeline large />
          <div className="grid grid-cols-4">
            {[['ZTA-Plain', 'Six 3×3 convolutions · deployed'], ['ZTA-ResNet8', 'MLPerf Tiny ResNet-8, adapted'], ['ZTA-Mobile', 'Depthwise-separable'], ['Baseline CNN', 'MaxPool and Dense · reference']].map(([n, d], i) => (
              <div key={n} className={`pr-6 ${i > 0 ? 'border-l border-line pl-6' : ''}`}>
                <div className="text-[25px] font-medium text-ink">{n}</div>
                <div className="mt-1 text-[19px] text-ink-2">{d}</div>
                <div className="mt-3">{i < 3 ? <SafetyLabel tone="safe" className="!text-[18px]">Runs on ztachip</SafetyLabel> : <SafetyLabel tone="danger" className="!text-[18px]">Would hang</SafetyLabel>}</div>
              </div>
            ))}
          </div>
        </div>
      ),
    },
    {
      kicker: 'Quantization', title: 'TensorFlow 2 cannot emit uint8, so we rewrite the file',
      node: (
        <div>
          <div className="grid grid-cols-4 border-b border-line">
            {STEPS.map((s, i) => (
              <div key={s.k} className={`pb-6 pr-6 ${i > 0 ? 'border-l border-line pl-6' : ''}`}>
                <Label>{String(i + 1).padStart(2, '0')} / {s.k}</Label>
                <div className="mt-3 text-[24px] font-medium text-ink">{s.t}</div>
                <p className="mt-2 min-h-[112px] text-[17px] leading-snug text-ink-2">{s.d}</p>
                <div className="num mt-3 font-mono text-[38px] font-semibold text-ink">{m ? `${(m.variants[STEP_VARIANT[i]].top1 * 100).toFixed(2)}%` : '—'}</div>
              </div>
            ))}
          </div>
          <div className="mt-10 font-mono text-[26px] text-ink">real = s·(q − z) = s·((q+128) − (z+128)) <span className="text-ink-3">  ·  same values, new format</span></div>
        </div>
      ),
    },
    {
      kicker: 'Tool · live', title: 'The compatibility checker, run on TensorFlow’s default export',
      node: <div className="h-[486px] overflow-auto pr-2 [zoom:1.15]"><CheckerView initial={`${dm}_int8_pc`} compact /></div>,
    },
    {
      kicker: 'Results', title: `${acc} on CIFAR-10 in the format ztachip runs`,
      node: M ? (
        <div className="space-y-6">
          <ResultsTable M={M} models={models} large />
          <AccuracyByVariant M={M} models={models} height={300} />
        </div>
      ) : <div className="text-[24px] text-ink-2">Results appear after make eval.</div>,
    },
    {
      kicker: 'Results', title: 'Errors and robustness are unchanged by quantization',
      node: (
        <div className="grid grid-cols-2 gap-16">
          <div className="[zoom:1.2]">
            <div className="mono-label mb-3">Confusion matrix · {MODEL_NAMES[dm]} · UINT8</div>
            {m && <ConfusionMatrix cm={m.variants.uint8.confusion} classes={classes} />}
          </div>
          <div>
            <Label className="mb-3">Gaussian noise · FP32 vs UINT8</Label>
            {R && Object.keys(R)[0] ? <RobustnessChart R={R} model={Object.keys(R)[0]} kind="gaussian_noise" height={300} /> : <div className="text-[20px] text-ink-2">Run make robustness.</div>}
            {meta?.figures.includes(`gradcam_${dm}.png`) && <img src={figureUrl(`gradcam_${dm}.png`)} alt="Grad-CAM examples" className="mt-6 w-full" />}
          </div>
        </div>
      ),
    },
    {
      kicker: 'Live demo', title: 'FP32 next to the exact uint8 ztachip model',
      node: <div className="h-[508px] overflow-auto [zoom:1.1]"><DemoPanel compact autoRun /></div>,
    },
    {
      kicker: 'Phase II · December', title: 'Deploying on the ZedBoard',
      node: (
        <div className="grid grid-cols-[1.35fr_1fr] gap-14">
          <ZedBoard />
          <ol>
            {PORT_STEPS.map((s, i) => (
              <li key={s.t} className="grid grid-cols-[40px_1fr] border-b border-line/70 py-3 text-[17px] leading-snug text-ink-2 last:border-0">
                <span className="font-mono text-[14px] text-ink-3">{String(i + 1).padStart(2, '0')}</span>
                <span><span className="font-medium text-ink">{s.t}.</span> {s.d}</span>
              </li>
            ))}
          </ol>
        </div>
      ),
    },
    {
      kicker: 'Plan and credit', title: 'Timeline',
      node: (
        <div className="grid grid-cols-2 gap-20">
          <ol>
            {TIMELINE.map((t) => (
              <li key={t.p} className="border-b border-line/70 py-4 last:border-0">
                <div className="flex items-baseline justify-between">
                  <span className="text-[25px] font-medium text-ink">{t.p} <span className="font-mono text-[15px] font-normal text-ink-3">{t.w}</span></span>
                  {t.done ? <SafetyLabel tone="safe" className="!text-[17px]">Done</SafetyLabel> : <SafetyLabel tone="neutral" className="!text-[17px]">Planned</SafetyLabel>}
                </div>
                <div className="mt-1 text-[18px] text-ink-2">{t.d}</div>
              </li>
            ))}
          </ol>
          <div>
            <table className="w-full">
              <thead><tr className="border-b border-line"><th className="pb-2 text-left"><Label>ztachip · Vuong Nguyen</Label></th><th className="pb-2 text-left"><Label>Ours</Label></th></tr></thead>
              <tbody>{CREDITS.map(([a, b], i) => (
                <tr key={i} className="border-b border-line/70 last:border-0">
                  <td className="py-2.5 pr-6 align-top text-[16px] text-ink-3">{a}</td><td className="py-2.5 align-top text-[17px] text-ink">{b}</td>
                </tr>
              ))}</tbody>
            </table>
            <div className="mt-12 text-[40px] font-semibold tracking-[-0.015em] text-ink">Thank you. Questions?</div>
          </div>
        </div>
      ),
    },
  ]
  const total = body.length + 1

  return [
    <div key="title" className="flex h-full flex-col justify-between px-[104px] py-[80px]">
      <div className="flex items-center gap-3"><Logo className="h-8 w-8" /><span className="text-[26px] font-bold tracking-[-0.01em] text-ink">{PROJECT.short}</span></div>
      <div>
        <Label>Minor project · Phase I evaluation · {PROJECT.date}</Label>
        <h1 className="mt-6 max-w-[1250px] text-[64px] font-semibold leading-[1.08] tracking-[-0.02em] text-ink">{PROJECT.title}</h1>
        <p className="mt-7 max-w-[1050px] text-[26px] leading-snug text-ink-2">CIFAR-10 image classification co-designed for ztachip, an open-source RISC-V AI accelerator for FPGAs.</p>
      </div>
      <div className="flex items-end justify-between border-t border-line pt-6 text-[20px]">
        <div><div className="text-ink">{PROJECT.team.join('  ·  ')}</div><div className="mt-1 text-ink-2">Guide: {PROJECT.guide}</div></div>
        <div className="text-right text-ink-2">{PROJECT.institute}</div>
      </div>
    </div>,
    ...body.map((s, i) => <Slide key={s.kicker + i} n={i + 2} total={total} kicker={s.kicker} title={s.title}>{s.node}</Slide>),
  ]
}

export default function Present() {
  const navigate = useNavigate()
  const slides = useSlides()
  const [i, setI] = useState(() => Math.min(Math.max(0, parseInt(location.hash.slice(1)) - 1 || 0), 99))
  const [scale, setScale] = useState(1)

  const go = useCallback((d: number) => {
    setI((x) => Math.min(slides.length - 1, Math.max(0, x + d)))
  }, [slides.length])

  useEffect(() => { history.replaceState(null, '', `#${i + 1}`) }, [i])
  useEffect(() => {
    const onHash = () => {
      const n = parseInt(location.hash.slice(1)) - 1
      if (n >= 0 && n < slides.length) setI(n)
    }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [slides.length])
  useLayoutEffect(() => {
    const fit = () => setScale(Math.min(window.innerWidth / W, window.innerHeight / H))
    fit()
    window.addEventListener('resize', fit)
    return () => window.removeEventListener('resize', fit)
  }, [])
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return
      if (['ArrowRight', 'PageDown', ' '].includes(e.key)) { e.preventDefault(); go(1) }
      else if (['ArrowLeft', 'PageUp'].includes(e.key)) { e.preventDefault(); go(-1) }
      else if (e.key === 'f' || e.key === 'F') document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen()
      else if (e.key === 'Escape' && !document.fullscreenElement) navigate('/')
      else if (e.key === 'Home') setI(0)
      else if (e.key === 'End') setI(slides.length - 1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [go, navigate, slides.length])

  return (
    <div className="fixed inset-0 flex items-center justify-center overflow-hidden bg-[#efeee9]">
      <div style={{ width: W, height: H, transform: `scale(${scale})` }} className="relative shrink-0 overflow-hidden bg-canvas">
        <AnimatePresence mode="wait">
          <motion.div key={i} className="absolute inset-0"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.15, ease: 'easeOut' }}>
            {slides[i]}
          </motion.div>
        </AnimatePresence>
        <div className="absolute inset-x-[104px] bottom-[36px] flex items-center gap-5">
          <div className="h-[2px] flex-1 overflow-hidden rounded-full bg-line">
            <motion.div className="h-full bg-ink-3" initial={false} animate={{ width: `${((i + 1) / slides.length) * 100}%` }} transition={{ duration: 0.2 }} />
          </div>
          <span className="font-mono text-[13px] text-ink-3">{PROJECT.short}</span>
        </div>
      </div>
      <div className="fixed bottom-3 right-3 flex gap-0.5 opacity-25 transition-opacity hover:opacity-100">
        <button className="btn btn-icon" onClick={() => go(-1)} aria-label="Previous slide"><ChevronLeft size={16} /></button>
        <button className="btn btn-icon" onClick={() => go(1)} aria-label="Next slide"><ChevronRight size={16} /></button>
        <button className="btn btn-icon" onClick={() => document.documentElement.requestFullscreen()} aria-label="Full screen"><Expand size={15} /></button>
        <button className="btn btn-icon" onClick={() => navigate('/')} aria-label="Exit presentation"><X size={16} /></button>
      </div>
    </div>
  )
}
