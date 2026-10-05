// Calm kit: one grammar for every page. Lines over boxes, mono labels, one accent,
// safety colours only as a dot + word, data colours only in charts.
import { ChevronsUpDown, Code2, Loader2 } from 'lucide-react'
import { motion } from 'motion/react'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { getJSON } from '@/lib/api'
import type { SourceSnippet } from '@/lib/types'

export function MonoLabel({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <div className={`mono-label ${className}`}>{children}</div>
}

export function Hairline({ dashed = false, className = '' }: { dashed?: boolean; className?: string }) {
  return <div className={`border-t border-line ${dashed ? 'border-dashed' : ''} ${className}`} />
}

/** Eyebrow, 23px title, one-sentence subtitle; trailing = mono summary OR one action. */
export function PageHeader({ eyebrow, title, children, trailing }: {
  eyebrow: string; title: string; children?: ReactNode; trailing?: ReactNode
}) {
  return (
    <header className="pt-10">
      <div className="flex items-end justify-between gap-6 pb-5">
        <div className="min-w-0 max-w-3xl">
          <MonoLabel className="mb-2.5">{eyebrow}</MonoLabel>
          <h1 className="text-title font-semibold tracking-[-0.01em] text-ink">{title}</h1>
          {children && <div className="prose-calm mt-2 text-body">{children}</div>}
        </div>
        {trailing && <div className="shrink-0 pb-1">{trailing}</div>}
      </div>
      <Hairline />
    </header>
  )
}

export function HeaderSummary({ parts }: { parts: ReactNode[] }) {
  return (
    <div className="figure-sm whitespace-nowrap text-ink-2">
      {parts.map((p, i) => <span key={i}>{i > 0 && <span className="text-ink-3">  ·  </span>}{p}</span>)}
    </div>
  )
}

/** A page zone: mono label (e.g. "02 / Approach"), optional lead sentence, hairline, content. */
export function Section({ kicker, title, detail, trailing, children, className = '' }: {
  kicker?: string; title?: string; detail?: ReactNode; trailing?: ReactNode; children: ReactNode; className?: string
}) {
  return (
    <motion.section
      initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true, margin: '-40px' }}
      transition={{ duration: 0.2, ease: 'easeOut' }}
      className={`pt-10 ${className}`}
    >
      {(kicker || title) && (
        <div className="mb-5">
          <div className="flex items-baseline justify-between gap-4 pb-2.5">
            <div className="flex items-baseline gap-3">
              {kicker && <MonoLabel className="!text-ink-2">{kicker}</MonoLabel>}
              {detail && <span className="figure-sm text-ink-3">{detail}</span>}
            </div>
            {trailing}
          </div>
          <Hairline />
          {title && <p className="mt-4 max-w-3xl text-heading font-semibold text-ink">{title}</p>}
        </div>
      )}
      {children}
    </motion.section>
  )
}

let figureCounter = 0
/** A figure without a frame; numbered caption below. */
export function Figure({ caption, children, className = '' }: { caption: ReactNode; children: ReactNode; className?: string }) {
  const [n] = useState(() => ++figureCounter)
  return (
    <figure className={className}>
      {children}
      <figcaption className="mt-3 flex gap-3 text-secondary text-ink-2">
        <span className="mono-label shrink-0 pt-[3px]">Fig. {n}</span>
        <span className="max-w-3xl">{caption}</span>
      </figcaption>
    </figure>
  )
}

/** 2-4 inline figures separated by vertical hairlines. Replaces stat cards. */
export function FigureStrip({ items }: { items: { label: string; value: ReactNode; detail?: ReactNode }[] }) {
  return (
    <div className="flex flex-wrap">
      {items.map((it, i) => (
        <div key={it.label} className={`min-w-[150px] py-1 pr-6 ${i > 0 ? 'border-l border-line pl-6' : ''}`}>
          <MonoLabel>{it.label}</MonoLabel>
          <div className="num mt-1.5 font-mono text-[23px] font-semibold tracking-[-0.01em] text-ink">{it.value}</div>
          {it.detail && <div className="mt-0.5 text-secondary text-ink-2">{it.detail}</div>}
        </div>
      ))}
    </div>
  )
}

type Tone = 'safe' | 'review' | 'danger' | 'neutral'
const TONE: Record<Tone, string> = { safe: 'bg-safe', review: 'bg-review', danger: 'bg-danger', neutral: 'bg-ink-3' }
const TONE_TEXT: Record<Tone, string> = { safe: 'text-ink', review: 'text-ink', danger: 'text-ink', neutral: 'text-ink-2' }

/** Safety colour as a 6px dot followed by a word. Never a fill. */
export function SafetyLabel({ tone, children, className = '' }: { tone: Tone; children: ReactNode; className?: string }) {
  return (
    <span className={`inline-flex items-center gap-1.5 whitespace-nowrap text-secondary ${TONE_TEXT[tone]} ${className}`}>
      <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${TONE[tone]}`} />{children}
    </span>
  )
}

export function StatusBadge({ status }: { status: 'PASS' | 'FAIL' | 'WARN' }) {
  const m = { PASS: ['safe', 'Pass'], FAIL: ['danger', 'Fail'], WARN: ['review', 'Warning'] } as const
  return <SafetyLabel tone={m[status][0]}>{m[status][1]}</SafetyLabel>
}

/** Underline tabs: mutually exclusive views of the same data. */
export function Tabs<T extends string>({ value, onChange, options }: {
  value: T; onChange: (v: T) => void; options: { value: T; label: ReactNode; count?: ReactNode }[]
}) {
  return (
    <div className="flex gap-6 border-b border-line" role="tablist">
      {options.map((o) => {
        const on = value === o.value
        return (
          <button key={o.value} role="tab" aria-selected={on} onClick={() => onChange(o.value)}
            className={`relative -mb-px flex items-center gap-1.5 pb-2 pt-1 text-body transition-colors ${on ? 'font-medium text-ink' : 'text-ink-2 hover:text-ink'}`}>
            {o.label}
            {o.count !== undefined && <span className={`figure-sm ${on ? 'text-ink-2' : 'text-ink-3'}`}>{o.count}</span>}
            {on && <motion.span layoutId={`tabline-${options.map((x) => x.value).join()}`} className="absolute inset-x-0 bottom-0 h-[1.5px] bg-ink" transition={{ duration: 0.15 }} />}
          </button>
        )
      })}
    </div>
  )
}

/** Chips filter; they never navigate. */
export function Chips<T extends string>({ value, onChange, options }: {
  value: T; onChange: (v: T) => void; options: { value: T; label: ReactNode; count?: ReactNode }[]
}) {
  return (
    <div className="flex flex-wrap gap-0.5">
      {options.map((o) => (
        <button key={o.value} className="chip" data-on={value === o.value} onClick={() => onChange(o.value)}>
          {o.label}{o.count !== undefined && <span className="count">{o.count}</span>}
        </button>
      ))}
    </div>
  )
}

/** Borderless menu reading as "Label  Value ⌃⌄". */
export function Select<T extends string>({ value, onChange, options, label }: {
  value: T; onChange: (v: T) => void; options: { value: T; label: string }[]; label?: string
}) {
  const cur = options.find((o) => o.value === value)?.label ?? ''
  return (
    <label className="relative inline-flex h-7 cursor-pointer items-center gap-1.5 rounded-md px-2 text-body font-medium hover:bg-hover">
      {label && <span className="text-ink-3">{label}</span>}
      <span className="max-w-[340px] truncate text-ink">{cur}</span>
      <ChevronsUpDown size={11} className="text-ink-3" />
      <select value={value} onChange={(e) => onChange(e.target.value as T)} className="absolute inset-0 cursor-pointer opacity-0" aria-label={label}>
        {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </label>
  )
}

export function Notice({ icon, title, children }: { icon?: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="notice flex gap-3">
      {icon && <div className="pt-0.5 text-ink-3">{icon}</div>}
      <div className="min-w-0">
        <div className="text-body font-medium text-ink">{title}</div>
        {children && <div className="mt-0.5 text-secondary text-ink-2">{children}</div>}
      </div>
    </div>
  )
}

export function Loading({ label = 'Loading' }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 py-8 text-secondary text-ink-2">
      <Loader2 size={14} className="animate-spin text-ink-3" /> {label}…
    </div>
  )
}

export function Empty({ title = 'Nothing here yet', children }: { title?: string; children: ReactNode }) {
  return (
    <div className="flex flex-col items-center px-8 py-12 text-center">
      <div className="text-body font-medium text-ink">{title}</div>
      <div className="mt-1 max-w-md text-secondary text-ink-2">{children}</div>
    </div>
  )
}

/** Opens the exact ztachip source lines behind a claim, in a floating sheet. */
export function SourceButton({ ref_, label = 'ztachip source' }: { ref_: string; label?: string }) {
  const dlg = useRef<HTMLDialogElement>(null)
  const [snips, setSnips] = useState<SourceSnippet[] | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const open = () => {
    dlg.current?.showModal()
    if (!snips) getJSON<SourceSnippet[]>(`/api/source?ref=${encodeURIComponent(ref_)}`).then(setSnips).catch((e) => setErr(String(e)))
  }
  return (
    <>
      <button onClick={open} className="link inline-flex items-center gap-1 font-mono text-figure-sm !font-normal">
        <Code2 size={12} /> {label}
      </button>
      <dialog ref={dlg} onClick={(e) => e.target === dlg.current && dlg.current?.close()}
        className="floating m-auto w-[min(900px,94vw)] p-0 text-ink">
        <div className="flex items-center justify-between gap-4 px-5 pb-3 pt-4">
          <div className="min-w-0">
            <MonoLabel>ztachip source · read-only, from the ztachip repository</MonoLabel>
            <div className="figure mt-1 truncate text-ink">{ref_}</div>
          </div>
          <button className="btn btn-secondary" onClick={() => dlg.current?.close()}>Done <span className="kbd">esc</span></button>
        </div>
        <Hairline />
        <div className="max-h-[70vh] overflow-auto px-5 py-4">
          {err && <div className="text-secondary text-danger">{err}</div>}
          {!snips && !err && <Loading />}
          {snips?.map((s) => (
            <div key={s.file + s.start} className="mb-5 last:mb-0">
              <MonoLabel className="mb-1.5 !normal-case !tracking-normal">{s.file}</MonoLabel>
              <pre className="overflow-x-auto rounded-[10px] border border-line bg-canvas py-2 text-[12.5px] leading-[1.6]">
                {s.lines.map((line, i) => {
                  const ln = s.start + i
                  const hi = ln >= s.highlight[0] && ln <= s.highlight[1]
                  return (
                    <div key={i} className={`flex pr-4 ${hi ? 'bg-accent-soft' : ''}`}>
                      <span className={`w-14 shrink-0 select-none pr-4 text-right ${hi ? 'text-accent' : 'text-ink-3'}`}>{ln}</span>
                      <code className="whitespace-pre">{line || ' '}</code>
                    </div>
                  )
                })}
              </pre>
            </div>
          ))}
        </div>
      </dialog>
    </>
  )
}

export function useInterval(fn: () => void, ms: number | null) {
  const saved = useRef(fn)
  useEffect(() => { saved.current = fn }, [fn])
  useEffect(() => {
    if (ms === null) return
    const t = setInterval(() => saved.current(), ms)
    return () => clearInterval(t)
  }, [ms])
}
