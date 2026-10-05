import { useEffect, useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, LabelList, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer,
  Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from 'recharts'
import { getJSON } from '@/lib/api'
import { COLORS, DATA, MODEL_COLOR, MODEL_NAMES, VARIANT_COLOR, VARIANT_LABEL } from '@/lib/format'
import type { History, Metrics, Robustness } from '@/lib/types'

// Calm chart grammar: hairline grid, mono ink-3 ticks, data palette only, floating tooltip.
const MONO = 'ui-monospace, "SF Mono", "JetBrains Mono Variable", monospace'
const tick = { fill: COLORS.ink3, fontSize: 11, fontFamily: MONO }
const axis = { stroke: COLORS.line, tick, tickLine: false, axisLine: { stroke: COLORS.line } }
const pctTick = (v: number) => `${Math.round(v)}%`
const grid = <CartesianGrid stroke={COLORS.line} vertical={false} />
const tip = {
  contentStyle: { borderRadius: 10, border: `1px solid ${COLORS.line}`, fontSize: 12.5, boxShadow: '0 8px 24px rgba(0,0,0,.10)', padding: '8px 10px' },
  labelStyle: { color: COLORS.ink, fontWeight: 500, marginBottom: 4 },
  itemStyle: { fontFamily: MONO, fontSize: 12, padding: 0 },
  cursor: { fill: 'rgba(37,43,49,0.045)' },
}
const legend = { wrapperStyle: { fontSize: 12.5, paddingTop: 10, color: COLORS.ink2 }, iconType: 'circle' as const, iconSize: 7 }
const barRadius: [number, number, number, number] = [2, 2, 0, 0]

export function AccuracyByVariant({ M, models, variants = ['keras', 'int8_pc', 'int8_pt', 'uint8'], height = 320 }: {
  M: Metrics; models: string[]; variants?: string[]; height?: number
}) {
  const data = models.map((m) => ({
    name: MODEL_NAMES[m] + (M[m].ztachip_compatible ? '' : ' (not compatible)'),
    ...Object.fromEntries(variants.map((v) => [v, +(M[m].variants[v as 'uint8'].top1 * 100).toFixed(2)])),
  }))
  const vals = models.flatMap((m) => variants.map((v) => M[m].variants[v as 'uint8'].top1 * 100))
  const lo = Math.max(0, Math.floor(Math.min(...vals) / 10) * 10 - 10)
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} barCategoryGap="24%" barGap={2}>
        {grid}
        <XAxis dataKey="name" {...axis} tick={{ ...tick, fontFamily: 'inherit', fontSize: 12.5, fill: COLORS.ink2 }} />
        <YAxis {...axis} axisLine={false} domain={[lo, 100]} tickFormatter={pctTick} width={44} />
        <Tooltip {...tip} formatter={(v) => `${v}%`} />
        <Legend {...legend} />
        {variants.map((v) => (
          <Bar key={v} dataKey={v} name={VARIANT_LABEL[v]} fill={VARIANT_COLOR[v]} radius={barRadius}>
            <LabelList dataKey={v} position="top" fontSize={10.5} fontFamily={MONO} fill={COLORS.ink3} formatter={(x) => Number(x).toFixed(1)} />
          </Bar>
        ))}
      </BarChart>
    </ResponsiveContainer>
  )
}

export function TrainingCurves({ models, metric = 'accuracy' }: { models: string[]; metric?: 'accuracy' | 'loss' }) {
  const [hist, setHist] = useState<Record<string, History>>({})
  useEffect(() => {
    models.forEach((m) => getJSON<History>(`/api/history/${m}`).then((h) => setHist((s) => ({ ...s, [m]: h }))).catch(() => {}))
  }, [models.join()]) // eslint-disable-line react-hooks/exhaustive-deps
  const n = Math.max(0, ...Object.values(hist).map((h) => h.accuracy.length))
  const k = metric === 'accuracy' ? 100 : 1
  const data = Array.from({ length: n }, (_, i) => ({
    epoch: i + 1,
    ...Object.fromEntries(Object.entries(hist).flatMap(([m, h]) => [
      [`${m}_val`, h[`val_${metric}`][i] !== undefined ? +(h[`val_${metric}`][i] * k).toFixed(3) : null],
      [`${m}_train`, h[metric][i] !== undefined ? +(h[metric][i] * k).toFixed(3) : null],
    ])),
  }))
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data}>
        {grid}
        <XAxis dataKey="epoch" {...axis} />
        <YAxis {...axis} axisLine={false} tickFormatter={metric === 'accuracy' ? pctTick : (v: number) => v.toFixed(2)} width={44}
          domain={metric === 'accuracy' ? [20, 100] : ['auto', 'auto']} />
        <Tooltip {...tip} cursor={{ stroke: COLORS.line }} />
        <Legend {...legend} />
        {Object.keys(hist).flatMap((m) => [
          <Line key={`${m}_val`} dataKey={`${m}_val`} name={`${MODEL_NAMES[m]} · validation`} stroke={MODEL_COLOR[m]} strokeWidth={2} dot={false} />,
          <Line key={`${m}_train`} dataKey={`${m}_train`} name={`${MODEL_NAMES[m]} · train`} stroke={MODEL_COLOR[m]} strokeWidth={1.2} strokeDasharray="3 3" dot={false} legendType="none" />,
        ])}
      </LineChart>
    </ResponsiveContainer>
  )
}

// Opaque wash of a data colour over the canvas, like Calm's wash(color, strength:).
function wash(hex: string, f: number) {
  const c = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16))
  const p = [0xf8, 0xf7, 0xf4]
  return `rgb(${c.map((v, i) => Math.round(p[i] + (v - p[i]) * f)).join(',')})`
}

export function ConfusionMatrix({ cm, classes }: { cm: number[][]; classes: string[] }) {
  const [hover, setHover] = useState<[number, number] | null>(null)
  return (
    <div className="overflow-x-auto">
      <div className="inline-grid min-w-[540px] gap-[3px]" style={{ gridTemplateColumns: `84px repeat(${classes.length}, minmax(38px, 1fr))` }}>
        <div className="mono-label self-end pb-1 !text-[10px]">true \ pred</div>
        {classes.map((c, j) => (
          <div key={c} className={`truncate pb-1 text-center font-mono text-[10.5px] ${hover?.[1] === j ? 'text-ink' : 'text-ink-3'}`}>{c.slice(0, 5)}</div>
        ))}
        {cm.map((row, i) => {
          const tot = row.reduce((a, b) => a + b, 0)
          return [
            <div key={`l${i}`} className={`flex items-center pr-2 text-secondary ${hover?.[0] === i ? 'text-ink' : 'text-ink-2'}`}>{classes[i]}</div>,
            ...row.map((v, j) => {
              const f = v / tot
              const diag = i === j
              const bg = v === 0 ? 'rgba(37,43,49,0.03)' : diag ? wash(DATA.slate, 0.15 + 0.85 * f) : wash(DATA.rose, Math.min(1, 0.12 + f * 3.5))
              const on = hover && hover[0] === i && hover[1] === j
              return (
                <div key={`${i}-${j}`} onMouseEnter={() => setHover([i, j])} onMouseLeave={() => setHover(null)}
                  className="num flex aspect-square items-center justify-center rounded-[4px] font-mono text-[11px]"
                  style={{ background: bg, color: `rgba(37,43,49,${v ? (diag ? 0.85 : 0.7) : 0.3})`, boxShadow: on ? `inset 0 0 0 2px ${COLORS.accent}` : undefined }}
                  title={`${classes[i]} → ${classes[j]}: ${v} (${(f * 100).toFixed(1)}%)`}>
                  {v}
                </div>
              )
            }),
          ]
        })}
      </div>
      <div className="figure-sm mt-2 h-5 text-ink-2">
        {hover ? `${cm[hover[0]][hover[1]]} ${classes[hover[0]]} images predicted as ${classes[hover[1]]}` : 'Hover a cell'}
      </div>
    </div>
  )
}

export function PerClassF1({ M, model, classes }: { M: Metrics; model: string; classes: string[] }) {
  const v = M[model].variants
  const data = classes.map((c, i) => ({ c, fp32: +(v.keras.f1[i] * 100).toFixed(1), uint8: +(v.uint8.f1[i] * 100).toFixed(1) }))
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} barGap={1}>
        {grid}
        <XAxis dataKey="c" {...axis} interval={0} tick={{ ...tick, fontSize: 10 }} />
        <YAxis {...axis} axisLine={false} tickFormatter={pctTick} width={40} domain={[40, 100]} />
        <Tooltip {...tip} formatter={(x) => `${x}%`} />
        <Legend {...legend} />
        <Bar dataKey="fp32" name="FP32" fill={VARIANT_COLOR.keras} radius={barRadius} />
        <Bar dataKey="uint8" name="UINT8 · ztachip" fill={VARIANT_COLOR.uint8} radius={barRadius} />
      </BarChart>
    </ResponsiveContainer>
  )
}

export function Reliability({ M, model }: { M: Metrics; model: string }) {
  const series = (['keras', 'uint8'] as const).map((v) => ({
    v, data: M[model].variants[v].reliability.map(([c, a, n]) => ({ c: +(c * 100).toFixed(1), a: +(a * 100).toFixed(1), n })),
  }))
  return (
    <ResponsiveContainer width="100%" height={280}>
      <ScatterChart>
        {grid}
        <XAxis type="number" dataKey="c" name="confidence" domain={[0, 100]} tickFormatter={pctTick} {...axis} />
        <YAxis type="number" dataKey="a" name="accuracy" domain={[0, 100]} tickFormatter={pctTick} width={40} {...axis} axisLine={false} />
        <ZAxis type="number" dataKey="n" range={[16, 140]} name="images" />
        <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 100, y: 100 }]} stroke={COLORS.ink3} strokeDasharray="3 3" />
        <Tooltip {...tip} />
        <Legend {...legend} />
        {series.map((s) => (
          <Scatter key={s.v} data={s.data} name={`${VARIANT_LABEL[s.v]} · ECE ${M[model].variants[s.v].ece.toFixed(3)}`}
            fill={VARIANT_COLOR[s.v]} line={{ strokeWidth: 1.5 }} />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  )
}

export function LatencyChart({ M, models }: { M: Metrics; models: string[] }) {
  const keys = [['keras_fp32', 'Keras FP32', DATA.stone], ['tflite_fp32_1thread', 'TFLite FP32', DATA.sky], ['tflite_int8_pt_1thread', 'TFLite INT8', DATA.sand], ['tflite_uint8_1thread', 'TFLite UINT8', DATA.slate]] as const
  const data = models.map((m) => ({ name: MODEL_NAMES[m], ...Object.fromEntries(keys.map(([k]) => [k, +M[m].latency_ms[k][0].toFixed(3)])) }))
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} barGap={2}>
        {grid}
        <XAxis dataKey="name" {...axis} tick={{ ...tick, fontFamily: 'inherit', fontSize: 12.5, fill: COLORS.ink2 }} />
        <YAxis {...axis} axisLine={false} tickFormatter={(v: number) => `${v} ms`} width={56} />
        <Tooltip {...tip} formatter={(v) => `${v} ms`} />
        <Legend {...legend} />
        {keys.map(([k, l, c]) => <Bar key={k} dataKey={k} name={l} fill={c} radius={barRadius} />)}
      </BarChart>
    </ResponsiveContainer>
  )
}

export function Pareto({ M, models }: { M: Metrics; models: string[] }) {
  const data = models.map((m) => ({
    name: MODEL_NAMES[m], macs: +(M[m].profile.macs / 1e6).toFixed(2), acc: +(M[m].variants.uint8.top1 * 100).toFixed(2),
    params: M[m].profile.params, ok: M[m].ztachip_compatible,
  }))
  return (
    <ResponsiveContainer width="100%" height={280}>
      <ScatterChart margin={{ right: 40 }}>
        {grid}
        <XAxis type="number" dataKey="macs" name="MACs" tickFormatter={(v: number) => `${v} M`} {...axis} />
        <YAxis type="number" dataKey="acc" name="UINT8 accuracy" domain={['dataMin - 4', 'dataMax + 3']} tickFormatter={pctTick} width={40} {...axis} axisLine={false} />
        <ZAxis type="number" dataKey="params" range={[110, 650]} name="params" />
        <Tooltip {...tip} />
        <Scatter data={data.filter((d) => d.ok)} fill={DATA.slate} name="runs on ztachip">
          <LabelList dataKey="name" position="right" fontSize={12} fill={COLORS.ink2} />
        </Scatter>
        <Scatter data={data.filter((d) => !d.ok)} fill={DATA.stone} shape="cross" name="cannot run on ztachip">
          <LabelList dataKey="name" position="right" fontSize={12} fill={COLORS.ink3} />
        </Scatter>
        <Legend {...legend} />
      </ScatterChart>
    </ResponsiveContainer>
  )
}

export function RobustnessChart({ R, model, kind, height = 260 }: { R: Robustness; model: string; kind: string; height?: number }) {
  const r = R[model]
  const clean = r.clean as unknown as { fp32: number; uint8: number }
  const k = r[kind] as unknown as { fp32: number[]; uint8: number[] }
  const data = [{ s: 0, fp32: clean.fp32, uint8: clean.uint8 }, ...k.fp32.map((v, i) => ({ s: i + 1, fp32: v, uint8: k.uint8[i] }))]
    .map((d) => ({ s: d.s, fp32: +(d.fp32 * 100).toFixed(1), uint8: +(d.uint8 * 100).toFixed(1) }))
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data}>
        {grid}
        <XAxis dataKey="s" {...axis} tickFormatter={(v: number) => (v === 0 ? 'clean' : `s${v}`)} />
        <YAxis {...axis} axisLine={false} tickFormatter={pctTick} width={40} domain={[0, 100]} />
        <Tooltip {...tip} cursor={{ stroke: COLORS.line }} />
        <Legend {...legend} />
        <Line dataKey="fp32" name="FP32" stroke={VARIANT_COLOR.keras} strokeWidth={2} dot={{ r: 2.5 }} />
        <Line dataKey="uint8" name="UINT8 · ztachip" stroke={VARIANT_COLOR.uint8} strokeWidth={2} strokeDasharray="4 3" dot={{ r: 2.5 }} />
      </LineChart>
    </ResponsiveContainer>
  )
}
