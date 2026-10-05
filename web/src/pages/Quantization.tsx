import { useState } from 'react'
import { AccuracyByVariant } from '@/components/charts'
import { Figure, Loading, MonoLabel, PageHeader, SafetyLabel, Section, Select } from '@/components/ui'
import { useApi, useMetrics } from '@/lib/api'
import { ALL_MODELS, MODEL_NAMES, ZTA_MODELS } from '@/lib/format'
import type { TensorRow } from '@/lib/types'

export const STEPS = [
  { k: 'FP32', t: 'Float model', d: '32-bit weights, as Keras trains them.', ok: null },
  { k: 'INT8 per-channel', t: 'TensorFlow default', d: 'int8 tensors with one scale per channel. ztachip reads only scale[0] and handles only uint8, so results are silently wrong.', ok: false },
  { k: 'INT8 per-tensor', t: 'Our converter settings', d: 'One scale per tensor and int8 inputs and outputs, so no QUANTIZE operators are added.', ok: null },
  { k: 'UINT8', t: 'Our rewriter', d: 'Every int8 value q becomes q+128 and every zero point z becomes z+128; scales are unchanged. ztachip executes this.', ok: true },
] as const
export const STEP_VARIANT = ['keras', 'int8_pc', 'int8_pt', 'uint8'] as const

export default function Quantization() {
  const { data: M } = useMetrics()
  const [model, setModel] = useState('zta_plain')
  const { data: T, loading } = useApi<{ tensors: TensorRow[]; ops: string[] }>(`/api/model/${model}/tensors`)
  return (
    <>
      <PageHeader eyebrow="Method · Quantization" title="Getting TensorFlow 2 to produce a model ztachip can read">
        <p>ztachip executes only TF1-style asymmetric uint8 with one scale per tensor, a format TensorFlow 2 can no longer emit, so we convert to per-tensor int8 and rewrite the file ourselves.</p>
      </PageHeader>

      <Section kicker="01 / Four stages" trailing={<Select label="Accuracy for" value={model} onChange={setModel} options={ZTA_MODELS.map((m) => ({ value: m, label: MODEL_NAMES[m] }))} />}>
        <ol className="grid border-b border-line sm:grid-cols-4">
          {STEPS.map((s, i) => (
            <li key={s.k} className={`py-4 pr-5 ${i > 0 ? 'sm:border-l sm:border-line sm:pl-5' : ''}`}>
              <MonoLabel>{String(i + 1).padStart(2, '0')} / {s.k}</MonoLabel>
              <div className="mt-2 text-body font-medium text-ink">{s.t}</div>
              <p className="mt-1 min-h-[84px] text-secondary text-ink-2">{s.d}</p>
              <div className="mt-3 flex items-baseline justify-between gap-2">
                <span className="num font-mono text-[23px] font-semibold text-ink">
                  {M?.[model] ? `${(M[model].variants[STEP_VARIANT[i]].top1 * 100).toFixed(2)}%` : '—'}
                </span>
                {s.ok === false && <SafetyLabel tone="danger">Fails on ztachip</SafetyLabel>}
                {s.ok === true && <SafetyLabel tone="safe">Runs</SafetyLabel>}
              </div>
            </li>
          ))}
        </ol>
      </Section>

      <Section kicker="02 / Why the rewrite is exact">
        <div className="grid gap-10 lg:grid-cols-2">
          <pre className="figure leading-[2] text-ink">
            <span className="text-ink-3">{'// affine quantization\n'}</span>
            {'real = s · (q − z)\n\n'}
            <span className="text-ink-3">{'// int8 → uint8\n'}</span>
            {'q′ = q + 128     z′ = z + 128\n'}
            {'s · (q′ − z′) = s · (q − z) = real\n\n'}
            <span className="text-ink-3">{'// requantization multiplier unchanged\n'}</span>
            {'M = s_in · s_w / s_out\n'}
            <span className="text-ink-3">{'// int32 bias: scale s_in·s_w, zero point 0, untouched'}</span>
          </pre>
          <div className="prose-calm text-body">
            <p>Every real number the network computes with is identical before and after the rewrite. Only the storage format changes.</p>
            <p>
              Executed, the two files can differ by 1–2 least-significant bits in the output, because TensorFlow Lite’s int8 and
              legacy uint8 kernels round the requantization step differently. We measured at most 2 LSB and the same top-1 class on
              99.7–100% of inputs, so every UINT8 accuracy on this site is measured on the uint8 file itself.
            </p>
          </div>
        </div>
      </Section>

      {M && (
        <Section kicker="03 / Accuracy by stage">
          <Figure caption="Test accuracy at each stage for every model.">
            <AccuracyByVariant M={M} models={ALL_MODELS.filter((m) => m in M)} />
          </Figure>
        </Section>
      )}

      <Section kicker="04 / Tensor inspector" detail={T ? `${T.tensors.length} tensors` : undefined}>
        <p className="mb-4 text-secondary text-ink-2">Every quantized int8 tensor of {MODEL_NAMES[model]} before and after the rewrite: scales match exactly and zero points shift by +128.</p>
        {loading ? <Loading /> : T && (
          <div className="max-h-[440px] overflow-auto">
            <table className="list num">
              <thead className="sticky top-0 bg-canvas">
                <tr><th>Tensor</th><th>Shape</th><th className="r">int8 scale</th><th className="r">int8 zp</th><th className="r">uint8 scale</th><th className="r">uint8 zp</th></tr>
              </thead>
              <tbody>
                {T.tensors.map((t) => (
                  <tr key={t.name}>
                    <td className="max-w-[300px] truncate font-mono text-figure-sm text-ink" title={t.name}>{t.name.replace(/^.*?\//, '')}</td>
                    <td className="figure-sm text-ink-2">[{t.shape.join(',')}]</td>
                    <td className="r figure-sm text-ink-2">{t.int8.scale.toExponential(4)}</td>
                    <td className="r figure-sm text-ink-2">{t.int8.zero_point}</td>
                    <td className="r figure-sm text-ink">{t.uint8.scale.toExponential(4)}</td>
                    <td className="r figure-sm text-ink">{t.uint8.zero_point}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>
    </>
  )
}
