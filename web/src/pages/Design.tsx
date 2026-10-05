import { useState } from 'react'
import Treemap from '@/components/Treemap'
import { Figure, FigureStrip, Loading, PageHeader, SafetyLabel, Section, Select, SourceButton } from '@/components/ui'
import { useApi } from '@/lib/api'
import { DATA, MODEL_NAMES, ZTA_MODELS, num } from '@/lib/format'
import type { Profile } from '@/lib/types'

export const DESIGN_RULES: { limit: string; decision: string; ref: string }[] = [
  { limit: 'No MAX_POOL_2D', decision: 'Downsample with 3×3 stride-2 convolutions', ref: 'SW/apps/nn/tf_util.cpp:74-100' },
  { limit: 'No FULLY_CONNECTED', decision: 'Classifier is a 1×1 convolution on the pooled 1×1 map, which ztachip runs as an inner product', ref: 'SW/apps/nn/nn_conv2d.cpp:51-56' },
  { limit: 'No SOFTMAX', decision: 'The model outputs logits; softmax runs on the host and ztachip ranks raw uint8 values', ref: 'SW/apps/nn/nn_util.cpp:347-387' },
  { limit: 'Average pool must be global; no MEAN', decision: 'One AveragePooling2D(8×8) at the end, not GlobalAveragePooling2D, which lowers to MEAN', ref: 'SW/apps/nn/kernels/fcn.m:111-152' },
  { limit: '1×1 or 3×3 kernels; 1×1 ignores stride', decision: '3×3 convolutions; the ResNet shortcut is 3×3 stride 2 instead of 1×1 stride 2', ref: 'SW/apps/nn/kernels/conv.m:738-745' },
  { limit: 'uint8, per-tensor quantization only', decision: 'Per-tensor post-training quantization, then our int8 → uint8 rewriter', ref: 'SW/apps/nn/tf_util.cpp:196-231' },
  { limit: 'No MUL (Keras Rescaling)', decision: 'Input scale 1/255 is baked into quantization, so raw camera bytes feed in directly', ref: 'SW/apps/nn/nn.cpp:276-289' },
  { limit: 'Unsupported operators hang the board', decision: 'Batch fixed to 1 at export; a dynamic Reshape emits SHAPE, STRIDED_SLICE and PACK', ref: 'SW/apps/nn/tf.cpp:400-401' },
  { limit: 'Weights padded to 8-wide vectors', decision: 'Every convolution has a multiple of 8 channels', ref: 'SW/base/zta.h:60-62' },
]

const ZOO = [
  { id: 'zta_plain', s: 'Six 3×3 convolution blocks, 32-32-64-64-128-128 channels, two stride-2 steps, global average pool, 1×1 classifier.', ops: 'CONV_2D · AVERAGE_POOL_2D · RESHAPE', ok: true, note: 'Deployed model' },
  { id: 'zta_resnet8', s: 'MLPerf Tiny ResNet-8 (16/32/64) with residual ADD and 3×3 stride-2 shortcuts.', ops: 'adds ADD', ok: true, note: 'MLPerf Tiny reference' },
  { id: 'zta_mobile', s: 'MobileNet-style depthwise-separable blocks with ReLU6.', ops: 'adds DEPTHWISE_CONV_2D', ok: true, note: 'Depthwise test' },
  { id: 'chatgpt_baseline', s: 'Conv and MaxPool twice, Flatten, Dense, Softmax: the textbook CNN.', ops: 'MAX_POOL_2D · FULLY_CONNECTED · SOFTMAX', ok: false, note: 'Reference only' },
]

const TYPE_COLOR: Record<string, string> = {
  Conv2D: DATA.slate, DepthwiseConv2D: DATA.violet, Dense: DATA.rose, Add: DATA.sage, AveragePooling2D: DATA.sand, MaxPooling2D: DATA.sand,
}

export default function Design() {
  const [model, setModel] = useState('zta_plain')
  const { data: prof, loading } = useApi<Profile>(`/api/model/${model}/layers`)
  const layers = prof?.layers ?? []
  const items = layers.filter((l) => l.macs > 0).map((l, i) => ({
    id: `${l.name}-${i}`, label: `${l.type.replace('2D', '')} ${l.output.join('×')}`, value: l.macs,
    color: TYPE_COLOR[l.type] ?? DATA.stone, detail: `${num(l.params)} params`,
  }))
  return (
    <>
      <PageHeader eyebrow="Method · Design" title="Every layer choice comes from a line of ztachip code">
        <p>Before training anything we read ztachip’s TFLite importer and turned each limitation into a design rule that the checker later enforces.</p>
      </PageHeader>

      <Section kicker="01 / Limits and decisions">
        <table className="list">
          <thead><tr><th className="w-[30%]">ztachip limitation</th><th>Our decision</th><th className="r">Evidence</th></tr></thead>
          <tbody>
            {DESIGN_RULES.map((r) => (
              <tr key={r.limit}>
                <td className="font-medium text-ink">{r.limit}</td>
                <td className="text-ink-2">{r.decision}</td>
                <td className="r whitespace-nowrap"><SourceButton ref_={r.ref} label={r.ref.split('/').pop()!} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </Section>

      <Section kicker="02 / Model zoo">
        <table className="list">
          <thead><tr><th>Model</th><th>Structure</th><th>Operators</th><th className="r">ztachip</th></tr></thead>
          <tbody>
            {ZOO.map((z) => (
              <tr key={z.id}>
                <td className="whitespace-nowrap"><div className="font-medium text-ink">{MODEL_NAMES[z.id]}</div><div className="text-secondary text-ink-3">{z.note}</div></td>
                <td className="text-secondary text-ink-2">{z.s}</td>
                <td className="figure-sm text-ink-2">{z.ops}</td>
                <td className="r">{z.ok ? <SafetyLabel tone="safe">Runs</SafetyLabel> : <SafetyLabel tone="danger">Would hang</SafetyLabel>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Section>

      <Section kicker="03 / Where the compute goes" trailing={
        <Select label="Model" value={model} onChange={setModel} options={[...ZTA_MODELS, 'chatgpt_baseline'].map((m) => ({ value: m, label: MODEL_NAMES[m] }))} />}>
        {loading || !prof ? <Loading /> : (
          <>
            <FigureStrip items={[
              { label: 'Parameters', value: num(prof.params) },
              { label: 'MACs per image', value: `${(prof.macs / 1e6).toFixed(2)} M` },
              { label: 'Peak activation', value: `${(prof.peak_activation_bytes_uint8 / 1024).toFixed(0)} KiB`, detail: 'one uint8 feature map' },
            ]} />
            <div className="mt-6">
              <Figure caption="Each tile is one layer; its area is the layer’s multiply-accumulates. Slate is convolution, violet depthwise, sage add, rose dense. BatchNorm and ReLU are folded into the convolutions by the converter.">
                <Treemap items={items} format={(v) => `${(v / 1e6).toFixed(2)} M MACs`} />
              </Figure>
            </div>
          </>
        )}
      </Section>
    </>
  )
}
