import { ArrowRight, Cpu, Layers, ShieldCheck, Sparkles, Wrench } from 'lucide-react'
import { Link } from 'react-router'
import Pipeline from '@/components/Pipeline'
import { FigureStrip, MonoLabel, PageHeader, SafetyLabel, Section } from '@/components/ui'
import { useApi, useMetrics } from '@/lib/api'
import { MODEL_NAMES } from '@/lib/format'
import type { Meta } from '@/lib/types'

export const CONTRIBUTIONS = [
  { icon: Layers, t: 'Hardware-aware CNNs', d: 'Three CIFAR-10 models built only from operators ztachip executes: 3×3 stride-2 convolutions instead of MaxPool, a 1×1 convolution instead of Dense, softmax on the host.' },
  { icon: Wrench, t: 'int8 → uint8 rewriter', d: 'TensorFlow 2 no longer produces the uint8 models ztachip runs. The rewriter maps q → q+128 and zp → zp+128, so every real value stays identical.' },
  { icon: ShieldCheck, t: 'Compatibility checker', d: 'One rule per hardware constraint, each traced to the line of ztachip C++ that imposes it. It accepts ztachip’s own MobileNet and rejects the textbook CNN.' },
  { icon: Sparkles, t: 'Evaluation', d: 'Accuracy at every quantization stage, confusion matrices, calibration, Grad-CAM, robustness to camera noise, CPU latency and a cost model.' },
  { icon: Cpu, t: 'Phase II package', d: 'The deployable model, labels, golden test vectors in ztachip’s own test format, and an on-board test for the ZedBoard.' },
]

export const CREDITS: [string, string][] = [
  ['RTL accelerator, RISC-V SoC, kernel compiler, runtime', 'Hardware-aware model design for ztachip’s operator set'],
  ['TFLite execution engine (SW/apps/nn)', 'Training, evaluation, calibration, robustness, Grad-CAM'],
  ['MobileNet, SSD and LLM demos', 'int8 → uint8 TFLite rewriter'],
  ['Arty A7 reference design', 'ztachip compatibility checker'],
  ['', 'Golden vectors and on-board test, ZedBoard port (Phase II), this site'],
]

export const UNSUPPORTED = ['MAX_POOL_2D', 'FULLY_CONNECTED', 'SOFTMAX', 'MEAN', 'MUL and PAD', 'int8 tensors', 'Per-channel scales', '5×5 and 7×7 kernels']

export default function Overview() {
  const { data: M } = useMetrics()
  const { data: meta } = useApi<Meta>('/api/meta')
  const dm = meta?.deploy_model ?? 'zta_plain'
  const m = M?.[dm]
  return (
    <>
      <PageHeader eyebrow="Minor project · Phase I · CIFAR-10" title="A CNN co-designed to run on an open-source FPGA accelerator">
        <p>
          We train image classifiers that fit the exact operators and number format of <strong>ztachip</strong>, an
          open-source RISC-V AI accelerator, and build the tools that prove a model will run on it before it reaches the board.
        </p>
      </PageHeader>

      <Section>
        <MonoLabel>{MODEL_NAMES[dm]} · CIFAR-10 test set · 10,000 images</MonoLabel>
        <div className="mt-2 flex flex-wrap items-baseline gap-x-3">
          <span className="num font-mono text-[44px] font-semibold leading-none tracking-[-0.02em] text-ink">
            {m ? `${(m.variants.uint8.top1 * 100).toFixed(1)}%` : '—'}
          </span>
          <span className="text-body text-ink-2">accuracy in the uint8 format ztachip executes</span>
        </div>
        <div className="mt-7">
          <FigureStrip items={[
            { label: 'Quantization cost', value: m ? `${((m.variants.keras.top1 - m.variants.uint8.top1) * 100).toFixed(2)} pp` : '—', detail: 'FP32 to UINT8 drop' },
            { label: 'Deployed model', value: m ? `${m.variants.uint8.size_kib!.toFixed(0)} KiB` : '—', detail: m ? `from ${m.variants.fp32.size_kib!.toFixed(0)} KiB in FP32` : undefined },
            { label: 'Hardware rules', value: meta?.rules ?? 21, detail: 'each traced to ztachip source' },
          ]} />
        </div>
      </Section>

      <Section kicker="01 / The problem" title="A normal CNN would hang the board.">
        <div className="grid gap-10 lg:grid-cols-[1.3fr_1fr]">
          <div className="prose-calm text-body">
            <p>
              ztachip runs TensorFlow Lite models, but its importer (<code>SW/apps/nn/</code>) supports a narrow subset: seven
              operators, 1×1 and 3×3 kernels, and only the older <strong>uint8, per-tensor</strong> number format.
            </p>
            <p>
              An unsupported model raises no error. An unknown layer reaches <code>assert(0)</code>, and on this bare-metal system{' '}
              <code>_exit()</code> is an infinite loop, so the board freezes. Other features, such as int8 tensors or per-channel
              scales, run but compute wrong numbers.
            </p>
            <p>The textbook CNN and TensorFlow’s default int8 export both fall into these traps.</p>
          </div>
          <div>
            <MonoLabel className="mb-2">What ztachip cannot run</MonoLabel>
            <ul className="grid grid-cols-2 gap-x-6">
              {UNSUPPORTED.map((x) => (
                <li key={x} className="border-b border-line/70 py-2"><SafetyLabel tone="danger"><span className="font-mono text-figure-sm">{x}</span></SafetyLabel></li>
              ))}
            </ul>
            <Link to="/design" className="link mt-3 inline-flex items-center gap-1 text-body">How each limit shaped the model <ArrowRight size={13} /></Link>
          </div>
        </div>
      </Section>

      <Section kicker="02 / Approach" title="From training to a board-ready file.">
        <Pipeline />
        <p className="mt-3 text-secondary text-ink-2">Steps 01–04 are built and tested. Step 05 runs on the FPGA in Phase II.</p>
      </Section>

      <Section kicker="03 / Contributions" title="What we built.">
        <ul>
          {CONTRIBUTIONS.map(({ icon: Icon, t, d }) => (
            <li key={t} className="grid grid-cols-[28px_200px_1fr] items-baseline gap-3 border-b border-line/70 py-3.5 max-md:grid-cols-[28px_1fr]">
              <Icon size={15} strokeWidth={1.75} className="translate-y-0.5 text-ink-3" />
              <span className="text-body font-medium text-ink">{t}</span>
              <span className="text-secondary text-ink-2 max-md:col-start-2">{d}</span>
            </li>
          ))}
        </ul>
        <Link to="/demo" className="link mt-4 inline-flex items-center gap-1 text-body">Try the live demo <ArrowRight size={13} /></Link>
      </Section>

      <Section kicker="04 / Credit" title="What is ours and what is ztachip.">
        <table className="list">
          <thead><tr><th>ztachip · Vuong Nguyen · Apache-2.0</th><th>This project</th></tr></thead>
          <tbody>{CREDITS.map(([a, b], i) => <tr key={i}><td className="text-secondary text-ink-2">{a}</td><td className="text-body">{b}</td></tr>)}</tbody>
        </table>
        <p className="mt-4 text-secondary text-ink-2">Nothing has run on an FPGA yet. Hardware results come in Phase II (December).</p>
      </Section>
    </>
  )
}
