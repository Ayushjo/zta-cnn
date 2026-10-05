import { ArrowRight } from 'lucide-react'

const STEPS = [
  { t: 'Train', s: 'Hardware-aware CNN', d: 'Only operators ztachip executes', done: true },
  { t: 'Quantize', s: 'Per-tensor int8', d: 'TF2 post-training quantization', done: true },
  { t: 'Rewrite', s: 'int8 → uint8', d: 'The TF1-style format ztachip reads', done: true },
  { t: 'Check', s: 'Compatibility linter', d: 'Rules traced to ztachip C++', done: true },
  { t: 'Deploy', s: 'ztachip on FPGA', d: 'Phase II · December', done: false },
]

/** Five numbered columns separated by hairlines; the Phase II step is quiet. */
export default function Pipeline({ large = false }: { large?: boolean }) {
  return (
    <ol className="grid grid-cols-2 border-y border-line sm:grid-cols-5" aria-label="Pipeline">
      {STEPS.map((st, i) => (
        <li key={st.t} className={`relative py-4 pr-4 ${i > 0 ? 'sm:border-l sm:border-line sm:pl-4' : ''} ${st.done ? '' : 'opacity-55'}`}>
          <div className={`mono-label ${large ? '!text-[15px]' : ''}`}>{String(i + 1).padStart(2, '0')} / {st.t}</div>
          <div className={`mt-2 font-medium text-ink ${large ? 'text-[26px]' : 'text-body'}`}>{st.s}</div>
          <div className={`mt-0.5 text-ink-2 ${large ? 'text-[18px]' : 'text-secondary'}`}>{st.d}</div>
          {i < STEPS.length - 1 && (
            <ArrowRight size={large ? 18 : 12} className="absolute -right-[7px] top-1/2 hidden -translate-y-1/2 bg-canvas text-ink-3 sm:block" style={large ? { right: -10 } : undefined} />
          )}
        </li>
      ))}
    </ol>
  )
}
