import CheckerView from '@/components/CheckerView'
import { PageHeader, Section } from '@/components/ui'

export default function Checker() {
  return (
    <>
      <PageHeader eyebrow="Method · Compatibility checker" title="Will this model run on ztachip?">
        <p>
          ztachip reports nothing for an unsupported model, so the checker tests a <code>.tflite</code> file against every
          constraint we found in its importer; View code opens the lines that impose each one.
        </p>
      </PageHeader>
      <Section>
        <CheckerView />
      </Section>
      <Section kicker="Try">
        <ul className="text-secondary text-ink-2">
          <li className="border-b border-line/70 py-2"><span className="text-ink">ztachip’s MobileNet v2</span> passes. It is the positive control.</li>
          <li className="border-b border-line/70 py-2"><span className="text-ink">ZTA-Plain · INT8 per-channel</span>, TensorFlow’s default export, fails on type and scale rules.</li>
          <li className="py-2"><span className="text-ink">ZTA-Plain · UINT8</span>, the same model through our pipeline, passes.</li>
        </ul>
      </Section>
    </>
  )
}
