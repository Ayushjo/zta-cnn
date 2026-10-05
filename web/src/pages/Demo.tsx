import DemoPanel from '@/components/DemoPanel'
import { PageHeader, Section } from '@/components/ui'

export default function Demo() {
  return (
    <>
      <PageHeader eyebrow="Project · Live demo" title="Classify an image, in FP32 and as ztachip would">
        <p>Each image is centre-cropped to 32×32 and run through the float model and the exact uint8 file packaged for the accelerator.</p>
      </PageHeader>
      <Section>
        <DemoPanel />
      </Section>
    </>
  )
}
