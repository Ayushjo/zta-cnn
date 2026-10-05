import ZedBoard from '@/components/ZedBoard'
import { Empty, Figure, MonoLabel, PageHeader, SafetyLabel, Section, SourceButton } from '@/components/ui'
import { useApi } from '@/lib/api'
import type { Golden } from '@/lib/types'

export const PORT_STEPS: { t: string; d: string; ref?: string }[] = [
  { t: 'Memory', d: 'Replace the Arty’s MIG DDR controller with the Zynq PS DDR through an S_AXI_HP port.', ref: 'HW/examples/GHRD/main.v' },
  { t: 'Bus width', d: 'HP ports are 64-bit, so set exmem_data_width_c to 64; the reference design uses 128.', ref: 'HW/src/config.vhd:55' },
  { t: 'Board I/O', d: 'Drop the Arty-only Ethernet IP, re-pin the XDC for ZedBoard Pmods, clock from FCLK through an MMCM.' },
  { t: 'Software', d: 'Rebase RAM in SW/linker.ld onto the PS-DDR window and keep NUM_PCORE at 4.' },
  { t: 'Fit check first', d: 'The Z-7020 has about 84% of the Arty A7-100T’s LUTs and ztachip’s utilisation is unpublished, so synthesis comes first. With an Arty A7-100T, the reference flow is used directly.' },
]

export const TIMELINE = [
  { p: 'Phase I', w: 'Oct 4–6', d: 'Software toolchain, trained and quantized models, compatibility proof, CPU baseline, this site', done: true },
  { p: 'Phase II', w: 'Oct–Nov', d: 'Quantization-aware training, width sweep, ZedBoard synthesis and port' },
  { p: 'Phase III', w: 'Nov', d: 'Model in ztachip firmware, test_cifar10 on the board, camera to 32×32 resize chain' },
  { p: 'Phase IV', w: 'Dec', d: 'On-board accuracy, latency, power and FPGA resources against the CPU; final report' },
]

export default function Hardware() {
  const { data: G } = useApi<Golden>('/api/golden')
  return (
    <>
      <PageHeader eyebrow="Evidence · Hardware plan" title="From a verified file to the ZedBoard">
        <p>ztachip’s neural-network path cannot run on a PC or in its RTL simulator, so hardware numbers come only from the board; nothing has run on the FPGA yet.</p>
      </PageHeader>

      <Section kicker="01 / Proposed system">
        <Figure caption="ztachip sits in the programmable logic and reaches the PS-side DDR through a 64-bit high-performance AXI port. Camera and VGA are optional for the golden test.">
          <ZedBoard />
        </Figure>
      </Section>

      <Section kicker="02 / Porting ztachip">
        <ol>
          {PORT_STEPS.map((s, i) => (
            <li key={s.t} className="grid grid-cols-[48px_170px_1fr_auto] items-baseline gap-3 border-b border-line/70 py-3 max-md:grid-cols-[48px_1fr]">
              <span className="mono-label">{String(i + 1).padStart(2, '0')}</span>
              <span className="text-body font-medium text-ink">{s.t}</span>
              <span className="text-secondary text-ink-2 max-md:col-start-2">{s.d}</span>
              <span className="max-md:col-start-2">
                {s.ref?.startsWith('SW') ? <SourceButton ref_={s.ref} /> : s.ref ? <span className="figure-sm text-ink-3">{s.ref}</span> : null}
              </span>
            </li>
          ))}
        </ol>
      </Section>

      <Section kicker="03 / Golden test vectors" detail={G ? `${G.vectors.length} vectors` : undefined}>
        <p className="mb-5 max-w-3xl text-secondary text-ink-2">
          One test image per class as a 24-bit BMP, the format ztachip’s CreateWithBitmap reads, each with the expected raw uint8
          output in the format of ztachip’s own classifier.bin. An on-board test_cifar10() prints the top class, the largest byte
          difference and the inference time for each.
        </p>
        {G ? (
          <div className="grid grid-cols-2 gap-x-4 gap-y-5 sm:grid-cols-5">
            {G.vectors.map((v, k) => (
              <div key={k}>
                <img src={v.image} alt={v.label} className="pixelated aspect-square w-full rounded-md" />
                <div className="mt-2 text-body font-medium capitalize text-ink">{v.label}</div>
                <div className="figure-sm text-ink-3">expect raw {v.ztachip_top5[0][1]}</div>
              </div>
            ))}
          </div>
        ) : <Empty title="Not generated yet">Created by <code className="inline-code">make export</code> after training.</Empty>}
      </Section>

      <Section kicker="04 / Timeline">
        <ol>
          {TIMELINE.map((t) => (
            <li key={t.p} className="grid grid-cols-[110px_90px_1fr_auto] items-baseline gap-3 border-b border-line/70 py-3 max-md:grid-cols-[110px_1fr]">
              <span className="text-body font-medium text-ink">{t.p}</span>
              <MonoLabel>{t.w}</MonoLabel>
              <span className="text-secondary text-ink-2 max-md:col-span-2">{t.d}</span>
              {t.done ? <SafetyLabel tone="safe">Done</SafetyLabel> : <SafetyLabel tone="neutral">Planned</SafetyLabel>}
            </li>
          ))}
        </ol>
      </Section>
    </>
  )
}
