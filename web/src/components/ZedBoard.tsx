// Proposed Phase II system on the ZedBoard (Zynq-7020). Inline SVG so it scales on a projector.
export default function ZedBoard() {
  const box = (x: number, y: number, w: number, h: number, title: string, lines: string[], accent = false, dashed = false) => (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={10} fill={accent ? 'rgb(132 155 184 / 0.16)' : 'transparent'}
        stroke={accent ? '#849bb8' : 'var(--color-line)'} strokeWidth={accent ? 1.4 : 1} strokeDasharray={dashed ? '5 4' : undefined} />
      <text x={x + 14} y={y + 26} fontFamily="var(--font-sans)" fontSize={15} fontWeight={500} fill="var(--color-ink)">{title}</text>
      {lines.map((l, i) => (
        <text key={i} x={x + 14} y={y + 48 + i * 18} fontFamily="var(--font-mono)" fontSize={11.5} fill="var(--color-ink-2)">{l}</text>
      ))}
    </g>
  )
  return (
    <svg viewBox="0 0 1000 430" className="w-full" role="img" aria-label="ZedBoard block diagram">
      <rect x={10} y={10} width={980} height={410} rx={16} fill="none" stroke="var(--color-line)" strokeWidth={1} />
      <text x={30} y={40} fontFamily="var(--font-mono)" fontSize={12} letterSpacing="1.5" fill="var(--color-ink-3)">ZEDBOARD · XC7Z020</text>
      <text x={40} y={78} fontFamily="var(--font-mono)" fontSize={11} fill="var(--color-ink-3)">PROCESSING SYSTEM (PS)</text>
      <text x={430} y={78} fontFamily="var(--font-mono)" fontSize={11} fill="var(--color-ink-3)">PROGRAMMABLE LOGIC (PL)</text>
      <line x1={395} y1={60} x2={395} y2={400} stroke="var(--color-line)" strokeDasharray="4 5" />
      {box(40, 95, 320, 110, 'ARM Cortex-A9', ['FSBL / ps7_init: configures DDR', 'loads bitstream + firmware'])}
      {box(40, 235, 320, 150, 'DDR3 · 512 MB', ['PS memory controller', 'model weights, tensors,', 'firmware (RISC-V code)'])}
      {box(430, 95, 330, 290, 'ztachip SoC', ['VexRiscv RV32IM · control', 'tensor engine: 4 pcores', '  × 16 threads, 8-wide vectors', 'TFLite runtime (uint8)', '', 'runs cifar10_zta.tflite', 'golden test: test_cifar10()'], true)}
      {box(800, 95, 170, 80, 'Camera', ['Pmod · OV7670-style'], false, true)}
      {box(800, 195, 170, 80, 'VGA', ['Pmod · result overlay'], false, true)}
      {box(800, 295, 170, 90, 'UART', ['115200 baud', 'Top-5 + timing'])}
      <g stroke="var(--color-ink-2)" strokeWidth={1.6} fill="none">
        <path d="M360 310 H430" markerEnd="url(#a2)" markerStart="url(#a2s)" />
      </g>
      <text x={364} y={300} fontFamily="var(--font-mono)" fontSize={10.5} fill="var(--color-ink-2)">S_AXI_HP</text>
      <text x={364} y={330} fontFamily="var(--font-mono)" fontSize={10.5} fill="var(--color-ink-2)">64-bit</text>
      <g stroke="var(--color-line)" strokeWidth={1.6}>
        <path d="M760 135 H800" /><path d="M760 235 H800" /><path d="M760 340 H800" /><path d="M200 205 V235" />
      </g>
      <defs>
        <marker id="a2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--color-ink-2)" /></marker>
        <marker id="a2s" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M10 0L0 5L10 10z" fill="var(--color-ink-2)" /></marker>
      </defs>
    </svg>
  )
}
