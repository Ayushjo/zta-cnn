import { useLayoutEffect, useRef, useState } from 'react'

export interface TreemapItem { id: string; label: string; value: number; color: string; detail?: string }
interface Rect { x: number; y: number; w: number; h: number }

// Squarified treemap (Bruls, Huizing & van Wijk, 2000).
function squarify(items: TreemapItem[], box: Rect): (Rect & { item: TreemapItem })[] {
  const total = items.reduce((a, b) => a + b.value, 0)
  if (!total || box.w <= 0 || box.h <= 0) return []
  const scale = (box.w * box.h) / total
  const nodes = items.filter((i) => i.value > 0).sort((a, b) => b.value - a.value).map((item) => ({ item, a: item.value * scale }))
  const out: (Rect & { item: TreemapItem })[] = []
  let { x, y, w, h } = box
  const worst = (row: { a: number }[], side: number) => {
    const s = row.reduce((t, r) => t + r.a, 0)
    const mx = Math.max(...row.map((r) => r.a)), mn = Math.min(...row.map((r) => r.a))
    return Math.max((side * side * mx) / (s * s), (s * s) / (side * side * mn))
  }
  let row: typeof nodes = []
  const flush = () => {
    const s = row.reduce((t, r) => t + r.a, 0)
    if (w >= h) {
      const cw = s / h
      let cy = y
      row.forEach((r) => { const ch = r.a / cw; out.push({ x, y: cy, w: cw, h: ch, item: r.item }); cy += ch })
      x += cw; w -= cw
    } else {
      const ch = s / w
      let cx = x
      row.forEach((r) => { const cw = r.a / ch; out.push({ x: cx, y, w: cw, h: ch, item: r.item }); cx += cw })
      y += ch; h -= ch
    }
    row = []
  }
  for (const n of nodes) {
    const side = Math.min(w, h)
    if (row.length && worst([...row, n], side) > worst(row, side)) flush()
    row.push(n)
  }
  if (row.length) flush()
  return out
}

/** Calm treemap: 3px gutters, soft top wash, labels inside, accent stroke on hover. */
export default function Treemap({ items, height = 320, format }: { items: TreemapItem[]; height?: number; format: (v: number) => string }) {
  const ref = useRef<HTMLDivElement>(null)
  const [w, setW] = useState(0)
  const [hover, setHover] = useState<string | null>(null)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(() => setW(el.clientWidth))
    ro.observe(el)
    setW(el.clientWidth)
    return () => ro.disconnect()
  }, [])
  const tiles = squarify(items, { x: 0, y: 0, w, h: height })
  const total = items.reduce((a, b) => a + b.value, 0)
  const hov = tiles.find((t) => t.item.id === hover)
  return (
    <div>
      <div ref={ref} className="relative w-full overflow-hidden rounded-[10px]" style={{ height }} role="list" aria-label="Treemap">
        {tiles.map((t) => {
          const iw = t.w - 3, ih = t.h - 3
          const on = hover === t.item.id
          return (
            <div key={t.item.id} role="listitem" aria-label={`${t.item.label}, ${format(t.item.value)}`}
              onMouseEnter={() => setHover(t.item.id)} onMouseLeave={() => setHover(null)}
              className="absolute overflow-hidden transition-[background-image] duration-100"
              style={{
                left: t.x + 1.5, top: t.y + 1.5, width: Math.max(0, iw), height: Math.max(0, ih),
                borderRadius: Math.min(6, Math.min(iw, ih) / 3),
                background: `linear-gradient(rgba(255,255,255,${on ? 0.28 : 0.14}), rgba(255,255,255,${on ? 0.12 : 0})), ${t.item.color}`,
                boxShadow: on ? 'inset 0 0 0 2px var(--color-accent)' : undefined,
              }}>
              {iw > 52 && ih > 22 && (
                <div className="px-2 pt-1.5">
                  <div className="truncate text-[12px] font-medium" style={{ color: 'rgba(37,43,49,0.85)' }}>{t.item.label}</div>
                  {iw > 80 && ih > 46 && <div className="font-mono text-[11px]" style={{ color: 'rgba(37,43,49,0.6)' }}>{format(t.item.value)}</div>}
                </div>
              )}
            </div>
          )
        })}
      </div>
      <div className="figure-sm mt-2 h-5 text-ink-2">
        {hov ? `${hov.item.label} — ${format(hov.item.value)} · ${((hov.item.value / total) * 100).toFixed(1)}%${hov.item.detail ? ` · ${hov.item.detail}` : ''}` : 'Area is compute (multiply-accumulates per image)'}
      </div>
    </div>
  )
}
