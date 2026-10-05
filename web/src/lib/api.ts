import { useEffect, useState } from 'react'
import type { Metrics } from './types'

/** Static build (GitHub Pages): API answers are pre-exported JSON files, no backend. */
export const STATIC = import.meta.env.VITE_STATIC === '1'
export const BASE = import.meta.env.BASE_URL

// Build-time snapshot of reports/metrics.json (copied by `make web-build`), used when the
// backend is unreachable so the results pages still render during the talk.
const snapshots = import.meta.glob<{ default: unknown }>('../data/*.json', { eager: true })
const snapshot = (name: string) => snapshots[`../data/${name}.json`]?.default

const slug = (s: string) => s.replace(/[^A-Za-z0-9]+/g, '_').replace(/^_+|_+$/g, '')

/** Map a live API path to its pre-exported file (see server/export_static.py). */
function staticPath(path: string): string {
  const u = new URL(path, 'http://x')
  const p = u.pathname.replace(/^\/api\//, '')
  let m: RegExpMatchArray | null
  if (p === 'samples') return `samples-${((Number(u.searchParams.get('seed') ?? 1) - 1) % 3) + 1}.json`
  if (p === 'source') return `source/${slug(u.searchParams.get('ref') ?? '')}.json`
  if ((m = p.match(/^predict_index\/(\d+)$/))) return `predict/${u.searchParams.get('model')}/${m[1]}.json`
  return `${p}.json`
}

export const apiUrl = (path: string) => (STATIC ? `${BASE}static-api/${staticPath(path)}` : path)
export const figureUrl = (name: string) => (STATIC ? `${BASE}figures/${name}` : `/static/figures/${name}`)

const cache = new Map<string, unknown>()

export async function getJSON<T>(path: string): Promise<T> {
  if (cache.has(path)) return cache.get(path) as T
  const r = await fetch(apiUrl(path))
  if (!r.ok) throw new Error(`${r.status} ${path}`)
  const data = (await r.json()) as T
  cache.set(path, data)
  return data
}

export async function postForm<T>(path: string, form: FormData, signal?: AbortSignal): Promise<T> {
  if (STATIC) throw new Error('Live prediction needs the local server (make web)')
  const r = await fetch(path, { method: 'POST', body: form, signal })
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail ?? `${r.status}`)
  return (await r.json()) as T
}

export function useApi<T>(path: string | null, fallback?: string) {
  const [data, setData] = useState<T | null>(path && cache.has(path) ? (cache.get(path) as T) : null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    if (!path) return
    let live = true
    getJSON<T>(path)
      .then((d) => live && (setData(d), setError(null)))
      .catch((e: Error) => {
        if (!live) return
        const snap = fallback ? (snapshot(fallback) as T | undefined) : undefined
        if (snap) setData(snap)
        else setError(e.message)
      })
    return () => {
      live = false
    }
  }, [path, fallback])
  return { data, error, loading: !data && !error }
}

export const useMetrics = () => useApi<Metrics>('/api/metrics', 'metrics')
