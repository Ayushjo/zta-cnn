import { FileUp } from 'lucide-react'
import { motion } from 'motion/react'
import { useEffect, useState } from 'react'
import { STATIC, getJSON, postForm, useApi } from '@/lib/api'
import type { CheckResult } from '@/lib/types'
import { FigureStrip, Hairline, Loading, MonoLabel, SafetyLabel, Select, SourceButton, StatusBadge } from './ui'

export default function CheckerView({ initial = 'ztachip_mobilenet', compact = false }: { initial?: string; compact?: boolean }) {
  const { data: samples } = useApi<{ id: string; name: string }[]>('/api/check/samples')
  const [sample, setSample] = useState(initial)
  const [res, setRes] = useState<CheckResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    if (!sample) return
    setBusy(true)
    getJSON<CheckResult>(`/api/check/${sample}`).then((r) => (setRes(r), setErr(null))).catch((e) => setErr(String(e))).finally(() => setBusy(false))
  }, [sample])

  const upload = async (f?: File) => {
    if (!f) return
    setBusy(true)
    const form = new FormData()
    form.append('file', f)
    try {
      setRes(await postForm<CheckResult>('/api/check', form))
      setSample('')
      setErr(null)
    } catch (e) {
      setErr((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const failed = res?.rules.filter((r) => r.status !== 'PASS') ?? []
  const rules = res ? (compact ? [...failed, ...res.rules.filter((r) => r.status === 'PASS')].slice(0, 8) : res.rules) : []
  const badOps = new Set((res?.rules.find((r) => r.id === 'ops')?.issues ?? []).map((i) => i.split(' ').pop()))

  return (
    <div>
      <div className="flex flex-wrap items-center gap-2">
        <Select label="Model" value={sample} onChange={setSample}
          options={[...(sample ? [] : [{ value: '', label: res?.name ?? 'Uploaded file' }]), ...(samples ?? []).map((s) => ({ value: s.id, label: s.name }))]} />
        {!STATIC && <label className="btn btn-secondary cursor-pointer">
          <FileUp size={13} /> Choose .tflite…
          <input type="file" accept=".tflite" className="hidden" onChange={(e) => upload(e.target.files?.[0])} />
        </label>}
        {STATIC && <span className="text-secondary text-ink-3">Checking your own file needs the local server.</span>}
        {busy && <Loading label="Checking" />}
      </div>
      {err && <p className="mt-3 text-secondary text-danger">{err}</p>}

      {res && (
        <motion.div key={res.name + res.compatible} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.15 }}>
          <div className="mt-6">
            <SafetyLabel tone={res.compatible ? 'safe' : 'danger'} className="!text-body font-medium">
              {res.compatible ? 'Runs on ztachip' : 'Would not run correctly on ztachip'}
            </SafetyLabel>
            <div className="mt-1 text-[23px] font-semibold tracking-[-0.01em] text-ink">{res.name}</div>
          </div>
          {!compact && (
            <div className="mt-5">
              <FigureStrip items={[
                { label: 'Rules passed', value: `${res.rules.filter((r) => r.status === 'PASS').length} / ${res.rules.length}` },
                { label: 'Input', value: <span className="text-[17px]">{res.info.input.dtype} [{res.info.input.shape.join(',')}]</span>,
                  detail: res.info.camera_bytes_direct === undefined ? undefined : res.info.camera_bytes_direct ? 'raw camera bytes feed in directly' : 'needs input rescaling' },
                { label: 'Output', value: <span className="text-[17px]">{res.info.output.dtype} [{res.info.output.shape.join(',')}]</span> },
              ]} />
            </div>
          )}
          <div className="mt-5 flex flex-wrap items-center gap-x-4 gap-y-1">
            <MonoLabel>Operators</MonoLabel>
            {Object.entries(res.ops).map(([op, n]) => (
              <span key={op} className={`figure-sm ${badOps.has(op) ? 'text-danger' : 'text-ink-2'}`}>{op} ×{n}</span>
            ))}
          </div>
          <Hairline className="mt-5" />
          <table className="list">
            <thead><tr><th className="w-28">Status</th><th>Rule</th>{!compact && <th className="r">Evidence</th>}</tr></thead>
            <tbody>
              {rules.map((r) => (
                <tr key={r.id}>
                  <td><StatusBadge status={r.status} /></td>
                  <td>
                    <div className="text-body text-ink">{r.rule}</div>
                    {r.issues.length > 0 && (
                      <ul className="mt-1 space-y-0.5">
                        {r.issues.slice(0, compact ? 2 : 6).map((i) => (
                          <li key={i} className={`figure-sm text-ink-2 ${compact ? 'max-w-[1150px] truncate' : 'break-all'}`}>{i}</li>
                        ))}
                      </ul>
                    )}
                    {compact && <div className="mt-1"><SourceButton ref_={r.ref} label="View code" /></div>}
                  </td>
                  {!compact && (
                    <td className="r whitespace-nowrap">
                      <div className="figure-sm mb-0.5 max-w-[280px] truncate text-ink-3" title={r.ref}>{r.ref.replace('SW/apps/nn/', '')}</div>
                      <SourceButton ref_={r.ref} label="View code" />
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </motion.div>
      )}
    </div>
  )
}
