export type Variant = 'keras' | 'fp32' | 'int8_pc' | 'int8_pt' | 'uint8'

export interface VariantMetrics {
  top1: number
  top3: number
  precision: number[]
  recall: number[]
  f1: number[]
  ece: number
  reliability: [number, number, number][]
  confusion: number[][]
  agreement_with_keras?: number
  size_kib?: number
}

export interface Rule {
  id: string
  rule: string
  status: 'PASS' | 'FAIL' | 'WARN'
  issues: string[]
  ref: string
}

export interface Layer {
  name: string
  type: string
  output: number[]
  params: number
  macs: number
}

export interface Profile {
  params: number
  macs: number
  peak_activation_bytes_uint8: number
  weight_bytes_uint8: number
  zta_bound_ms: number
  zta_bound_compute_ms: number
  zta_bound_memory_ms: number
  layers: Layer[]
}

export interface ModelMetrics {
  variants: Record<Variant, VariantMetrics>
  latency_ms: Record<string, [number, number]>
  ztachip_compatible: boolean
  ztachip_check: Rule[]
  profile: Profile
  train?: { best_val_accuracy: number; epochs: number; sec_per_epoch: number }
}

export interface Env { cpu?: string; tensorflow?: string; python?: string; test_images?: number }

export type Metrics = Record<string, ModelMetrics> & { _env?: Env }

export interface Meta {
  classes: string[]
  deploy_model: string
  models: { id: string; name: string; available: boolean; ztachip: boolean }[]
  env: Env
  has_metrics: boolean
  has_robustness: boolean
  has_golden: boolean
  rules: number
  figures: string[]
}

export interface Prediction {
  model: string
  preview: string
  fp32: { probs: number[]; top: number; ms: number }
  uint8: { probs: number[]; top: number; ms: number; raw: number[]; out_scale: number; out_zero_point: number }
  agree: boolean
  ztachip_top5: { index: number; label: string; raw: number; pct: number }[]
  gradcam: string | null
  true?: number
}

export interface Sample { index: number; label: string; image: string }

export interface CheckResult {
  name: string
  compatible: boolean
  ops: Record<string, number>
  info: {
    input: { shape: number[]; dtype: string; scale: number; zero_point: number }
    output: { shape: number[]; dtype: string; scale: number; zero_point: number }
    classes: number
    camera_bytes_direct?: boolean
  }
  rules: Rule[]
}

export interface SourceSnippet { file: string; start: number; highlight: [number, number]; lines: string[] }

export interface History {
  accuracy: number[]
  val_accuracy: number[]
  loss: number[]
  val_loss: number[]
  params: number
  best_val_accuracy: number
}

export type Robustness = Record<string, Record<string, { fp32: number | number[]; uint8: number | number[] }>>

export interface TensorRow {
  name: string
  shape: number[]
  int8: { dtype: string; scale: number; zero_point: number }
  uint8: { dtype: string; scale: number; zero_point: number }
}

export interface Golden {
  model: string
  vectors: { index: number; label: string; raw_uint8: number[]; ztachip_top5: [string, number, number][]; image: string }[]
}
