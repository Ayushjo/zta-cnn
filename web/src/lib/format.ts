export const MODEL_NAMES: Record<string, string> = {
  zta_plain: 'ZTA-Plain',
  zta_resnet8: 'ZTA-ResNet8',
  zta_mobile: 'ZTA-Mobile',
  chatgpt_baseline: 'Baseline CNN',
}
export const ZTA_MODELS = ['zta_plain', 'zta_resnet8', 'zta_mobile']
export const ALL_MODELS = [...ZTA_MODELS, 'chatgpt_baseline']

export const VARIANT_LABEL: Record<string, string> = {
  keras: 'FP32 (Keras)',
  fp32: 'FP32 (TFLite)',
  int8_pc: 'INT8 per-channel',
  int8_pt: 'INT8 per-tensor',
  uint8: 'UINT8 · ztachip',
}

// Calm data palette (charts and bars only; never text or chrome).
export const DATA = {
  slate: '#849bb8', violet: '#a795c7', rose: '#c78797', sage: '#7ba89c', sand: '#b9a071', sky: '#8facc0', stone: '#a2a4ac',
}
export const COLORS = {
  ink: '#252b31', ink2: '#66717e', ink3: '#949aa2', line: '#e4e3df', accent: '#7966da',
  ...DATA,
}
// UINT8 (the deployed model) is slate; FP32 reference is stone; TF's incompatible default is rose.
export const VARIANT_COLOR: Record<string, string> = {
  keras: DATA.stone, fp32: DATA.stone, int8_pc: DATA.rose, int8_pt: DATA.sand, uint8: DATA.slate,
}
export const MODEL_COLOR: Record<string, string> = {
  zta_plain: DATA.slate, zta_resnet8: DATA.sage, zta_mobile: DATA.sand, chatgpt_baseline: DATA.stone,
}

export const pct = (x: number, d = 1) => `${(x * 100).toFixed(d)}%`
export const num = (x: number) => x.toLocaleString('en-US')
export const mega = (x: number, d = 1) => `${(x / 1e6).toFixed(d)} M`
