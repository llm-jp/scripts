// サーバ (evalviz_server) が返す形。サーバ側の ljeval.py / components.py / index.py / httpd.py と対応させる (変えたら両方直す)

export type Lang = 'JA' | 'EN' | 'ALL'
export type Status = 'correct' | 'wrong' | 'out_of_label' | 'unparsed' | 'empty' | 'scored'

export interface DatasetScore {
  lang: 'JA' | 'EN'
  category: string | null
  /** カテゴリの集計に使う主指標 */
  metric: string
  score: number
  /** 選択肢外・抽出不可の率 (llm-jp-eval の {ds}_ool) */
  ool: number | null
}

export interface LjEvalSummary {
  version: string
  dir: string
  status: 'done' | 'missing' | 'broken'
  has_offline?: boolean
  avg?: Partial<Record<Lang, number | null>>
  /** 言語 -> カテゴリ -> スコア (v1.4.1 は ALL のみ) */
  categories?: Partial<Record<Lang, Record<string, number>>>
  category_desc?: Record<string, string>
  datasets?: Record<string, DatasetScore>
  n_samples?: number | null
  few_shots?: number | null
  infer_s?: number | null
  run_name?: string | null
}

export interface SwallowSummary {
  status: string
  scores?: Record<string, number>
  avg?: { EN: number | null; JA: number | null }
  n_en?: number
  n_ja?: number
}

export interface JudgeSummary {
  status: string
  judge_model?: string
  /** ベンチマーク -> 観点 -> 値 */
  scores?: Record<string, Record<string, number>>
  errors?: Record<string, Record<string, number>>
}

export interface SafetyBench {
  kind: string
  metrics: Record<string, number | null>
  headline: number | null
  headline_label: string
}

export interface SafetySummary {
  status: string
  benchmarks?: Record<string, SafetyBench>
}

export interface Run {
  id: string
  name: string
  group: string
  root: string
  dir: string
  model: string
  model_path: string
  step: number | null
  /** 名前のステップ番号を * にしたもの (チェックポイントの推移をまとめる) */
  series: string | null
  settings: Record<string, string | number | boolean>
  finished_at: string | null
  created_at: string | null
  error?: string
  components: {
    'llm-jp-eval'?: Record<string, LjEvalSummary>
    swallow?: SwallowSummary
    'llm-jp-judge'?: JudgeSummary
    'safety-eval'?: SafetySummary
  }
}

export interface RunDetail extends Run {
  config: Record<string, unknown>
  raw_scores: Record<string, Record<string, number>>
}

export interface Def {
  key: string
  label: string
  short?: string
  desc: string
}

export interface Meta {
  statuses: Def[]
  flags: Def[]
  en_datasets: string[]
  category_desc: Record<string, string>
  built_at: number
  refreshing: boolean
  roots: string[]
  last_error: string
}

export interface DatasetAnalysis {
  n: number
  lang: 'JA' | 'EN'
  /** 正誤の判定方法 (exact = 完全一致、math = 数式の同値の近似、null = 正誤なし) */
  judge: 'exact' | 'math' | null
  approx: boolean
  status: Partial<Record<Status, number>>
  flags: Record<string, number>
  out_len: number | null
  reas_len: number | null
  labels?: string[]
  pred_dist?: Record<string, number>
  gold_dist?: Record<string, number>
}

export interface RunAnalysis {
  run: string
  version: string
  available: boolean
  n?: number
  status?: Partial<Record<Status, number>>
  flags?: Record<string, number>
  datasets?: Record<string, DatasetAnalysis>
  out_len_median?: number | null
  reas_len_median?: number | null
  out_len_mean?: number | null
  reas_len_mean?: number | null
}

export interface DatasetInfo {
  dataset: string
  lang: 'JA' | 'EN'
  metrics: string[]
  pattern: string
  labels: string[]
  judge: 'exact' | 'math' | null
  approx: boolean
}

export interface SampleCell {
  st: Status
  p: string
  fl: string[]
  /** 出力・推論の文字数 */
  ol: number
  rl: number
}

export interface SampleRow {
  key: string
  index: number
  input: string
  gold: string
  /** 選択中の実行の順。その実行に無いサンプルは null */
  cells: (SampleCell | null)[]
}

export interface SamplesResponse {
  info: DatasetInfo | null
  rows: SampleRow[]
}

export interface SampleRunDetail {
  run: string
  found: boolean
  index?: number
  input?: string
  prompt?: string
  system_prompt?: string | null
  gold?: string
  generated?: string
  reasoning?: string
  metadata?: Record<string, unknown>
  instruction?: string
  output_length?: number
  status?: Status
  pred?: string
  flags?: string[]
}

export interface SampleResponse {
  info: DatasetInfo | null
  key: string
  runs: SampleRunDetail[]
}

export interface FeaturedModel {
  name: string
  family: string | null
  params: string | null
  type: string | null
  /** 他組織のモデル (比較対象) */
  baseline: boolean
  note: string | null
  runs: string[]
  /** 評価 -> 値を取る実行の ID。llm-jp-eval は標準条件に合う実行だけ */
  source: Partial<Record<'llm-jp-eval' | 'swallow' | 'llm-jp-judge' | 'safety-eval', string>>
}

export interface FeaturedResponse {
  title: string | null
  standard: { 'llm-jp-eval'?: { version?: string; max_num_samples?: number } }
  models: FeaturedModel[]
  /** 設定の問題 (実行が見つからない・標準条件の結果が無い等) */
  problems: string[]
  path: string
  error: string
}
