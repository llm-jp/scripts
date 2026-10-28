import { useQuery } from '@tanstack/react-query'
import { createFileRoute, Link, useNavigate } from '@tanstack/react-router'
import type { ReactNode } from 'react'
import { useMemo } from 'react'
import { analysisQuery, metaQuery } from '@/api/queries'
import type { RunAnalysis, Status } from '@/api/types'
import { CountMatrix } from '@/components/charts/RunBars'
import { StatusBar, StatusLegend } from '@/components/charts/StatusBar'
import { NoRunsSelected, PageHeader } from '@/components/layout/PageHeader'
import { Section } from '@/components/layout/Section'
import { RunLabel } from '@/components/runs/RunLabel'
import { RunColumnHead } from '@/components/runs/RunColumnHead'
import { VersionSelect } from '@/components/runs/VersionSelect'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { type SelectedRun, useSelectedRuns } from '@/hooks/useSelectedRuns'
import { useVersion } from '@/hooks/useVersion'
import { compact, pct } from '@/lib/format'
import { ERROR_STATUSES, STATUSES } from '@/lib/status'
import { fmtRate, seqShade } from '@/lib/shade'
import { cn } from '@/lib/utils'

interface ErrorsSearch {
  /** データセット別の内訳に出すもの: "st:<状態>" か "fl:<フラグ>" */
  by?: string
  /** ラベルの偏りを見るデータセット */
  lds?: string
}

export const Route = createFileRoute('/errors')({
  validateSearch: (s: Record<string, unknown>): ErrorsSearch => ({
    by: typeof s.by === 'string' && s.by ? s.by : undefined,
    lds: typeof s.lds === 'string' && s.lds ? s.lds : undefined,
  }),
  component: ErrorsPage,
})

function ErrorsPage() {
  const search = Route.useSearch()
  const navigate = useNavigate({ from: Route.fullPath })
  const set = (patch: Partial<ErrorsSearch>) => navigate({ search: (p) => ({ ...p, ...patch }), replace: true, resetScroll: false })
  const { ids, selected } = useSelectedRuns()
  const { ver, versions, setVer } = useVersion(selected)
  const analysis = useQuery(analysisQuery(ids, ver))
  const meta = useQuery(metaQuery())
  const byRun = useMemo(() => new Map((analysis.data ?? []).filter((a) => a.available).map((a) => [a.run, a])), [analysis.data])
  const runs = selected.filter((r) => byRun.has(r.id))
  const missing = selected.filter((r) => analysis.data && !byRun.has(r.id))

  if (!ids.length) return <NoRunsSelected what="誤りパターン" />
  const by = search.by ?? 'st:wrong'
  const toSamples = (ds: string, key: string) =>
    navigate({ to: '/samples', search: (p) => ({ runs: p.runs, ver: p.ver, ds, ...(key.startsWith('st:') ? { st: key.slice(3) as Status } : { fl: key.slice(3) }) }) })
  const flagDefs = (meta.data?.flags ?? []).filter((f) => runs.some((r) => byRun.get(r.id)?.flags?.[f.key]))

  return (
    <div className="flex flex-col gap-4 p-4">
      <PageHeader
        title="誤りパターン"
        description={
          <>
            llm-jp-eval {ver} の推論結果 (offline/*.eval-generated.json) をサンプルごとに分類した集計。正誤は llm-jp-eval の回答抽出を移植して作り直したもので、
            exact_match の正解率と ool は result.json と一致する。数式の同値判定は近似、BLEU・F1 等だけのデータセットは「正誤なし」。
          </>
        }
      >
        <VersionSelect ver={ver} versions={versions} onChange={setVer} />
      </PageHeader>
      {analysis.isFetching && !analysis.data && <p className="text-sm text-muted-foreground">推論結果を集計中… (初回は実行ごとに数秒〜数十秒。結果はサーバにキャッシュされる)</p>}
      {analysis.error && <p className="text-sm text-[var(--status-critical)]">集計に失敗: {String(analysis.error)}</p>}
      {missing.length > 0 && (
        <p className="text-xs text-muted-foreground">
          推論結果の無い実行 (v1.4.1、または offline/ が無い) は除いた: {missing.map((r) => r.name).join(', ')}
        </p>
      )}

      {runs.length > 0 && (
        <>
          <Section
            title="サンプルの状態"
            description="全データセットのサンプルの内訳。区間のクリックで、その状態のデータセット別の内訳へ"
            actions={<StatusLegend />}
          >
            <div className="grid grid-cols-[minmax(10rem,20rem)_1fr_auto] items-center gap-x-3 gap-y-1.5 text-sm">
              <span />
              <span />
              <span className="grid grid-cols-[4.5rem_4.5rem_4.5rem] text-right text-xs text-muted-foreground">
                <span title="正誤を出せるサンプルでの正解の割合">正解率</span>
                <span title="出力が空・抽出不可・選択肢外の割合 (llm-jp-eval の ool)">形式の失敗</span>
                <span>サンプル</span>
              </span>
              {runs.map((r) => {
                const a = byRun.get(r.id)!
                const st = a.status ?? {}
                const judged = (st.correct ?? 0) + (st.wrong ?? 0) + (st.out_of_label ?? 0) + (st.unparsed ?? 0) + (st.empty ?? 0)
                const fail = (st.out_of_label ?? 0) + (st.unparsed ?? 0) + (st.empty ?? 0)
                return (
                  <div key={r.id} className="contents">
                    <RunLabel run={r} color={r.color} compact />
                    <StatusBar counts={st} onSegment={(s) => set({ by: `st:${s}` })} />
                    <span className="grid grid-cols-[4.5rem_4.5rem_4.5rem] text-right tabular">
                      <span>{pct(judged ? (st.correct ?? 0) / judged : null)}</span>
                      <span>{pct(a.n ? fail / a.n : null)}</span>
                      <span className="text-muted-foreground">{a.n}</span>
                    </span>
                  </div>
                )
              })}
            </div>
          </Section>

          <Section
            title="誤りの兆候"
            description={
              <>
                出力の中身から経験則で付けた印 (重複あり)。割合は全サンプルに対する率。セルのクリックで、その印のデータセット別の内訳へ。
                {flagDefs.map((f) => (
                  <span key={f.key} className="mt-0.5 block">
                    <b className="font-medium text-foreground">{f.label}</b>: {f.desc}
                  </span>
                ))}
              </>
            }
          >
            {flagDefs.length === 0 ? (
              <p className="text-sm text-muted-foreground">該当なし</p>
            ) : (
              <CountMatrix
                runs={runs}
                asRate
                columns={flagDefs.map((f) => ({ key: f.key, label: f.label, title: f.desc }))}
                count={(rid, k) => byRun.get(rid)?.flags?.[k] ?? 0}
                total={(rid) => byRun.get(rid)?.n ?? 0}
                onCell={(_, k) => set({ by: `fl:${k}` })}
              />
            )}
          </Section>

          <Section
            title="データセット別の内訳"
            description="選んだ状態・印のサンプルの割合をデータセットごとに。どの実行にも無いデータセットは出さない。セルのクリックでサンプル比較 (その状態・印で絞り込み) へ"
            actions={
              <Select value={by} onValueChange={(v) => set({ by: v })}>
                <SelectTrigger size="sm" className="w-56">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {ERROR_STATUSES.map((s) => (
                    <SelectItem key={s} value={`st:${s}`}>
                      状態: {STATUSES[s].symbol} {STATUSES[s].label}
                    </SelectItem>
                  ))}
                  {(meta.data?.flags ?? []).map((f) => (
                    <SelectItem key={f.key} value={`fl:${f.key}`}>
                      印: {f.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            }
          >
            <DatasetBreakdown runs={runs} byRun={byRun} by={by} onCell={toSamples} />
          </Section>

          <div className="grid gap-4 xl:grid-cols-2">
            <Section title="出力の長さ" description="サンプルあたりの文字数の中央値 (括弧内は平均)。推論は reasoning_content (reasoning parser で分離したもの)">
              <LengthTable runs={runs} byRun={byRun} />
            </Section>
            <Section
              title="ラベルの偏り"
              description="選択肢のあるデータセットで、予測したラベルの件数を正解の分布と並べる。特定のラベルに寄っていれば、問題を読まずに答えている疑いがある"
              actions={<LabelDatasetSelect runs={runs} byRun={byRun} value={search.lds} onChange={(v) => set({ lds: v })} />}
            >
              <LabelBias runs={runs} byRun={byRun} ds={search.lds ?? firstLabelDataset(runs, byRun)} />
            </Section>
          </div>
        </>
      )}
    </div>
  )
}

function DatasetBreakdown({ runs, byRun, by, onCell }: { runs: SelectedRun[]; byRun: Map<string, RunAnalysis>; by: string; onCell: (ds: string, key: string) => void }) {
  const [kind, key] = [by.slice(0, 2), by.slice(3)]
  const count = (a: RunAnalysis | undefined, ds: string) => {
    const d = a?.datasets?.[ds]
    return d ? (kind === 'st' ? (d.status[key as Status] ?? 0) : (d.flags[key] ?? 0)) : null
  }
  const dss = new Set<string>()
  for (const r of runs) for (const ds of Object.keys(byRun.get(r.id)?.datasets ?? {})) dss.add(ds)
  const rows = [...dss]
    .map((ds) => ({
      ds,
      rates: runs.map((r) => {
        const d = byRun.get(r.id)?.datasets?.[ds]
        const c = count(byRun.get(r.id), ds)
        return d && c != null ? c / Math.max(1, d.n) : null
      }),
    }))
    .filter((x) => x.rates.some((v) => v))
    .sort((a, b) => Math.max(...b.rates.map((v) => v ?? 0)) - Math.max(...a.rates.map((v) => v ?? 0)) || a.ds.localeCompare(b.ds))
  if (!rows.length) return <p className="text-sm text-muted-foreground">該当するサンプルなし</p>
  return (
    <div className="max-h-[560px] overflow-auto">
      <table className="text-sm">
        <thead className="sticky top-0 z-10 bg-background">
          <tr className="text-xs text-muted-foreground">
            <th className="px-2 py-1 text-left font-medium">データセット</th>
            {runs.map((r) => (
              <th key={r.id} className="w-32 max-w-32 min-w-32 px-2 py-1 align-bottom font-medium">
                <RunColumnHead run={r} all={runs} />
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(({ ds, rates }) => (
            <tr key={ds} className="border-t">
              <td className="px-2 py-1 font-mono text-xs">{ds}</td>
              {rates.map((v, i) => {
                const n = count(byRun.get(runs[i].id), ds)
                return (
                  <td key={runs[i].id} className="p-0.5">
                    <button
                      type="button"
                      disabled={!n}
                      onClick={() => onCell(ds, by)}
                      className="w-full rounded px-2 py-0.5 text-right text-xs tabular enabled:hover:ring-2 enabled:hover:ring-foreground/30"
                      style={{ background: n ? seqShade(v ?? 0) : undefined, color: (v ?? 0) > 0.6 ? '#fff' : undefined }}
                      title={n != null ? `${n} / ${byRun.get(runs[i].id)?.datasets?.[ds]?.n} 件` : 'このデータセットの結果なし'}
                    >
                      {v == null ? <span className="text-muted-foreground/50">—</span> : n ? fmtRate(v) : <span className="text-muted-foreground/50">·</span>}
                    </button>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function LengthTable({ runs, byRun }: { runs: SelectedRun[]; byRun: Map<string, RunAnalysis> }) {
  return (
    <table className="text-sm">
      <thead>
        <tr className="text-xs text-muted-foreground">
          <th className="px-2 py-1 text-left font-medium">実行</th>
          <th className="px-2 py-1 text-right font-medium">最終出力 (字)</th>
          <th className="px-2 py-1 text-right font-medium">推論 (字)</th>
        </tr>
      </thead>
      <tbody>
        {runs.map((r) => {
          const a = byRun.get(r.id)!
          return (
            <tr key={r.id} className="border-t">
              <td className="max-w-[20rem] px-2 py-1">
                <RunLabel run={r} color={r.color} compact />
              </td>
              <td className="px-2 py-1 text-right whitespace-nowrap tabular">
                {compact(a.out_len_median)} <span className="text-xs text-muted-foreground">({compact(a.out_len_mean)})</span>
              </td>
              <td className="px-2 py-1 text-right whitespace-nowrap tabular">
                {compact(a.reas_len_median)} <span className="text-xs text-muted-foreground">({compact(a.reas_len_mean)})</span>
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

function labelDatasets(runs: SelectedRun[], byRun: Map<string, RunAnalysis>): string[] {
  const s = new Set<string>()
  for (const r of runs) for (const [ds, d] of Object.entries(byRun.get(r.id)?.datasets ?? {})) if (d.labels?.length) s.add(ds)
  return [...s].sort()
}

function firstLabelDataset(runs: SelectedRun[], byRun: Map<string, RunAnalysis>): string | undefined {
  // 既定は、予測の分布が正解の分布から最も離れている (偏りが大きい) データセット
  let best: [string, number] | undefined
  for (const ds of labelDatasets(runs, byRun)) {
    for (const r of runs) {
      const d = byRun.get(r.id)?.datasets?.[ds]
      if (!d?.pred_dist || !d.gold_dist) continue
      const keys = new Set([...Object.keys(d.pred_dist), ...Object.keys(d.gold_dist)])
      const tv = [...keys].reduce((a, k) => a + Math.abs((d.pred_dist![k] ?? 0) - (d.gold_dist![k] ?? 0)), 0) / 2 / Math.max(1, d.n)
      if (!best || tv > best[1]) best = [ds, tv]
    }
  }
  return best?.[0]
}

function LabelDatasetSelect({ runs, byRun, value, onChange }: { runs: SelectedRun[]; byRun: Map<string, RunAnalysis>; value?: string; onChange: (v: string) => void }) {
  const dss = labelDatasets(runs, byRun)
  if (!dss.length) return null
  return (
    <Select value={value ?? firstLabelDataset(runs, byRun)} onValueChange={onChange}>
      <SelectTrigger size="sm" className="w-52">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {dss.map((ds) => (
          <SelectItem key={ds} value={ds}>
            {ds}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

function LabelBias({ runs, byRun, ds }: { runs: SelectedRun[]; byRun: Map<string, RunAnalysis>; ds?: string }) {
  if (!ds) return <p className="text-sm text-muted-foreground">選択肢のあるデータセットがない</p>
  const dsa = runs.map((r) => byRun.get(r.id)?.datasets?.[ds])
  const gold = dsa.find((d) => d?.gold_dist)?.gold_dist ?? {}
  const labels = [...(dsa.find((d) => d?.labels)?.labels ?? [])]
  if (dsa.some((d) => d?.pred_dist?.['(選択肢外)'])) labels.push('(選択肢外)')
  const cols: { id: string; head: ReactNode; dist: Record<string, number>; n: number }[] = [
    { id: '__gold', head: <span className="font-medium">正解</span>, dist: gold, n: Object.values(gold).reduce((a, b) => a + b, 0) },
    ...runs.flatMap((r, i) => (dsa[i]?.pred_dist ? [{ id: r.id, head: <RunColumnHead run={r} all={runs} />, dist: dsa[i]!.pred_dist!, n: dsa[i]!.n }] : [])),
  ]
  return (
    <div className="overflow-auto">
      <table className="text-sm">
        <thead>
          <tr className="text-xs text-muted-foreground">
            <th className="px-2 py-1 text-left font-medium">ラベル</th>
            {cols.map((c) => (
              <th key={c.id} className="w-28 max-w-28 min-w-28 px-2 py-1 text-right align-bottom font-medium">
                {c.head}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {labels.map((l) => (
            <tr key={l} className="border-t">
              <td className={cn('max-w-48 truncate px-2 py-1 font-mono text-xs', l === '(選択肢外)' && 'text-muted-foreground')} title={l}>
                {l}
              </td>
              {cols.map((c) => {
                const v = (c.dist[l] ?? 0) / Math.max(1, c.n)
                return (
                  <td key={c.id} className="p-0.5">
                    <div className="rounded px-2 py-0.5 text-right text-xs tabular" style={{ background: seqShade(v), color: v > 0.6 ? '#fff' : undefined }} title={`${c.dist[l] ?? 0} / ${c.n}`}>
                      {c.dist[l] ? `${(v * 100).toFixed(0)}%` : <span className="text-muted-foreground/50">·</span>}
                    </div>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-muted-foreground">
        <Link to="/samples" search={(p) => ({ runs: p.runs, ver: p.ver, ds })} className="underline">
          {ds} のサンプルを比べる
        </Link>
      </p>
    </div>
  )
}
