import { useQuery } from '@tanstack/react-query'
import { createFileRoute, Link, useNavigate } from '@tanstack/react-router'
import { TriangleAlert } from 'lucide-react'
import { useMemo } from 'react'
import { analysisQuery, metaQuery } from '@/api/queries'
import { RunBars } from '@/components/charts/RunBars'
import { LineChart, Scatter } from '@/components/charts/Plot'
import { NoRunsSelected, PageHeader } from '@/components/layout/PageHeader'
import { Section } from '@/components/layout/Section'
import { RunLabel } from '@/components/runs/RunLabel'
import { RunColumnHead } from '@/components/runs/RunColumnHead'
import { VersionSelect } from '@/components/runs/VersionSelect'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { type SelectedRun, useSelectedRuns } from '@/hooks/useSelectedRuns'
import { useVersion } from '@/hooks/useVersion'
import { compact } from '@/lib/format'
import { headlines } from '@/lib/metrics'
import { conditionKey, conditionLabel, ljEval } from '@/lib/runs'
import { divShade, seqShade } from '@/lib/shade'
import { cn } from '@/lib/utils'

interface OverviewSearch {
  /** カテゴリ表の表示: 値 / 先頭の実行との差 */
  diff?: boolean
  /** 散布図の縦軸の言語 */
  lang?: 'JA' | 'EN'
}

export const Route = createFileRoute('/overview')({
  validateSearch: (s: Record<string, unknown>): OverviewSearch => ({
    diff: s.diff === true || s.diff === 'true' ? true : undefined,
    lang: s.lang === 'EN' ? 'EN' : undefined,
  }),
  component: OverviewPage,
})

function OverviewPage() {
  const search = Route.useSearch()
  const navigate = useNavigate({ from: Route.fullPath })
  const set = (patch: Partial<OverviewSearch>) => navigate({ search: (p) => ({ ...p, ...patch }), replace: true, resetScroll: false })
  const { ids, selected } = useSelectedRuns()
  const { ver, versions, setVer } = useVersion(selected)
  const analysis = useQuery(analysisQuery(ids, ver))
  const hs = useMemo(() => headlines(selected, ver), [selected, ver])

  if (!ids.length) return <NoRunsSelected what="総合比較" />
  const conditions = [...new Map(selected.filter((r) => ljEval(r, ver)).map((r) => [conditionKey(r, ver), conditionLabel(r, ver)])).values()]
  const lang = search.lang ?? 'JA'
  const lenOf = new Map((analysis.data ?? []).filter((a) => a.available).map((a) => [a.run, (a.out_len_mean ?? 0) + (a.reas_len_mean ?? 0)]))

  // チェックポイントの推移: ステップ番号のある実行を系列 (名前のステップを * にしたもの) ごとにまとめる。色は系列の先頭の実行の色
  const series = new Map<string, SelectedRun[]>()
  for (const r of selected) if (r.step != null && r.series) series.set(r.series, [...(series.get(r.series) ?? []), r])
  const lines = [...series.entries()].filter(([, rs]) => rs.length >= 2)

  return (
    <div className="flex flex-col gap-4 p-4">
      <PageHeader
        title="総合比較"
        description={`選択中の ${selected.length} 実行を、評価ごとの代表値で比べる。棒は良い順に並べ、色は実行の識別色 (並べ替えても変わらない)。`}
      >
        <VersionSelect ver={ver} versions={versions} onChange={setVer} />
      </PageHeader>

      {conditions.length > 1 && (
        <div className="flex items-start gap-2 rounded-lg border border-[var(--status-warning)] bg-[var(--status-warning)]/10 px-3 py-2 text-sm">
          <TriangleAlert className="mt-0.5 size-4 shrink-0" />
          <div>
            llm-jp-eval {ver} の評価条件が揃っていない (サンプル数・chat template・推論の設定)。スコアの差には条件の差が含まれる:
            <ul className="mt-1 list-disc pl-5 text-xs text-muted-foreground">
              {conditions.map((c) => (
                <li key={c}>{c}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      <div className="grid gap-4 xl:grid-cols-2 2xl:grid-cols-3">
        {hs.map((h) => (
          <Section
            key={h.id}
            title={
              <span>
                <span className="mr-2 text-xs font-normal text-muted-foreground">{h.source}</span>
                {h.title}
                {h.lowerIsBetter && <span className="ml-2 text-xs font-normal text-muted-foreground">小さいほど良い</span>}
              </span>
            }
            description={h.description}
          >
            <RunBars
              sort
              lowerIsBetter={h.lowerIsBetter}
              max={h.max}
              data={selected.map((r) => {
                const v = h.get(r, ver)
                return { run: r, value: v, label: v == null ? '—' : v.toFixed(h.digits) }
              })}
            />
          </Section>
        ))}
      </div>

      {selected.some((r) => ljEval(r, ver)?.categories) && (
        <Section
          title={`カテゴリ別 (llm-jp-eval ${ver})`}
          description={
            search.diff
              ? '先頭の実行 (基準) との差。青は基準を上回り、赤は下回る (±0.1 で最も濃い)。列見出しのクリックでデータセット別へ'
              : '色の濃さは列 (カテゴリ) の中での相対的な高さ。列見出しのクリックで、そのカテゴリのデータセット別の表へ'
          }
          actions={
            <ToggleGroup type="single" size="sm" variant="outline" value={search.diff ? 'diff' : 'value'} onValueChange={(v) => v && set({ diff: v === 'diff' || undefined })}>
              <ToggleGroupItem value="value">値</ToggleGroupItem>
              <ToggleGroupItem value="diff">基準との差</ToggleGroupItem>
            </ToggleGroup>
          }
        >
          <CategoryTable runs={selected} ver={ver} diff={!!search.diff} />
        </Section>
      )}

      <div className="grid gap-4 xl:grid-cols-2">
        <Section
          title="出力の長さとスコア"
          description="横軸はサンプルあたりの平均文字数 (最終出力 + 推論)。トークン数は推論結果に残らないので文字数で代える。右上ほど長く考えて高い、左上ほど短い出力で高い"
          actions={
            <ToggleGroup type="single" size="sm" variant="outline" value={lang} onValueChange={(v) => v && set({ lang: v === 'EN' ? 'EN' : undefined })}>
              <ToggleGroupItem value="JA">日本語</ToggleGroupItem>
              <ToggleGroupItem value="EN">英語</ToggleGroupItem>
            </ToggleGroup>
          }
        >
          {analysis.isFetching && !analysis.data ? (
            <p className="text-sm text-muted-foreground">推論結果を集計中… (初回は実行ごとに数秒〜数十秒)</p>
          ) : (
            <Scatter
              xLabel="平均出力長 (文字)"
              yLabel={`${lang === 'JA' ? '日本語' : '英語'} AVG`}
              fmtX={compact}
              fmtY={(v) => v.toFixed(2)}
              points={selected.flatMap((r) => {
                const y = ljEval(r, ver)?.avg?.[lang]
                const x = lenOf.get(r.id)
                return y != null && x != null ? [{ id: r.id, x, y, color: r.color, label: <RunLabel run={r} compact /> }] : []
              })}
            />
          )}
          <div className="mt-2 flex flex-wrap gap-x-3 gap-y-0.5 text-xs">
            {selected.map((r) => (
              <RunLabel key={r.id} run={r} color={r.color} compact className="max-w-64" />
            ))}
          </div>
        </Section>
        <Section
          title="チェックポイントの推移"
          description="実行名かモデルパスにステップ番号 (step・iter 等) のある実行を、名前の残りが同じものごとに線で結ぶ。2 点以上ある系列だけ描く"
        >
          {lines.length === 0 ? (
            <p className="text-sm text-muted-foreground">同じ系列のチェックポイントが 2 つ以上選ばれていない</p>
          ) : (
            <div className="flex flex-col gap-2">
              {(['JA', 'EN'] as const).map((l) => (
                <div key={l}>
                  <div className="text-xs font-medium text-muted-foreground">{l === 'JA' ? '日本語' : '英語'} AVG</div>
                  <LineChart
                    height={220}
                    xLabel="ステップ"
                    yLabel="AVG"
                    fmtX={compact}
                    fmtY={(v) => v.toFixed(3)}
                    series={lines.map(([s, rs]) => ({
                      id: s,
                      color: rs[0].color,
                      label: s,
                      points: rs.flatMap((r) => {
                        const y = ljEval(r, ver)?.avg?.[l]
                        return y != null ? [{ x: r.step!, y }] : []
                      }),
                    }))}
                  />
                </div>
              ))}
              <div className="flex flex-col gap-0.5 text-xs text-muted-foreground">
                {lines.map(([s, rs]) => (
                  <span key={s} className="inline-flex items-center gap-1.5">
                    <span className="size-2.5 rounded-full" style={{ background: rs[0].color }} />
                    <span className="truncate font-mono">{s}</span> ({rs.length} 点)
                  </span>
                ))}
              </div>
            </div>
          )}
        </Section>
      </div>

      {selected.some((r) => r.components['llm-jp-judge']?.status === 'done') && (
        <Section title="llm-jp-judge" description="score_table.json の全観点。ジャッジのモデルが違う実行どうしは比べられない">
          <AspectTable
            runs={selected}
            rows={(r) => Object.entries(r.components['llm-jp-judge']?.scores ?? {}).flatMap(([b, as]) => Object.entries(as).map(([a, v]) => [`${b}:${a}`, v] as [string, number]))}
            extra={(r) => r.components['llm-jp-judge']?.judge_model ?? ''}
            extraLabel="ジャッジ"
          />
        </Section>
      )}
      {selected.some((r) => r.components['safety-eval']?.status === 'done') && (
        <Section title="safety-eval" description="evaluate_count の数値をベンチマークごとに">
          <AspectTable
            runs={selected}
            rows={(r) =>
              Object.entries(r.components['safety-eval']?.benchmarks ?? {}).flatMap(([b, x]) =>
                Object.entries(x.metrics).flatMap(([k, v]) => (v == null ? [] : [[`${b}:${k}`, v] as [string, number]])),
              )
            }
          />
        </Section>
      )}
    </div>
  )
}

function CategoryTable({ runs, ver, diff }: { runs: SelectedRun[]; ver?: string; diff: boolean }) {
  const meta = useQuery(metaQuery())
  // 列: 言語 × カテゴリ (日本語を先に)。どの実行にも無い列は出さない
  const cols = useMemo(() => {
    const out: { lang: 'JA' | 'EN' | 'ALL'; cat: string }[] = []
    for (const lang of ['JA', 'EN', 'ALL'] as const) {
      const cats = new Set<string>()
      for (const r of runs) for (const c of Object.keys(ljEval(r, ver)?.categories?.[lang] ?? {})) cats.add(c)
      for (const c of cats) out.push({ lang, cat: c })
    }
    return out
  }, [runs, ver])
  const base = runs[0]
  const val = (r: SelectedRun, lang: string, cat: string) => ljEval(r, ver)?.categories?.[lang as 'JA']?.[cat]
  return (
    <div className="overflow-auto">
      <table className="text-sm">
        <thead>
          <tr className="text-xs text-muted-foreground">
            <th />
            <th className="px-2 py-1 text-right font-medium">AVG</th>
            {cols.map((c, i) => (
              <th
                key={`${c.lang}${c.cat}`}
                className={cn('px-1.5 py-1 text-center font-medium', i > 0 && cols[i - 1].lang !== c.lang && 'border-l')}
                title={meta.data?.category_desc[c.cat]}
              >
                <Link to="/datasets" search={(p) => ({ runs: p.runs, ver: p.ver, cat: c.cat, lang: c.lang === 'ALL' ? undefined : c.lang })} className="hover:underline">
                  <div className="text-[10px] font-normal">{c.lang === 'ALL' ? '' : c.lang === 'JA' ? '日' : '英'}</div>
                  {c.cat}
                </Link>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => {
            const avg = ljEval(r, ver)?.avg
            return (
              <tr key={r.id} className="border-t">
                <td className="max-w-[20rem] px-2 py-1">
                  <RunLabel run={r} color={r.color} compact />
                </td>
                <td className="px-2 py-1 text-right text-xs whitespace-nowrap tabular">
                  {avg?.JA != null ? `${avg.JA.toFixed(3)} / ${avg.EN?.toFixed(3) ?? '—'}` : (avg?.ALL?.toFixed(3) ?? '—')}
                </td>
                {cols.map((c, i) => {
                  const v = val(r, c.lang, c.cat)
                  const colVals = runs.map((x) => val(x, c.lang, c.cat)).filter((x): x is number => x != null)
                  const lo = Math.min(...colVals)
                  const hi = Math.max(...colVals)
                  let bg: string | undefined
                  let ink: string | undefined
                  let text = v == null ? '·' : v.toFixed(2)
                  if (v != null && diff) {
                    const b = val(base, c.lang, c.cat)
                    if (r.id === base.id) bg = undefined
                    else if (b != null) {
                      const d = divShade((v - b) / 0.1)
                      bg = d.bg
                      ink = d.ink
                      text = `${v - b >= 0 ? '+' : ''}${(v - b).toFixed(2)}`
                    }
                  } else if (v != null && hi > lo) {
                    const t = (v - lo) / (hi - lo)
                    bg = seqShade(0.15 + 0.7 * t)
                    ink = t > 0.75 ? '#fff' : undefined
                  }
                  return (
                    <td key={`${c.lang}${c.cat}`} className={cn('p-0.5', i > 0 && cols[i - 1].lang !== c.lang && 'border-l')}>
                      <div
                        className={cn('min-w-12 rounded px-1.5 py-0.5 text-center text-xs tabular', v == null && 'text-muted-foreground/50')}
                        style={{ background: bg, color: ink }}
                        title={v != null ? `${c.lang} ${c.cat}: ${v.toFixed(4)}` : undefined}
                      >
                        {text}
                      </div>
                    </td>
                  )
                })}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

/** 行 = 指標、列 = 実行の数値表 (judge・safety-eval の全観点) */
function AspectTable({ runs, rows, extra, extraLabel }: { runs: SelectedRun[]; rows: (r: SelectedRun) => [string, number][]; extra?: (r: SelectedRun) => string; extraLabel?: string }) {
  const per = runs.map((r) => new Map(rows(r)))
  const keys = [...new Set(per.flatMap((m) => [...m.keys()]))]
  return (
    <div className="max-h-[560px] overflow-auto">
      <table className="text-sm">
        <thead className="sticky top-0 bg-background">
          <tr className="text-xs text-muted-foreground">
            <th className="px-2 py-1 text-left font-medium">指標</th>
            {runs.map((r) => (
              <th key={r.id} className="w-36 max-w-36 min-w-36 px-2 py-1 align-bottom font-medium">
                <RunColumnHead run={r} all={runs}>
                  {extra && extra(r) && <div className="truncate text-[11px] font-normal">{`${extraLabel}: ${extra(r)}`}</div>}
                </RunColumnHead>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {keys.map((k) => (
            <tr key={k} className="border-t">
              <td className="px-2 py-1 font-mono text-xs">{k}</td>
              {per.map((m, i) => (
                <td key={runs[i].id} className="px-2 py-1 text-right tabular">
                  {m.get(k) == null ? <span className="text-muted-foreground/50">·</span> : Number(m.get(k)!.toFixed(4))}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
