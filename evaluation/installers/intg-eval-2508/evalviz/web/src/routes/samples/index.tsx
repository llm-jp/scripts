import { useQuery } from '@tanstack/react-query'
import { createFileRoute, Link, useNavigate } from '@tanstack/react-router'
import { useMemo, useState } from 'react'
import { metaQuery, samplesQuery } from '@/api/queries'
import type { SampleCell, SampleRow, Status } from '@/api/types'
import { StatusChip, StatusLegend } from '@/components/charts/StatusBar'
import { NoRunsSelected, PageHeader } from '@/components/layout/PageHeader'
import { RunColumnHead } from '@/components/runs/RunColumnHead'
import { VersionSelect } from '@/components/runs/VersionSelect'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { useSelectedRuns } from '@/hooks/useSelectedRuns'
import { useVersion } from '@/hooks/useVersion'
import { pct } from '@/lib/format'
import { ljEval } from '@/lib/runs'
import { FLAG_SHORT, STATUS_ORDER, STATUSES } from '@/lib/status'

type SortKey = 'index' | 'hard' | 'easy'

export interface SamplesSearch {
  ds?: string
  q?: string
  /** 結果が割れた行だけ */
  diff?: boolean
  st?: Status
  fl?: string
  sort?: SortKey
}

export const Route = createFileRoute('/samples/')({
  validateSearch: (s: Record<string, unknown>): SamplesSearch => ({
    ds: typeof s.ds === 'string' && s.ds ? s.ds : undefined,
    q: typeof s.q === 'string' && s.q ? s.q : undefined,
    diff: s.diff === true || s.diff === 'true' ? true : undefined,
    st: typeof s.st === 'string' && s.st in STATUSES ? (s.st as Status) : undefined,
    fl: typeof s.fl === 'string' && s.fl ? s.fl : undefined,
    sort: s.sort === 'hard' || s.sort === 'easy' ? s.sort : undefined,
  }),
  component: SamplesPage,
})

const PAGE = 300

/** 結果が割れている: 正誤が出るなら正解と非正解が混在、出ないなら状態か抽出した回答が実行で異なる */
function isSplit(row: SampleRow): boolean {
  const cs = row.cells.filter((c): c is SampleCell => !!c)
  if (cs.length < 2) return false
  const st = new Set(cs.map((c) => c.st))
  if (st.has('correct')) return st.size > 1
  return st.size > 1 || new Set(cs.map((c) => c.p)).size > 1
}

function nCorrect(row: SampleRow): number {
  return row.cells.filter((c) => c?.st === 'correct').length
}

function SamplesPage() {
  const search = Route.useSearch()
  const navigate = useNavigate({ from: Route.fullPath })
  const set = (patch: Partial<SamplesSearch>) => navigate({ search: (p) => ({ ...p, ...patch }), replace: true, resetScroll: false })
  const { ids, selected } = useSelectedRuns()
  const { ver, versions, setVer } = useVersion(selected)
  const meta = useQuery(metaQuery())
  const [limit, setLimit] = useState(PAGE)

  const datasets = useMemo(() => {
    const m = new Map<string, { lang: string; cat: string }>()
    for (const r of selected) for (const [ds, d] of Object.entries(ljEval(r, ver)?.datasets ?? {})) m.set(ds, { lang: d.lang, cat: d.category ?? '' })
    return [...m.entries()].sort((a, b) => b[1].lang.localeCompare(a[1].lang) || a[0].localeCompare(b[0]))
  }, [selected, ver])
  const ds = search.ds ?? datasets[0]?.[0]
  const samples = useQuery(samplesQuery(ids, ver, ds))
  const rows = useMemo(() => samples.data?.rows ?? [], [samples.data])
  const info = samples.data?.info

  const shown = useMemo(() => {
    const words = (search.q ?? '').toLowerCase().split(/\s+/).filter(Boolean)
    const out = rows.filter(
      (r) =>
        (!search.diff || isSplit(r)) &&
        (!search.st || r.cells.some((c) => c?.st === search.st)) &&
        (!search.fl || r.cells.some((c) => c?.fl.includes(search.fl!))) &&
        words.every((w) => (r.input + '\n' + r.gold + '\n' + r.cells.map((c) => c?.p ?? '').join('\n')).toLowerCase().includes(w)),
    )
    if (search.sort === 'hard') out.sort((a, b) => nCorrect(a) - nCorrect(b) || a.index - b.index)
    else if (search.sort === 'easy') out.sort((a, b) => nCorrect(b) - nCorrect(a) || a.index - b.index)
    return out
  }, [rows, search])

  if (!ids.length) return <NoRunsSelected what="サンプル比較" />

  // samplesQuery の cells はリクエストした ids の順。選択中だが存在しない実行は useSelectedRuns が落とすので id で引く
  const col = ids.map((id) => selected.find((r) => r.id === id))
  const colStats = ids.map((_, i) => {
    const cs = shown.map((r) => r.cells[i]).filter((c): c is SampleCell => !!c)
    return { n: cs.length, ok: cs.filter((c) => c.st === 'correct').length }
  })

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 p-4">
      <PageHeader
        title="サンプル比較"
        description="行 = サンプル、列 = 実行。セルは状態の記号・抽出した回答・誤りの兆候の印。行のクリックで、全実行の生成 (推論・最終出力) を並べて見る。"
      >
        <VersionSelect ver={ver} versions={versions} onChange={setVer} />
      </PageHeader>
      <div className="flex flex-wrap items-center gap-3">
        <Select value={ds} onValueChange={(v) => { setLimit(PAGE); set({ ds: v }) }}>
          <SelectTrigger size="sm" className="w-64">
            <SelectValue placeholder="データセット" />
          </SelectTrigger>
          <SelectContent>
            {datasets.map(([d, x]) => (
              <SelectItem key={d} value={d}>
                <span className="font-mono">{d}</span>
                <span className="text-muted-foreground">
                  {x.lang === 'JA' ? '日' : '英'} · {x.cat}
                </span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Input className="h-8 w-56" placeholder="入力・正解・回答の文字列" value={search.q ?? ''} onChange={(e) => set({ q: e.target.value || undefined })} />
        <Select value={search.st ?? '__all'} onValueChange={(v) => set({ st: v === '__all' ? undefined : (v as Status) })}>
          <SelectTrigger size="sm" className="w-44">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="__all">状態: すべて</SelectItem>
            {STATUS_ORDER.map((s) => (
              <SelectItem key={s} value={s}>
                どれかが {STATUSES[s].symbol} {STATUSES[s].label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Select value={search.fl ?? '__all'} onValueChange={(v) => set({ fl: v === '__all' ? undefined : v })}>
          <SelectTrigger size="sm" className="w-48">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="__all">印: すべて</SelectItem>
            {(meta.data?.flags ?? []).map((f) => (
              <SelectItem key={f.key} value={f.key}>
                どれかに「{f.label}」
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Select value={search.sort ?? 'index'} onValueChange={(v) => set({ sort: v === 'index' ? undefined : (v as SortKey) })}>
          <SelectTrigger size="sm" className="w-44">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="index">並び: 出題順</SelectItem>
            <SelectItem value="hard">並び: 正解が少ない順</SelectItem>
            <SelectItem value="easy">並び: 正解が多い順</SelectItem>
          </SelectContent>
        </Select>
        <div className="flex items-center gap-2 text-sm">
          <Switch id="diff" checked={!!search.diff} onCheckedChange={(v) => set({ diff: v || undefined })} />
          <Label htmlFor="diff">結果が割れた行だけ</Label>
        </div>
        <span className="ml-auto text-sm text-muted-foreground tabular">
          {shown.length} / {rows.length} サンプル
        </span>
      </div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
        <StatusLegend />
        {info && (
          <span>
            指標 {info.metrics.join(', ')} · 回答形式 {info.pattern}
            {info.labels.length > 0 && ` · 選択肢 ${info.labels.join(' / ')}`}
            {info.judge == null ? ' · 正誤なし (連続値の指標)' : info.approx ? ' · 正誤は近似' : ''}
          </span>
        )}
      </div>

      {samples.isFetching && !samples.data && <p className="text-sm text-muted-foreground">推論結果を集計中…</p>}
      {samples.error && <p className="text-sm text-[var(--status-critical)]">取得に失敗: {String(samples.error)}</p>}
      <div className="min-h-0 flex-1 overflow-auto rounded-lg border">
        <table className="border-separate border-spacing-0 text-sm">
          <thead className="sticky top-0 z-20 bg-background">
            <tr>
              <th className="sticky left-0 z-30 min-w-[24rem] border-b bg-background px-3 py-2 text-left align-bottom font-medium">入力 / 正解</th>
              {ids.map((id, i) => {
                const r = col[i]
                return (
                  <th key={id} className="w-44 max-w-44 min-w-44 border-b border-l px-2 py-2 text-left align-bottom font-normal">
                    {r ? (
                      <RunColumnHead run={r} all={selected} />
                    ) : (
                      <span className="text-xs text-muted-foreground">{id}</span>
                    )}
                    {info?.judge && (
                      <div className="mt-0.5 text-xs tabular">
                        {colStats[i].ok}/{colStats[i].n} <span className="text-muted-foreground">({pct(colStats[i].n ? colStats[i].ok / colStats[i].n : null)})</span>
                      </div>
                    )}
                  </th>
                )
              })}
            </tr>
          </thead>
          <tbody>
            {shown.slice(0, limit).map((row) => (
              <tr key={row.key} className="group cursor-pointer" onClick={() => navigate({ to: '/samples/$ds/$key', params: { ds: ds!, key: row.key } })}>
                <td className="sticky left-0 z-10 max-w-[32rem] border-b bg-background px-3 py-1 group-hover:bg-muted">
                  <Link to="/samples/$ds/$key" params={{ ds: ds!, key: row.key }} className="block truncate text-xs hover:underline" title={row.input}>
                    <span className="mr-1.5 text-muted-foreground tabular">#{row.index}</span>
                    {row.input}
                  </Link>
                  <div className="truncate text-[11px] text-muted-foreground" title={row.gold}>
                    正解: <span className="font-mono text-foreground">{row.gold}</span>
                  </div>
                </td>
                {row.cells.map((c, i) => (
                  <td key={ids[i]} className="max-w-44 border-b border-l px-2 py-1 group-hover:bg-muted/60">
                    {c ? (
                      <div className="flex min-w-0 items-center gap-1.5 text-xs">
                        <StatusChip status={c.st} />
                        <span className="min-w-0 truncate font-mono" title={c.p}>
                          {c.p || <span className="text-muted-foreground">(なし)</span>}
                        </span>
                        {c.fl.length > 0 && (
                          <span className="ml-auto shrink-0 rounded border px-1 text-[10px] text-muted-foreground" title={c.fl.map((f) => meta.data?.flags.find((x) => x.key === f)?.label ?? f).join(', ')}>
                            {c.fl.map((f) => FLAG_SHORT[f] ?? f).join('·')}
                          </span>
                        )}
                      </div>
                    ) : (
                      <span className="text-xs text-muted-foreground/50">—</span>
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {shown.length > limit && (
          <div className="p-3 text-center">
            <Button size="sm" variant="outline" onClick={() => setLimit(limit + PAGE)}>
              さらに {Math.min(PAGE, shown.length - limit)} 件 (残り {shown.length - limit})
            </Button>
          </div>
        )}
      </div>
    </div>
  )
}
