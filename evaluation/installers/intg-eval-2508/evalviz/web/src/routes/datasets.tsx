import { useQuery } from '@tanstack/react-query'
import { createFileRoute, Link, useNavigate } from '@tanstack/react-router'
import { useMemo } from 'react'
import { metaQuery } from '@/api/queries'
import { NoRunsSelected, PageHeader } from '@/components/layout/PageHeader'
import { RunColumnHead } from '@/components/runs/RunColumnHead'
import { VersionSelect } from '@/components/runs/VersionSelect'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { useSelectedRuns } from '@/hooks/useSelectedRuns'
import { useVersion } from '@/hooks/useVersion'
import { ljEval, runTitle } from '@/lib/runs'
import { divShade, seqShade } from '@/lib/shade'
import { cn } from '@/lib/utils'

type SortKey = 'cat' | 'spread' | 'gain' | 'loss'

interface DatasetsSearch {
  cat?: string
  lang?: 'JA' | 'EN'
  sort?: SortKey
  diff?: boolean
}

export const Route = createFileRoute('/datasets')({
  validateSearch: (s: Record<string, unknown>): DatasetsSearch => ({
    cat: typeof s.cat === 'string' && s.cat ? s.cat : undefined,
    lang: s.lang === 'JA' || s.lang === 'EN' ? s.lang : undefined,
    sort: s.sort === 'spread' || s.sort === 'gain' || s.sort === 'loss' ? s.sort : undefined,
    diff: s.diff === true || s.diff === 'true' ? true : undefined,
  }),
  component: DatasetsPage,
})

interface Row {
  ds: string
  lang: 'JA' | 'EN'
  cat: string
  metric: string
  vals: (number | null)[]
  ools: (number | null)[]
  spread: number
  /** 基準 (先頭の実行) との差の最大・最小 */
  maxDiff: number
  minDiff: number
}

function DatasetsPage() {
  const search = Route.useSearch()
  const navigate = useNavigate({ from: Route.fullPath })
  const set = (patch: Partial<DatasetsSearch>) => navigate({ search: (p) => ({ ...p, ...patch }), replace: true, resetScroll: false })
  const { ids, selected } = useSelectedRuns()
  const { ver, versions, setVer } = useVersion(selected)
  const meta = useQuery(metaQuery())

  const rows: Row[] = useMemo(() => {
    const m = new Map<string, Row>()
    selected.forEach((r, i) => {
      for (const [ds, d] of Object.entries(ljEval(r, ver)?.datasets ?? {})) {
        let row = m.get(ds)
        if (!row) {
          row = { ds, lang: d.lang, cat: d.category ?? '(なし)', metric: d.metric, vals: selected.map(() => null), ools: selected.map(() => null), spread: 0, maxDiff: 0, minDiff: 0 }
          m.set(ds, row)
        }
        row.vals[i] = d.score
        row.ools[i] = d.ool
      }
    })
    for (const row of m.values()) {
      const vs = row.vals.filter((v): v is number => v != null)
      row.spread = vs.length > 1 ? Math.max(...vs) - Math.min(...vs) : 0
      const b = row.vals[0]
      const ds = b == null ? [] : row.vals.slice(1).filter((v): v is number => v != null).map((v) => v - b)
      row.maxDiff = ds.length ? Math.max(...ds) : 0
      row.minDiff = ds.length ? Math.min(...ds) : 0
    }
    return [...m.values()]
  }, [selected, ver])

  const cats = useMemo(() => [...new Set(rows.map((r) => r.cat))].sort(), [rows])
  const shown = useMemo(() => {
    const out = rows.filter((r) => (!search.cat || r.cat === search.cat) && (!search.lang || r.lang === search.lang))
    const byCat = (a: Row, b: Row) => a.lang.localeCompare(b.lang) * -1 || a.cat.localeCompare(b.cat) || a.ds.localeCompare(b.ds)
    if (search.sort === 'spread') out.sort((a, b) => b.spread - a.spread || byCat(a, b))
    else if (search.sort === 'gain') out.sort((a, b) => b.maxDiff - a.maxDiff || byCat(a, b))
    else if (search.sort === 'loss') out.sort((a, b) => a.minDiff - b.minDiff || byCat(a, b))
    else out.sort(byCat)
    return out
  }, [rows, search])

  if (!ids.length) return <NoRunsSelected what="データセット別の比較" />

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 p-4">
      <PageHeader
        title="データセット別"
        description="行 = データセット、列 = 実行。値はカテゴリの集計に使う主指標 (指標名は行見出し)。色は行の中での相対的な高さ。⚠ は選択肢外・抽出不可の率 (ool) が 5% 以上。セルのクリックでサンプル比較へ"
      >
        <VersionSelect ver={ver} versions={versions} onChange={setVer} />
      </PageHeader>
      <div className="flex flex-wrap items-center gap-3">
        <ToggleGroup type="single" size="sm" variant="outline" value={search.lang ?? 'all'} onValueChange={(v) => v && set({ lang: v === 'all' ? undefined : (v as 'JA' | 'EN') })}>
          <ToggleGroupItem value="all">全言語</ToggleGroupItem>
          <ToggleGroupItem value="JA">日本語</ToggleGroupItem>
          <ToggleGroupItem value="EN">英語</ToggleGroupItem>
        </ToggleGroup>
        <Select value={search.cat ?? '__all'} onValueChange={(v) => set({ cat: v === '__all' ? undefined : v })}>
          <SelectTrigger size="sm" className="w-56">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="__all">カテゴリ: すべて</SelectItem>
            {cats.map((c) => (
              <SelectItem key={c} value={c}>
                {c} {meta.data?.category_desc[c] && <span className="text-muted-foreground">{meta.data.category_desc[c]}</span>}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Select value={search.sort ?? 'cat'} onValueChange={(v) => set({ sort: v === 'cat' ? undefined : (v as SortKey) })}>
          <SelectTrigger size="sm" className="w-64">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="cat">並び: 言語 → カテゴリ</SelectItem>
            <SelectItem value="spread">並び: 実行間の差が大きい順</SelectItem>
            <SelectItem value="gain">並び: 基準より上がった順</SelectItem>
            <SelectItem value="loss">並び: 基準より下がった順</SelectItem>
          </SelectContent>
        </Select>
        <div className="flex items-center gap-2 text-sm">
          <ToggleGroup type="single" size="sm" variant="outline" value={search.diff ? 'diff' : 'value'} onValueChange={(v) => v && set({ diff: v === 'diff' || undefined })}>
            <ToggleGroupItem value="value">値</ToggleGroupItem>
            <ToggleGroupItem value="diff">基準との差</ToggleGroupItem>
          </ToggleGroup>
          <Label className="text-xs text-muted-foreground">基準 = 先頭の実行 ({selected[0] && runTitle(selected[0])})</Label>
        </div>
        <span className="ml-auto text-sm text-muted-foreground tabular">
          {shown.length} / {rows.length} データセット
        </span>
      </div>

      <div className="min-h-0 flex-1 overflow-auto rounded-lg border">
        <table className="border-separate border-spacing-0 text-sm">
          <thead className="sticky top-0 z-20 bg-background">
            <tr>
              <th className="sticky left-0 z-30 min-w-[16rem] border-b bg-background px-3 py-2 text-left align-bottom font-medium">データセット</th>
              <th className="border-b px-2 py-2 text-right align-bottom text-xs font-medium text-muted-foreground">差 (最大-最小)</th>
              {selected.map((r) => (
                <th key={r.id} className="w-32 max-w-32 min-w-32 border-b border-l px-2 py-2 align-bottom font-normal">
                  <RunColumnHead run={r} all={selected} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((row) => {
              const vs = row.vals.filter((v): v is number => v != null)
              const lo = Math.min(...vs)
              const hi = Math.max(...vs)
              return (
                <tr key={row.ds} className="group">
                  <td className="sticky left-0 z-10 border-b bg-background px-3 py-1 group-hover:bg-muted">
                    <Link to="/samples" search={(p) => ({ runs: p.runs, ver: p.ver, ds: row.ds })} className="font-mono text-xs hover:underline">
                      {row.ds}
                    </Link>
                    <div className="text-[11px] text-muted-foreground">
                      {row.lang === 'JA' ? '日' : '英'} · {row.cat} · {row.metric}
                    </div>
                  </td>
                  <td className="border-b px-2 text-right text-xs text-muted-foreground tabular group-hover:bg-muted">{row.spread.toFixed(3)}</td>
                  {row.vals.map((v, i) => {
                    let bg: string | undefined
                    let ink: string | undefined
                    let text = v == null ? '' : v.toFixed(3)
                    if (v != null && search.diff && i > 0 && row.vals[0] != null) {
                      const d = v - row.vals[0]
                      ;({ bg, ink } = divShade(d / 0.1))
                      text = `${d >= 0 ? '+' : ''}${d.toFixed(3)}`
                    } else if (v != null && !search.diff && hi > lo) {
                      const t = (v - lo) / (hi - lo)
                      bg = seqShade(0.15 + 0.7 * t)
                      ink = t > 0.75 ? '#fff' : undefined
                    }
                    const ool = row.ools[i]
                    return (
                      <td key={selected[i].id} className="border-b border-l p-0.5 group-hover:bg-muted/60">
                        {v == null ? (
                          <div className="h-full bg-muted/30" />
                        ) : (
                          <Link
                            to="/samples"
                            search={(p) => ({ runs: p.runs, ver: p.ver, ds: row.ds })}
                            className={cn('flex items-center justify-end gap-1 rounded px-1.5 py-0.5 text-xs tabular hover:ring-2 hover:ring-foreground/30')}
                            style={{ background: bg, color: ink }}
                            title={`${row.metric}: ${v.toFixed(4)}${ool != null ? `\n選択肢外・抽出不可 (ool): ${(ool * 100).toFixed(1)}%` : ''}`}
                          >
                            {ool != null && ool >= 0.05 && <span className="text-[10px]">⚠</span>}
                            {text}
                          </Link>
                        )}
                      </td>
                    )
                  })}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
