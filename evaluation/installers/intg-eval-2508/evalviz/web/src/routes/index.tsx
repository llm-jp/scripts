import { useQuery } from '@tanstack/react-query'
import { createFileRoute, Link, useNavigate } from '@tanstack/react-router'
import { ArrowDown, ArrowUp, TriangleAlert } from 'lucide-react'
import { useMemo, useState } from 'react'
import { featuredQuery, metaQuery, runsQuery } from '@/api/queries'
import { RunBars } from '@/components/charts/RunBars'
import { PageHeader } from '@/components/layout/PageHeader'
import { Section } from '@/components/layout/Section'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { dateTime } from '@/lib/format'
import { BOARD_COLORS, type BoardEntry, boardEntries } from '@/lib/leaderboard'
import { type Headline, headlines } from '@/lib/metrics'
import { joinRunIds, ljEval } from '@/lib/runs'
import { seqShade } from '@/lib/shade'
import { cn } from '@/lib/utils'

interface BoardSearch {
  /** 表の並び替えに使う列の ID (既定は日本語 AVG) */
  sort?: string
  asc?: boolean
}

export const Route = createFileRoute('/')({
  validateSearch: (s: Record<string, unknown>): BoardSearch => ({
    sort: typeof s.sort === 'string' && s.sort ? s.sort : undefined,
    asc: s.asc === true || s.asc === 'true' ? true : undefined,
  }),
  component: BoardPage,
})

interface Column {
  id: string
  group: string
  label: string
  title: string
  digits: number
  lowerIsBetter?: boolean
  get: (e: BoardEntry) => number | null | undefined
}

function BoardPage() {
  const search = Route.useSearch()
  const navigate = useNavigate({ from: Route.fullPath })
  const featured = useQuery(featuredQuery())
  const runs = useQuery(runsQuery())
  const meta = useQuery(metaQuery())
  const byId = useMemo(() => new Map((runs.data ?? []).map((r) => [r.id, r])), [runs.data])
  const entries = useMemo(() => boardEntries(featured.data, byId), [featured.data, byId])
  const std = featured.data?.standard['llm-jp-eval']
  const ver = std?.version
  const hs = useMemo(() => headlines(entries, ver), [entries, ver])
  const [picked, setPicked] = useState<Set<string>>(new Set())

  // 表の列: 主要指標のあとに llm-jp-eval のカテゴリ (日本語 → 英語)
  const columns: Column[] = useMemo(() => {
    const cols: Column[] = hs.map((h: Headline) => ({
      id: h.id, group: h.source.replace(/ v[\d.]+$/, ''), label: h.title, title: h.description, digits: h.digits, lowerIsBetter: h.lowerIsBetter, get: (e) => h.get(e, ver),
    }))
    for (const lang of ['JA', 'EN'] as const) {
      const cats = new Set<string>()
      for (const e of entries) for (const c of Object.keys(ljEval(e, ver)?.categories?.[lang] ?? {})) cats.add(c)
      for (const c of cats)
        cols.push({
          id: `cat:${lang}:${c}`, group: `llm-jp-eval ${lang === 'JA' ? '日本語' : '英語'}のカテゴリ`, label: c, title: meta.data?.category_desc[c] ?? c, digits: 2,
          get: (e) => ljEval(e, ver)?.categories?.[lang]?.[c],
        })
    }
    return cols
  }, [hs, entries, ver, meta.data])

  const sortCol = columns.find((c) => c.id === (search.sort ?? 'lj-ja')) ?? columns[0]
  const rows = useMemo(() => {
    const out = [...entries]
    if (!sortCol) return out
    const dir = (search.asc ? 1 : -1) * (sortCol.lowerIsBetter ? -1 : 1)
    return out.sort((a, b) => {
      const x = sortCol.get(a)
      const y = sortCol.get(b)
      if (x == null) return y == null ? 0 : 1
      if (y == null) return -1
      return dir * (x - y)
    })
  }, [entries, sortCol, search.asc])

  const compare = (es: BoardEntry[]) =>
    navigate({ to: '/overview', search: { runs: joinRunIds(es.map((e) => e.compareId).filter((x): x is string => !!x)), ver } })

  if (featured.data?.error && !entries.length)
    return (
      <div className="m-4 rounded-lg border border-dashed p-8 text-sm text-muted-foreground">
        <p className="font-medium text-foreground">掲載モデルの設定を読めない</p>
        <p className="mt-1">{featured.data.error}</p>
        <p className="mt-3">
          書式は evalviz/featured.example.json を参照。設定が無くても
          <Link to="/runs" className="mx-1 text-foreground underline">
            実行を選んで
          </Link>
          比べられる。
        </p>
      </div>
    )

  const pickedEntries = rows.filter((e) => picked.has(e.id))
  const groupsOfCols = columns.reduce<{ group: string; n: number }[]>((acc, c) => {
    if (acc.length && acc[acc.length - 1].group === c.group) acc[acc.length - 1].n++
    else acc.push({ group: c.group, n: 1 })
    return acc
  }, [])
  const updated = entries.map((e) => e.finished_at ?? '').sort().pop()

  return (
    <div className="flex flex-col gap-4 p-4">
      <PageHeader
        title={featured.data?.title ?? 'モデルの評価結果'}
        description={
          <>
            主要なモデルを同じ条件で評価した結果。
            {ver && (
              <>
                llm-jp-eval は {ver}・データセットあたり {std?.max_num_samples == null ? '全件' : `${std.max_num_samples} 件`}の結果だけを載せる (条件の違う結果は値が比べられないため)。
              </>
            )}
            {updated && <> 最新の結果 {dateTime(updated)}。</>}
          </>
        }
      >
        <div className="flex gap-2">
          <Button size="sm" onClick={() => compare(pickedEntries.length ? pickedEntries : rows)} disabled={!entries.length}>
            {pickedEntries.length ? `選んだ ${pickedEntries.length} モデルを詳しく比べる` : '掲載モデルを詳しく比べる'}
          </Button>
          <Button asChild size="sm" variant="outline">
            <Link to="/runs">実行を選んで比べる</Link>
          </Button>
        </div>
      </PageHeader>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
        <span className="inline-flex items-center gap-1.5">
          <span className="size-2.5 rounded-full" style={{ background: BOARD_COLORS.own }} />
          LLM-jp のモデル
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="size-2.5 rounded-full" style={{ background: BOARD_COLORS.baseline }} />
          比較対象 (他組織のモデル)
        </span>
        <span>
          掲載モデルは <span className="font-mono">{featured.data?.path}</span> で管理する (gcg51557 のメンバーが編集でき、保存すると数分以内に反映される)
        </span>
      </div>

      {featured.isLoading || runs.isLoading ? (
        <p className="text-sm text-muted-foreground">読み込み中…</p>
      ) : (
        <>
          <div className="grid gap-4 xl:grid-cols-2 2xl:grid-cols-3">
            {/* 値のあるモデルが 1 つだけの指標は棒グラフにしない (表にだけ出す) */}
            {hs.filter((h) => entries.filter((e) => h.get(e, ver) != null).length >= 2).map((h) => (
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
                  data={entries.map((e) => {
                    const v = h.get(e, ver)
                    return { run: e, value: v, label: v == null ? '—' : v.toFixed(h.digits) }
                  })}
                />
              </Section>
            ))}
          </div>

          <Section
            title="一覧"
            description="列見出しのクリックで並べ替える。色の濃さは列の中での相対的な良さ。モデル名のクリックで、値を取った評価の実行へ。チェックしたモデルだけを「詳しく比べる」で総合比較・データセット別・誤りパターンに送れる"
          >
            <div className="overflow-auto">
              <table className="border-separate border-spacing-0 text-sm">
                <thead className="text-xs text-muted-foreground">
                  <tr>
                    <th colSpan={2} />
                    {groupsOfCols.map((g, i) => (
                      <th key={`${g.group}${i}`} colSpan={g.n} className="border-b border-l px-2 py-1 text-left font-normal">
                        {g.group}
                      </th>
                    ))}
                  </tr>
                  <tr>
                    <th className="w-8" />
                    <th className="min-w-[16rem] px-2 py-1 text-left font-medium">モデル</th>
                    {columns.map((c, i) => (
                      <th key={c.id} className={cn('px-1.5 py-1 text-center align-bottom font-medium', (i === 0 || columns[i - 1].group !== c.group) && 'border-l')} title={c.title}>
                        <button
                          type="button"
                          className={cn('inline-flex min-w-14 items-center justify-center gap-0.5 hover:text-foreground', c.label.length <= 12 && 'whitespace-nowrap', sortCol?.id === c.id && 'text-foreground')}
                          onClick={() => navigate({ search: (p) => ({ ...p, sort: c.id, asc: sortCol?.id === c.id && !search.asc ? true : undefined }), replace: true, resetScroll: false })}
                        >
                          {c.label}
                          {sortCol?.id === c.id && (search.asc ? <ArrowUp className="size-3" /> : <ArrowDown className="size-3" />)}
                        </button>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((e) => (
                    <tr key={e.id} className="group">
                      <td className="border-t px-2 py-1">
                        <Checkbox
                          checked={picked.has(e.id)}
                          onCheckedChange={() => setPicked((s) => new Set(s.has(e.id) ? [...s].filter((x) => x !== e.id) : [...s, e.id]))}
                          aria-label={`${e.name} を選ぶ`}
                        />
                      </td>
                      <td className="border-t px-2 py-1">
                        <div className="flex items-center gap-1.5">
                          <span className="size-2.5 shrink-0 rounded-full" style={{ background: e.color }} />
                          {e.compareId ? (
                            <Link to="/runs/$runId" params={{ runId: e.compareId }} className="font-medium hover:underline" title={e.model_path}>
                              {e.name}
                            </Link>
                          ) : (
                            <span className="font-medium">{e.name}</span>
                          )}
                        </div>
                        <div className="mt-0.5 flex flex-wrap gap-1 pl-4 text-[11px] text-muted-foreground">
                          {[e.featured.family, e.featured.params, e.featured.type].filter(Boolean).map((x) => (
                            <Badge key={x} variant="outline" className="h-4 px-1 text-[10px] font-normal">
                              {x}
                            </Badge>
                          ))}
                          {e.featured.note && <span>{e.featured.note}</span>}
                        </div>
                      </td>
                      {columns.map((c, i) => {
                        const v = c.get(e)
                        const vs = entries.map(c.get).filter((x): x is number => x != null)
                        const lo = Math.min(...vs)
                        const hi = Math.max(...vs)
                        const t = v == null || hi <= lo ? 0 : c.lowerIsBetter ? (hi - v) / (hi - lo) : (v - lo) / (hi - lo)
                        return (
                          <td key={c.id} className={cn('border-t p-0.5', (i === 0 || columns[i - 1].group !== c.group) && 'border-l')}>
                            <div
                              className={cn('min-w-12 rounded px-1.5 py-0.5 text-center text-xs tabular', v == null && 'text-muted-foreground/50')}
                              style={v != null && hi > lo ? { background: seqShade(0.15 + 0.7 * t), color: t > 0.75 ? '#fff' : undefined } : undefined}
                            >
                              {v == null ? '—' : v.toFixed(c.digits)}
                            </div>
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {(featured.data?.problems.length ?? 0) > 0 && (
            <div className="flex items-start gap-2 rounded-lg border border-[var(--status-warning)] bg-[var(--status-warning)]/10 px-3 py-2 text-sm">
              <TriangleAlert className="mt-0.5 size-4 shrink-0" />
              <div>
                掲載モデルの設定の問題 (該当する値は「—」になる):
                <ul className="mt-1 list-disc pl-5 text-xs text-muted-foreground">
                  {featured.data!.problems.map((p) => (
                    <li key={p}>{p}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
