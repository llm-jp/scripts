import { createFileRoute, Link, useNavigate } from '@tanstack/react-router'
import { ChevronRight } from 'lucide-react'
import { useMemo, useState } from 'react'
import type { Run } from '@/api/types'
import { PageHeader } from '@/components/layout/PageHeader'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { useSelectedRuns } from '@/hooks/useSelectedRuns'
import { dateTime } from '@/lib/format'
import { compareVersion, ljVersions, seriesColor } from '@/lib/runs'
import { cn } from '@/lib/utils'

interface RunsSearch {
  q?: string
  /** 結果の無い実行 (未完了・失敗した投入) も表示する */
  all?: boolean
}

export const Route = createFileRoute('/runs/')({
  validateSearch: (s: Record<string, unknown>): RunsSearch => ({
    q: typeof s.q === 'string' && s.q ? s.q : undefined,
    all: s.all === true || s.all === 'true' ? true : undefined,
  }),
  component: RunsPage,
})

function hasResult(r: Run): boolean {
  const c = r.components
  return ljVersions(r).length > 0 || [c.swallow, c['llm-jp-judge'], c['safety-eval']].some((x) => x?.status === 'done')
}

const f3 = (v: number | null | undefined) => (v == null ? '—' : v.toFixed(3))

function RunsPage() {
  const search = Route.useSearch()
  const navigate = useNavigate({ from: Route.fullPath })
  const set = (patch: Partial<RunsSearch>) => navigate({ search: (p) => ({ ...p, ...patch }), replace: true, resetScroll: false })
  const { ids, allRuns, toggle, setIds, isLoading } = useSelectedRuns()

  const visible = useMemo(() => {
    const words = (search.q ?? '').toLowerCase().split(/\s+/).filter(Boolean)
    return allRuns.filter(
      (r) =>
        (search.all || hasResult(r) || ids.includes(r.id)) &&
        words.every((w) => [r.name, r.model, r.model_path, r.group].some((x) => x.toLowerCase().includes(w))),
    )
  }, [allRuns, search, ids])
  const nHidden = useMemo(() => allRuns.filter((r) => !hasResult(r)).length, [allRuns])
  // グループは既定で畳む。絞り込み中、選択中の実行を含むグループ、グループが 1 つだけのときは開く
  const [toggled, setToggled] = useState<Set<string>>(new Set())
  const flip = (g: string) => setToggled((s) => new Set(s.has(g) ? [...s].filter((x) => x !== g) : [...s, g]))

  // 出力ルート配下のディレクトリごとにまとめ、新しい実行を上に
  const groups = useMemo(() => {
    const m = new Map<string, Run[]>()
    for (const r of visible) m.set(r.group, [...(m.get(r.group) ?? []), r])
    for (const rs of m.values()) rs.sort((a, b) => (b.finished_at ?? '').localeCompare(a.finished_at ?? '') || a.name.localeCompare(b.name))
    return [...m.entries()].sort((a, b) => a[0].localeCompare(b[0]))
  }, [visible])

  return (
    <div className="flex flex-col gap-4 p-4">
      <PageHeader
        title="実行を選んで比べる"
        description="主要なモデルの比較はトップページ (リーダーボード) にある。ここでは評価の出力 (qsub.py の出力ディレクトリ 1 つ = 1 モデル × 1 回の投入) を自分で選んで比べる。実験ディレクトリをクリックして開き、比べる実行にチェックを付ける。チェックした順に系列色が付く。"
      >
        {ids.length > 0 && (
          <div className="flex gap-2">
            <Button asChild size="sm">
              <Link to="/overview">総合比較へ ({ids.length})</Link>
            </Button>
            <Button asChild size="sm" variant="outline">
              <Link to="/errors">誤りパターンへ</Link>
            </Button>
          </div>
        )}
      </PageHeader>

      <div className="flex flex-wrap items-center gap-3">
        <Input
          className="h-8 w-96"
          placeholder="実行名・モデル・ディレクトリで絞り込み (空白区切りで AND)"
          value={search.q ?? ''}
          onChange={(e) => set({ q: e.target.value || undefined })}
        />
        <div className="flex items-center gap-2 text-sm">
          <Switch id="all" checked={!!search.all} onCheckedChange={(v) => set({ all: v || undefined })} />
          <Label htmlFor="all" title="どの評価の結果も無い実行 (実行中・失敗した投入)">
            結果の無い実行も表示 <span className="text-muted-foreground tabular">({nHidden})</span>
          </Label>
        </div>
        <span className="ml-auto text-sm text-muted-foreground tabular">
          {visible.length} / {allRuns.length} 実行
        </span>
      </div>

      {isLoading && <p className="text-sm text-muted-foreground">読み込み中…</p>}
      {groups.map(([group, runs]) => {
        const allSel = runs.every((r) => ids.includes(r.id))
        const nSel = runs.filter((r) => ids.includes(r.id)).length
        const openByDefault = !!search.q || nSel > 0 || groups.length === 1
        const open = openByDefault !== toggled.has(group)
        const vers = [...new Set(runs.flatMap(ljVersions))].sort(compareVersion)
        const latest = runs.map((r) => r.finished_at ?? '').sort().pop()
        return (
          <section key={group} className="rounded-lg border">
            <div className={cn('flex items-center gap-3 bg-muted/40 px-3 py-2', open && 'border-b')}>
              <button type="button" onClick={() => flip(group)} className="flex min-w-0 flex-1 items-center gap-2 text-left" aria-expanded={open}>
                <ChevronRight className={cn('size-4 shrink-0 transition-transform', open && 'rotate-90')} />
                <h2 className="truncate font-mono text-sm font-semibold">{group}</h2>
                <span className="shrink-0 text-sm text-muted-foreground tabular">
                  {runs.length} 件{nSel > 0 && ` (${nSel} 件を選択中)`}
                </span>
                <span className="ml-2 hidden shrink-0 gap-1 md:flex">
                  {vers.map((v) => (
                    <Badge key={v} variant="outline" className="font-normal">
                      llm-jp-eval {v}
                    </Badge>
                  ))}
                </span>
                <span className="ml-auto shrink-0 text-xs text-muted-foreground tabular">最終 {dateTime(latest || null)}</span>
              </button>
              <Button
                size="sm"
                variant="ghost"
                className="ml-auto"
                onClick={() => setIds(allSel ? ids.filter((x) => !runs.some((r) => r.id === x)) : [...ids, ...runs.map((r) => r.id).filter((x) => !ids.includes(x))])}
              >
                {allSel ? 'このグループを外す' : 'すべて選択'}
              </Button>
            </div>
            {open && (
            <table className="w-full table-fixed text-sm">
              <colgroup>
                <col className="w-9" />
                <col />
                <col className="w-[30%]" />
                <col className="w-[9rem]" />
                <col className="w-[9rem]" />
                <col className="w-[9.5rem]" />
              </colgroup>
              <thead className="text-xs text-muted-foreground">
                <tr className="border-b">
                  <th />
                  <th className="px-2 py-1.5 text-left font-medium">実行 (モデル)</th>
                  <th className="px-2 py-1.5 text-left font-medium">llm-jp-eval (日本語 / 英語)</th>
                  <th className="px-2 py-1.5 text-left font-medium">他の評価</th>
                  <th className="px-2 py-1.5 text-left font-medium">条件</th>
                  <th className="px-2 py-1.5 text-left font-medium">完了</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => {
                  const sel = ids.indexOf(r.id)
                  const lj = Object.entries(r.components['llm-jp-eval'] ?? {}).sort((a, b) => compareVersion(b[0], a[0]))
                  return (
                    <tr key={r.id} className={cn('border-b align-top last:border-0 hover:bg-muted/40', !hasResult(r) && 'text-muted-foreground')}>
                      <td className="px-2 py-1.5">
                        <Checkbox checked={sel >= 0} onCheckedChange={() => toggle(r.id)} aria-label={`${r.name} を選択`} />
                      </td>
                      <td className="min-w-0 px-2 py-1.5">
                        <span className="flex min-w-0 items-center gap-1.5">
                          {sel >= 0 && <span className="size-2.5 shrink-0 rounded-full" style={{ background: seriesColor(sel) }} />}
                          <Link to="/runs/$runId" params={{ runId: r.id }} className="truncate font-medium hover:underline" title={r.dir}>
                            {r.name}
                          </Link>
                          {r.step != null && <Badge variant="outline" className="tabular">step {r.step}</Badge>}
                        </span>
                        <div className="truncate text-xs text-muted-foreground" title={r.model_path}>
                          {r.model}
                        </div>
                      </td>
                      <td className="px-2 py-1.5">
                        {lj.length === 0 && <span className="text-muted-foreground">—</span>}
                        {lj.map(([v, c]) => (
                          <div key={v} className="flex items-baseline gap-2 tabular">
                            <span className="w-12 shrink-0 text-xs text-muted-foreground">{v}</span>
                            {c.status === 'done' ? (
                              <>
                                <span className="font-medium">{f3(c.avg?.JA ?? c.avg?.ALL)}</span>
                                <span className="text-muted-foreground">/</span>
                                <span className="font-medium">{f3(c.avg?.EN)}</span>
                                {c.n_samples != null && <span className="text-xs text-muted-foreground">{c.n_samples === -1 ? '全件' : `${c.n_samples} 件`}</span>}
                              </>
                            ) : (
                              <span className="text-xs text-muted-foreground">{c.status === 'missing' ? '結果なし' : '読めない'}</span>
                            )}
                          </div>
                        ))}
                      </td>
                      <td className="px-2 py-1.5">
                        <div className="flex flex-wrap gap-1">
                          {r.components.swallow?.status === 'done' && (
                            <Badge variant="secondary" className="tabular" title="swallow 英語平均">
                              swallow {f3(r.components.swallow.avg?.EN)}
                            </Badge>
                          )}
                          {r.components['llm-jp-judge']?.status === 'done' && (
                            <Badge variant="secondary" className="tabular" title="llm-jp-judge quality_ja 総合評価">
                              judge {r.components['llm-jp-judge'].scores?.quality_ja?.['総合評価']?.toFixed(2) ?? '✓'}
                            </Badge>
                          )}
                          {r.components['safety-eval']?.status === 'done' && <Badge variant="secondary">safety</Badge>}
                        </div>
                      </td>
                      <td className="px-2 py-1.5 text-xs text-muted-foreground">
                        {r.settings.basemodel && <div>basemodel</div>}
                        {r.settings.apply_chat_template && <div>chat template</div>}
                        {r.settings.reasoning_parser && <div>{String(r.settings.reasoning_parser)}</div>}
                        {r.settings.chat_template_args && <div className="truncate">{String(r.settings.chat_template_args)}</div>}
                      </td>
                      <td className="px-2 py-1.5 whitespace-nowrap text-muted-foreground tabular">{dateTime(r.finished_at)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
            )}
          </section>
        )
      })}
    </div>
  )
}
