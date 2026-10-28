import { RunLabel } from '@/components/runs/RunLabel'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import type { SelectedRun } from '@/hooks/useSelectedRuns'
import { fmtRate, seqShade } from '@/lib/shade'

export interface RunBarDatum {
  run: SelectedRun
  value: number | null | undefined
  label: string
  detail?: string
}

/**
 * 実行ごとの値の横棒 (1 指標 = 1 チャート)。色は実行の系列色、値は棒の右に直接書く。max は軸の上限。
 * sort で値の良い順に並べる (色は実行に付いているので並べ替えても塗り替わらない)。値の無い実行は末尾に「—」
 */
export function RunBars({ data, max = 1, sort, lowerIsBetter }: { data: RunBarDatum[]; max?: number; sort?: boolean; lowerIsBetter?: boolean }) {
  const rows = sort
    ? [...data].sort((a, b) => (a.value == null ? 1 : b.value == null ? -1 : lowerIsBetter ? a.value - b.value : b.value - a.value))
    : data
  return (
    <div className="grid grid-cols-[minmax(8rem,17rem)_minmax(7rem,1fr)_4.5rem] items-center gap-x-3 gap-y-1.5 text-sm">
      {rows.map((d) => (
        <div key={d.run.id} className="contents">
          <RunLabel run={d.run} color={d.run.color} compact />
          <div className="relative h-4">
            <div className="absolute inset-y-0 left-0 w-full rounded-[4px] bg-[var(--chart-grid)]/40" />
            {d.value != null && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <div
                    className="absolute inset-y-0 left-0 rounded-r-[4px]"
                    style={{ width: `${Math.max(0.5, Math.min(100, (d.value / max) * 100))}%`, background: d.run.color }}
                  />
                </TooltipTrigger>
                <TooltipContent>
                  {d.run.model}
                  {d.run.note && ` · ${d.run.note}`}
                  <br />
                  {d.label}
                  {d.detail && ` (${d.detail})`}
                </TooltipContent>
              </Tooltip>
            )}
          </div>
          <div className="text-right tabular">
            {d.value == null ? <span className="text-muted-foreground">—</span> : d.label}
            {d.detail && <span className="ml-1 text-xs text-muted-foreground">{d.detail}</span>}
          </div>
        </div>
      ))}
    </div>
  )
}

/** 行 = 実行、列 = カテゴリの件数表。セルの濃さは件数 (単色の濃淡)。クリックで絞り込み先へ */
export function CountMatrix({
  runs,
  columns,
  count,
  onCell,
  total,
  asRate,
}: {
  runs: SelectedRun[]
  columns: { key: string; label: string; title?: string }[]
  count: (runId: string, key: string) => number
  onCell?: (runId: string, key: string) => void
  total?: (runId: string) => number
  /** 件数の代わりに total に対する割合を出す (濃さも割合で決める) */
  asRate?: boolean
}) {
  const val = (rid: string, k: string) => (asRate && total ? count(rid, k) / Math.max(1, total(rid)) : count(rid, k))
  const max = Math.max(asRate ? 1e-9 : 1, ...runs.flatMap((r) => columns.map((c) => val(r.id, c.key))))
  return (
    <div className="overflow-auto">
      <table className="text-sm">
        <thead>
          <tr className="text-xs text-muted-foreground">
            <th className="px-2 py-1 text-left font-medium">実行</th>
            {total && <th className="px-2 py-1 text-right font-medium">計</th>}
            {columns.map((c) => (
              <th key={c.key} className="max-w-32 px-2 py-1 text-right align-bottom font-medium" title={c.title ?? c.key}>
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.id} className="border-t">
              <td className="max-w-[20rem] px-2 py-1">
                <RunLabel run={r} color={r.color} compact />
              </td>
              {total && <td className="px-2 text-right text-muted-foreground tabular">{total(r.id)}</td>}
              {columns.map((c) => {
                const n = count(r.id, c.key)
                const v = val(r.id, c.key)
                const bg = seqShade(v / max)
                return (
                  <td key={c.key} className="p-0.5">
                    <button
                      type="button"
                      disabled={!n || !onCell}
                      onClick={() => onCell?.(r.id, c.key)}
                      title={asRate ? `${n} 件` : undefined}
                      className="w-full rounded px-2 py-0.5 text-right tabular enabled:hover:ring-2 enabled:hover:ring-foreground/30"
                      style={{ background: n ? bg : undefined, color: n && v / max > 0.6 ? '#fff' : undefined }}
                    >
                      {n ? (asRate ? fmtRate(v) : n) : <span className="text-muted-foreground/50">·</span>}
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
