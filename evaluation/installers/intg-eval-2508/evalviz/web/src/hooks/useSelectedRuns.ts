import { useQuery } from '@tanstack/react-query'
import { useNavigate, useSearch } from '@tanstack/react-router'
import { useMemo } from 'react'
import { runsQuery } from '@/api/queries'
import type { Run } from '@/api/types'
import { joinRunIds, parseRunIds, runDisambiguator, seriesColor } from '@/lib/runs'

export interface SelectedRun extends Run {
  color: string
  index: number
  /** 選択中に同じモデルの実行があるときの区別 (実行名の異なる部分)。無ければ空 */
  note: string
}

/** URL の ?runs= で選択中の Run。選択順に系列色を割り当てる (色は Run に付き、絞り込みで塗り替えない) */
export function useSelectedRuns() {
  const search = useSearch({ strict: false }) as { runs?: string }
  const navigate = useNavigate()
  const ids = useMemo(() => parseRunIds(search.runs), [search.runs])
  const runsQ = useQuery(runsQuery())
  const byId = useMemo(() => new Map((runsQ.data ?? []).map((r) => [r.id, r])), [runsQ.data])
  const selected: SelectedRun[] = useMemo(() => {
    const runs = ids.map((id) => byId.get(id)).filter((r): r is Run => !!r)
    return runs.map((r) => ({ ...r, color: seriesColor(ids.indexOf(r.id)), index: ids.indexOf(r.id), note: runDisambiguator(r, runs) }))
  }, [ids, byId])
  const setIds = (next: string[]) =>
    navigate({ to: '.', search: (prev: Record<string, unknown>) => ({ ...prev, runs: joinRunIds(next) }), replace: true, resetScroll: false })
  const toggle = (id: string) => setIds(ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id])
  return { ids, selected, allRuns: runsQ.data ?? [], byId, isLoading: runsQ.isLoading, setIds, toggle }
}
