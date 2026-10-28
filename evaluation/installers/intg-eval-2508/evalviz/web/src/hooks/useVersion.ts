import { useNavigate, useSearch } from '@tanstack/react-router'
import { useMemo } from 'react'
import { compareVersion, ljVersions } from '@/lib/runs'
import type { SelectedRun } from './useSelectedRuns'

/**
 * 比較に使う llm-jp-eval のバージョン。バージョンが違うとデータセットも指標も違うので、画面は 1 つのバージョンだけを比べる。
 * URL の ?ver= を優先し、無ければ選択中の実行で結果のある数が最も多いもの (同数なら新しい方)
 */
export function useVersion(selected: SelectedRun[]) {
  const search = useSearch({ strict: false }) as { ver?: string }
  const navigate = useNavigate()
  const versions = useMemo(() => {
    const c = new Map<string, number>()
    for (const r of selected) for (const v of ljVersions(r)) c.set(v, (c.get(v) ?? 0) + 1)
    return [...c.entries()].sort((a, b) => b[1] - a[1] || compareVersion(b[0], a[0])).map(([v, n]) => ({ version: v, n }))
  }, [selected])
  const ver = search.ver && versions.some((v) => v.version === search.ver) ? search.ver : versions[0]?.version
  const setVer = (v: string) =>
    navigate({ to: '.', search: (prev: Record<string, unknown>) => ({ ...prev, ver: v }), replace: true, resetScroll: false })
  return { ver, versions, setVer }
}
