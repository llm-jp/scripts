import type { QueryClient } from '@tanstack/react-query'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { createRootRouteWithContext, Link, Outlet, retainSearchParams } from '@tanstack/react-router'
import { RefreshCw } from 'lucide-react'
import { metaQuery, refreshIndex } from '@/api/queries'
import { SelectedRunsChip } from '@/components/runs/SelectedRunsChip'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'

// 選択中の実行 (カンマ区切り) と比べる llm-jp-eval のバージョンは全画面で共有する URL 状態。ページを移っても保つ
export interface RootSearch {
  runs?: string
  ver?: string
}

export const Route = createRootRouteWithContext<{ queryClient: QueryClient }>()({
  validateSearch: (s: Record<string, unknown>): RootSearch => ({
    runs: typeof s.runs === 'string' && s.runs ? s.runs : undefined,
    ver: typeof s.ver === 'string' && s.ver ? s.ver : undefined,
  }),
  search: { middlewares: [retainSearchParams(['runs', 'ver'])] },
  component: RootLayout,
})

const NAV = [
  { to: '/', label: 'リーダーボード' },
  { to: '/runs', label: '実行を選ぶ' },
  { to: '/overview', label: '総合比較' },
  { to: '/datasets', label: 'データセット別' },
  { to: '/errors', label: '誤りパターン' },
  { to: '/samples', label: 'サンプル比較' },
] as const

function RootLayout() {
  return (
    <div className="flex min-h-svh flex-col">
      <header className="sticky top-0 z-30 flex h-12 shrink-0 items-center gap-4 border-b bg-background/95 px-4 backdrop-blur">
        <Link to="/" className="font-semibold tracking-tight">
          evalviz
        </Link>
        <nav className="flex items-center gap-1 text-sm">
          {NAV.map((n) => (
            <Link
              key={n.to}
              to={n.to}
              activeOptions={{ exact: n.to === '/', includeSearch: false }}
              className={cn('rounded-md px-2.5 py-1 text-muted-foreground hover:bg-muted hover:text-foreground')}
              activeProps={{ className: 'bg-muted text-foreground font-medium' }}
            >
              {n.label}
            </Link>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-2">
          <SelectedRunsChip />
          <IndexStatus />
        </div>
      </header>
      <main className="flex min-h-0 flex-1 flex-col">
        <Outlet />
      </main>
    </div>
  )
}

function IndexStatus() {
  const qc = useQueryClient()
  const { data } = useQuery({ ...metaQuery(), refetchInterval: (q) => (q.state.data?.refreshing ? 3000 : 60_000) })
  const built = data ? new Date(data.built_at * 1000) : null
  const onRefresh = async () => {
    await refreshIndex()
    await qc.invalidateQueries({ queryKey: ['meta'] })
    // 再インデックスは非同期。完了 (refreshing=false) 後に一覧系を取り直す
    const wait = async () => {
      const m = await qc.fetchQuery({ ...metaQuery(), staleTime: 0 })
      if (m.refreshing) return setTimeout(wait, 2000)
      await qc.invalidateQueries({ predicate: (q) => ['runs', 'run', 'featured', 'analysis', 'samples'].includes(q.queryKey[0] as string) })
    }
    setTimeout(wait, 1000)
  }
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button variant="ghost" size="sm" onClick={onRefresh} className="text-muted-foreground">
          <RefreshCw className={cn('size-3.5', data?.refreshing && 'animate-spin')} />
          <span className="tabular text-xs">{built ? built.toLocaleTimeString('ja-JP', { hour: '2-digit', minute: '2-digit' }) : '—'}</span>
        </Button>
      </TooltipTrigger>
      <TooltipContent side="bottom" className="max-w-sm">
        <div>最終インデックス {built?.toLocaleString('ja-JP') ?? '—'}。クリックで再インデックス (サーバは 3 分ごとに自動で更新)</div>
        {data?.roots.map((r) => (
          <div key={r} className="font-mono text-[11px] opacity-80">
            {r}
          </div>
        ))}
        {data?.last_error && <div className="text-red-300">前回の失敗: {data.last_error}</div>}
      </TooltipContent>
    </Tooltip>
  )
}
