import { Link } from '@tanstack/react-router'
import { X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { useSelectedRuns } from '@/hooks/useSelectedRuns'
import { RunLabel } from './RunLabel'

/** ヘッダーの「選択中の実行」。どの画面からでも外せる。追加は実行一覧で */
export function SelectedRunsChip() {
  const { ids, selected, setIds } = useSelectedRuns()
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="outline" size="sm" className="gap-1.5">
          <span className="flex -space-x-1">
            {selected.slice(0, 8).map((r) => (
              <span key={r.id} className="size-2.5 rounded-full ring-2 ring-background" style={{ background: r.color }} />
            ))}
          </span>
          選択中の実行 {ids.length}
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-[32rem] p-2">
        {selected.length === 0 && <p className="p-2 text-sm text-muted-foreground">未選択。リーダーボードか「実行を選ぶ」でチェックを付ける</p>}
        <ul className="flex flex-col gap-0.5">
          {selected.map((r) => (
            <li key={r.id} className="flex items-center gap-2 rounded px-1.5 py-1 hover:bg-muted">
              <Link to="/runs/$runId" params={{ runId: r.id }} className="min-w-0 flex-1">
                <RunLabel run={r} color={r.color} />
              </Link>
              <Button variant="ghost" size="icon" className="size-6" onClick={() => setIds(ids.filter((x) => x !== r.id))} aria-label="選択を外す">
                <X className="size-3.5" />
              </Button>
            </li>
          ))}
        </ul>
        {selected.length > 0 && (
          <div className="mt-2 flex justify-between border-t pt-2">
            <Button variant="ghost" size="sm" onClick={() => setIds([])}>
              すべて外す
            </Button>
            <Button asChild variant="ghost" size="sm">
              <Link to="/runs">実行の一覧で追加</Link>
            </Button>
          </div>
        )}
      </PopoverContent>
    </Popover>
  )
}
