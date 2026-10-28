import type { Run } from '@/api/types'
import { runTitle } from '@/lib/runs'
import { cn } from '@/lib/utils'

/** 実行の表示: 系列色の点 + モデル名 + (補足)。補足は同じモデルの実行を区別する文字列 (runDisambiguator) か実行名 */
export function RunLabel({ run, color, compact = false, className }: { run: Run & { note?: string }; color?: string; compact?: boolean; className?: string }) {
  const sub = compact ? run.note : run.name
  return (
    <span className={cn('inline-flex max-w-[20rem] min-w-0 items-center gap-1.5', className)} title={`${run.model_path || run.model}\n${run.dir}`}>
      {color && <span className="size-2.5 shrink-0 rounded-full" style={{ background: color }} />}
      <span className="truncate font-medium">{runTitle(run)}</span>
      {sub && <span className="truncate text-xs text-muted-foreground">{sub}</span>}
    </span>
  )
}
