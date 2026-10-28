import type { ReactNode } from 'react'
import { Link } from '@tanstack/react-router'
import type { SelectedRun } from '@/hooks/useSelectedRuns'
import { shortTitle } from '@/lib/runs'
import { cn } from '@/lib/utils'

/** 表の列見出しの実行。系列色の点 + 短い名前 (折り返す) + 同じモデルの実行を区別する補足。全体はツールチップ */
export function RunColumnHead({ run, all, className, children }: { run: SelectedRun; all: SelectedRun[]; className?: string; children?: ReactNode }) {
  return (
    <div className={cn('w-full min-w-0 text-left', className)}>
      <Link to="/runs/$runId" params={{ runId: run.id }} className="block hover:underline" title={`${run.model}\n${run.name}\n${run.dir}`}>
        <span className="flex items-start gap-1.5">
          <span className="mt-1 size-2.5 shrink-0 rounded-full" style={{ background: run.color }} />
          <span className="min-w-0 text-xs font-medium break-all">{shortTitle(run, all)}</span>
        </span>
      </Link>
      {run.note && <div className="truncate text-[11px] font-normal text-muted-foreground" title={run.name}>{run.note}</div>}
      {children}
    </div>
  )
}
