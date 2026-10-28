import type { Status } from '@/api/types'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { STATUS_ORDER, STATUSES } from '@/lib/status'
import { cn } from '@/lib/utils'

/** サンプルの状態の積み上げ横棒 (セグメント間 2px の隙間、端は角丸)。区間のクリックで絞り込み先へ */
export function StatusBar({
  counts,
  className,
  onSegment,
  order = STATUS_ORDER,
}: {
  counts: Partial<Record<Status, number>>
  className?: string
  onSegment?: (s: Status) => void
  order?: Status[]
}) {
  const n = order.reduce((a, o) => a + (counts[o] ?? 0), 0)
  const segs = order.filter((o) => (counts[o] ?? 0) > 0)
  return (
    <div className={cn('flex h-3 w-full gap-[2px]', className)}>
      {segs.map((o) => (
        <Tooltip key={o}>
          <TooltipTrigger asChild>
            <button
              type="button"
              className={cn('h-full min-w-[3px] rounded-[3px] first:rounded-l-[4px] last:rounded-r-[4px]', onSegment ? 'cursor-pointer' : 'cursor-default')}
              style={{ flexGrow: counts[o], background: STATUSES[o].color }}
              onClick={() => onSegment?.(o)}
              aria-label={`${STATUSES[o].label} ${counts[o]}`}
            />
          </TooltipTrigger>
          <TooltipContent>
            {STATUSES[o].symbol} {STATUSES[o].label}: {counts[o]} / {n} ({(((counts[o] ?? 0) / Math.max(1, n)) * 100).toFixed(1)}%)
          </TooltipContent>
        </Tooltip>
      ))}
    </div>
  )
}

export function StatusLegend({ statuses = STATUS_ORDER }: { statuses?: Status[] }) {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
      {statuses.map((o) => (
        <span key={o} className="inline-flex items-center gap-1">
          <span className="size-2.5 rounded-[3px]" style={{ background: STATUSES[o].color }} />
          {STATUSES[o].symbol} {STATUSES[o].label}
        </span>
      ))}
    </div>
  )
}

export function StatusChip({ status, className }: { status: Status; className?: string }) {
  const s = STATUSES[status]
  return (
    <span
      className={cn('inline-flex size-4 shrink-0 items-center justify-center rounded-[4px] text-[10px] font-bold', className)}
      style={{ background: s.color, color: s.ink }}
      title={s.label}
    >
      {s.symbol}
    </span>
  )
}

export function StatusBadge({ status, className }: { status: Status; className?: string }) {
  const s = STATUSES[status]
  return (
    <span className={cn('inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium whitespace-nowrap', className)}>
      <span className="size-2 rounded-full" style={{ background: s.color }} />
      {s.symbol} {s.label}
    </span>
  )
}
