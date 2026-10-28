import { useQuery } from '@tanstack/react-query'
import { createFileRoute, Link } from '@tanstack/react-router'
import { ArrowLeft, ChevronRight } from 'lucide-react'
import type { ReactNode } from 'react'
import { useState } from 'react'
import { metaQuery, sampleQuery } from '@/api/queries'
import type { SampleRunDetail } from '@/api/types'
import { StatusBadge } from '@/components/charts/StatusBar'
import { NoRunsSelected, PageHeader } from '@/components/layout/PageHeader'
import { RunLabel } from '@/components/runs/RunLabel'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { type SelectedRun, useSelectedRuns } from '@/hooks/useSelectedRuns'
import { useVersion } from '@/hooks/useVersion'
import { num } from '@/lib/format'
import { cn } from '@/lib/utils'

export const Route = createFileRoute('/samples/$ds/$key')({
  component: SamplePage,
})

function Fold({ title, defaultOpen, children, note }: { title: ReactNode; defaultOpen?: boolean; children: ReactNode; note?: ReactNode }) {
  const [open, setOpen] = useState(!!defaultOpen)
  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger className="flex w-full items-center gap-1 text-left text-xs font-medium text-muted-foreground hover:text-foreground">
        <ChevronRight className={cn('size-3.5 transition-transform', open && 'rotate-90')} />
        {title}
        {note && <span className="ml-auto font-normal">{note}</span>}
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-1">{children}</CollapsibleContent>
    </Collapsible>
  )
}

function Pre({ text, className }: { text?: string | null; className?: string }) {
  if (!text) return <p className="text-xs text-muted-foreground">(空)</p>
  return <pre className={cn('max-h-[28rem] overflow-auto rounded-md bg-muted/50 p-2 text-xs leading-relaxed break-words whitespace-pre-wrap', className)}>{text}</pre>
}

function SamplePage() {
  const { ds, key } = Route.useParams()
  const { ids, selected } = useSelectedRuns()
  const { ver } = useVersion(selected)
  const q = useQuery({ ...sampleQuery(ids, ver ?? '', ds, key), enabled: ids.length > 0 && !!ver })
  const meta = useQuery(metaQuery())
  if (!ids.length) return <NoRunsSelected what="サンプル" />
  const first = q.data?.runs.find((r) => r.found)
  const info = q.data?.info
  const flagLabel = (f: string) => meta.data?.flags.find((x) => x.key === f)?.label ?? f

  return (
    <div className="flex flex-col gap-4 p-4">
      <PageHeader
        title={
          <span className="flex items-center gap-2">
            <Button asChild size="icon" variant="ghost" className="size-7">
              <Link to="/samples" search={(p) => ({ runs: p.runs, ver: p.ver, ds })} aria-label="サンプル比較へ戻る">
                <ArrowLeft className="size-4" />
              </Link>
            </Button>
            <span className="font-mono">{ds}</span>
            {first?.index != null && <span className="text-muted-foreground tabular">#{first.index}</span>}
          </span>
        }
        description={
          info && (
            <>
              llm-jp-eval {ver} · 指標 {info.metrics.join(', ')} · 回答形式 {info.pattern}
              {info.labels.length > 0 && ` · 選択肢 ${info.labels.join(' / ')}`}
              {first?.output_length != null && ` · 出力長の上限 ${first.output_length} トークン`}
            </>
          )
        }
      />
      {q.isFetching && !q.data && <p className="text-sm text-muted-foreground">読み込み中…</p>}
      {q.error && <p className="text-sm text-[var(--status-critical)]">取得に失敗: {String(q.error)}</p>}

      {first && (
        <section className="flex flex-col gap-2 rounded-lg border p-4">
          <div className="grid gap-3 lg:grid-cols-[1fr_16rem]">
            <div>
              <div className="mb-1 text-xs font-medium text-muted-foreground">入力</div>
              <Pre text={first.input} />
            </div>
            <div>
              <div className="mb-1 text-xs font-medium text-muted-foreground">正解</div>
              <Pre text={first.gold} className="font-mono" />
            </div>
          </div>
          <Fold title="プロンプト全体 (few-shot・指示を含む)">
            {first.system_prompt && (
              <>
                <div className="text-xs text-muted-foreground">system</div>
                <Pre text={first.system_prompt} />
              </>
            )}
            <Pre text={first.prompt} />
          </Fold>
          {first.metadata && Object.keys(first.metadata).length > 0 && (
            <Fold title="メタデータ">
              <Pre text={JSON.stringify(first.metadata, null, 1)} />
            </Fold>
          )}
        </section>
      )}

      <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(${Math.min(3, Math.max(1, ids.length))}, minmax(0, 1fr))` }}>
        {(q.data?.runs ?? []).map((d) => (
          <RunOutput key={d.run} d={d} run={selected.find((r) => r.id === d.run)} flagLabel={flagLabel} />
        ))}
      </div>
    </div>
  )
}

function RunOutput({ d, run, flagLabel }: { d: SampleRunDetail; run?: SelectedRun; flagLabel: (f: string) => string }) {
  return (
    <section className="flex min-w-0 flex-col gap-2 rounded-lg border p-3" style={{ borderTopColor: run?.color, borderTopWidth: 3 }}>
      <div className="flex min-w-0 items-center gap-2">
        {run ? <RunLabel run={run} color={run.color} compact className="min-w-0" /> : <span className="text-sm">{d.run}</span>}
        {d.status && <StatusBadge status={d.status} className="ml-auto" />}
      </div>
      {!d.found ? (
        <p className="text-sm text-muted-foreground">この実行にはこのサンプルが無い (推論結果が無いか、サンプル数が少ない)</p>
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-1.5 text-xs">
            <span className="text-muted-foreground">抽出した回答</span>
            <code className="rounded bg-muted px-1.5 py-0.5 break-all">{d.pred || '(なし)'}</code>
            {d.flags?.map((f) => (
              <Badge key={f} variant="outline" className="font-normal">
                {flagLabel(f)}
              </Badge>
            ))}
          </div>
          {d.reasoning ? (
            <Fold title="推論 (reasoning_content)" note={`${num(d.reasoning.length)} 字`}>
              <Pre text={d.reasoning} />
            </Fold>
          ) : null}
          <div>
            <div className="mb-1 flex text-xs font-medium text-muted-foreground">
              最終出力 <span className="ml-auto font-normal">{num(d.generated?.length ?? 0)} 字</span>
            </div>
            <Pre text={d.generated} />
          </div>
        </>
      )}
    </section>
  )
}
