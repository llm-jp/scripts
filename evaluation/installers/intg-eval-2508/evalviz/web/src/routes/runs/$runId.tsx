import { useQuery } from '@tanstack/react-query'
import { createFileRoute, Link } from '@tanstack/react-router'
import { runQuery } from '@/api/queries'
import type { LjEvalSummary, RunDetail } from '@/api/types'
import { PageHeader } from '@/components/layout/PageHeader'
import { Section } from '@/components/layout/Section'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { useSelectedRuns } from '@/hooks/useSelectedRuns'
import { dateTime } from '@/lib/format'
import { compareVersion } from '@/lib/runs'

export const Route = createFileRoute('/runs/$runId')({
  component: RunPage,
})

const f = (v: number | null | undefined, d = 4) => (v == null ? '—' : Number(v.toFixed(d)))

function RunPage() {
  const { runId } = Route.useParams()
  const q = useQuery(runQuery(runId))
  const { ids, toggle } = useSelectedRuns()
  const r = q.data
  if (q.error) return <p className="p-4 text-sm text-[var(--status-critical)]">{String(q.error)}</p>
  if (!r) return <p className="p-4 text-sm text-muted-foreground">読み込み中…</p>
  const lj = Object.entries(r.components['llm-jp-eval'] ?? {}).sort((a, b) => compareVersion(b[0], a[0]))
  return (
    <div className="flex flex-col gap-4 p-4">
      <PageHeader
        title={r.name}
        description={
          <>
            <span className="font-mono">{r.dir}</span>
            <br />
            モデル <span className="font-mono">{r.model_path || r.model}</span> · 完了 {dateTime(r.finished_at)}
          </>
        }
      >
        <Button size="sm" variant={ids.includes(r.id) ? 'secondary' : 'default'} onClick={() => toggle(r.id)}>
          {ids.includes(r.id) ? '比較から外す' : '比較に加える'}
        </Button>
      </PageHeader>

      <Section title="評価条件" description="qsub.py が書いた config.json (既定値のままの項目を含む)">
        <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-0.5 text-xs">
          {Object.entries(r.config).map(([k, v]) => (
            <div key={k} className="contents">
              <dt className="font-mono text-muted-foreground">{k}</dt>
              <dd className="font-mono break-all">{typeof v === 'object' ? JSON.stringify(v) : String(v)}</dd>
            </div>
          ))}
          {Object.keys(r.config).length === 0 && <dd className="text-muted-foreground">config.json なし</dd>}
        </dl>
      </Section>

      {lj.map(([ver, c]) => (
        <LjEvalSection key={ver} run={r} c={c} raw={r.raw_scores[ver]} />
      ))}

      {r.components.swallow?.status === 'done' && (
        <Section title="swallow" description={`英語平均 ${f(r.components.swallow.avg?.EN, 3)} (${r.components.swallow.n_en} タスク) · 日本語平均 ${f(r.components.swallow.avg?.JA, 3)} (${r.components.swallow.n_ja} タスク)。-1 (未実行) は除いた`}>
          <KvGrid items={Object.entries(r.components.swallow.scores ?? {})} />
        </Section>
      )}
      {r.components['llm-jp-judge']?.status === 'done' && (
        <Section title="llm-jp-judge" description={`ジャッジ ${r.components['llm-jp-judge'].judge_model ?? '?'}`}>
          <KvGrid items={Object.entries(r.components['llm-jp-judge'].scores ?? {}).flatMap(([b, as]) => Object.entries(as).map(([a, v]) => [`${b}:${a}`, v] as [string, number]))} />
          {Object.keys(r.components['llm-jp-judge'].errors ?? {}).length > 0 && (
            <>
              <div className="mt-3 mb-1 text-xs font-medium text-muted-foreground">ジャッジ出力の解析失敗率 (%)</div>
              <KvGrid items={Object.entries(r.components['llm-jp-judge'].errors ?? {}).flatMap(([b, as]) => Object.entries(as).map(([a, v]) => [`${b}:${a}`, v] as [string, number]))} />
            </>
          )}
        </Section>
      )}
      {r.components['safety-eval']?.status === 'done' && (
        <Section title="safety-eval">
          <KvGrid
            items={Object.entries(r.components['safety-eval'].benchmarks ?? {}).flatMap(([b, x]) =>
              Object.entries(x.metrics).flatMap(([k, v]) => (v == null ? [] : [[`${b}:${k}`, v] as [string, number]])),
            )}
          />
        </Section>
      )}
    </div>
  )
}

function KvGrid({ items }: { items: [string, number][] }) {
  return (
    <dl className="grid grid-cols-[repeat(auto-fill,minmax(18rem,1fr))] gap-x-6 gap-y-0.5 text-sm">
      {items.map(([k, v]) => (
        <div key={k} className="flex justify-between gap-2 border-b border-dashed border-transparent">
          <dt className="truncate font-mono text-xs text-muted-foreground" title={k}>
            {k}
          </dt>
          <dd className="tabular">{f(v)}</dd>
        </div>
      ))}
    </dl>
  )
}

function LjEvalSection({ run, c, raw }: { run: RunDetail; c: LjEvalSummary; raw?: Record<string, number> }) {
  if (c.status !== 'done')
    return (
      <Section title={`llm-jp-eval ${c.version}`}>
        <p className="text-sm text-muted-foreground">{c.status === 'missing' ? 'result.json が無い (実行中か失敗)' : 'result.json を読めない'}</p>
      </Section>
    )
  const ds = Object.entries(c.datasets ?? {}).sort((a, b) => b[1].lang.localeCompare(a[1].lang) || (a[1].category ?? '').localeCompare(b[1].category ?? '') || a[0].localeCompare(b[0]))
  return (
    <Section
      title={`llm-jp-eval ${c.version}`}
      description={`${c.n_samples === -1 ? '全件' : `データセットあたり ${c.n_samples ?? '?'} 件`} · ${c.few_shots ?? '?'}-shot${c.infer_s != null ? ` · 推論 ${Math.round(c.infer_s)} 秒` : ''}`}
      actions={
        c.has_offline && (
          <Button asChild size="sm" variant="outline">
            <Link to="/samples" search={(p) => ({ runs: p.runs?.split(',').includes(run.id) ? p.runs : [p.runs, run.id].filter(Boolean).join(','), ver: c.version })}>
              サンプルを見る
            </Link>
          </Button>
        )
      }
    >
      <div className="flex flex-wrap gap-2 text-sm">
        {Object.entries(c.avg ?? {}).map(([l, v]) => (
          <Badge key={l} variant="secondary" className="tabular">
            {l} AVG {f(v, 4)}
          </Badge>
        ))}
      </div>
      {Object.entries(c.categories ?? {}).map(([lang, cats]) => (
        <div key={lang} className="mt-2">
          <div className="mb-1 text-xs font-medium text-muted-foreground">{lang} のカテゴリ</div>
          <KvGrid items={Object.entries(cats)} />
        </div>
      ))}
      <div className="mt-3 mb-1 text-xs font-medium text-muted-foreground">データセット (主指標と ool)</div>
      <table className="text-sm">
        <tbody>
          {ds.map(([name, d]) => (
            <tr key={name} className="border-t">
              <td className="px-2 py-0.5 font-mono text-xs">{name}</td>
              <td className="px-2 py-0.5 text-xs text-muted-foreground">
                {d.lang} · {d.category ?? '—'} · {d.metric}
              </td>
              <td className="px-2 py-0.5 text-right tabular">{f(d.score)}</td>
              <td className="px-2 py-0.5 text-right text-xs text-muted-foreground tabular">{d.ool != null && `ool ${f(d.ool, 3)}`}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {raw && Object.keys(raw).length > 0 && (
        <details className="mt-3">
          <summary className="cursor-pointer text-xs font-medium text-muted-foreground">result.json の全スコア ({Object.keys(raw).length})</summary>
          <div className="mt-2">
            <KvGrid items={Object.entries(raw).filter(([, v]) => typeof v === 'number')} />
          </div>
        </details>
      )}
    </Section>
  )
}
