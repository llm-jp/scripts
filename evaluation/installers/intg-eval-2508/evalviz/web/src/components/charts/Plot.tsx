// 散布図と折れ線 (SVG)。軸は 1 本ずつ、グリッドは細い実線、点は 8px 以上で 2px の面色の縁取り、ホバーでツールチップ
import { useEffect, useRef, useState } from 'react'
import type { MouseEvent, ReactNode } from 'react'

const M = { top: 12, right: 16, bottom: 40, left: 52 }

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [w, setW] = useState(600)
  useEffect(() => {
    if (!ref.current) return
    const ro = new ResizeObserver((es) => setW(Math.max(240, es[0].contentRect.width)))
    ro.observe(ref.current)
    return () => ro.disconnect()
  }, [])
  return [ref, w] as const
}

/** 見やすい目盛り (1, 2, 5 × 10^n 刻み) */
function ticks(lo: number, hi: number, n = 5): number[] {
  if (!(hi > lo)) return [lo]
  const raw = (hi - lo) / n
  const p = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 5, 10].map((m) => m * p).find((s) => s >= raw) ?? raw
  const out = []
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Number(v.toPrecision(12)))
  return out
}

/** 目盛りの間隔から小数の桁数を決める (0.005 刻みなら 3 桁) */
function tickFmt(ts: number[]): (v: number) => string {
  const step = ts.length > 1 ? ts[1] - ts[0] : 1
  const d = Math.max(0, Math.min(6, -Math.floor(Math.log10(step) + 1e-9)))
  return (v) => v.toFixed(d)
}

function extent(vs: number[], pad = 0.06, zero = false): [number, number] {
  let lo = Math.min(...vs)
  let hi = Math.max(...vs)
  if (zero) lo = Math.min(0, lo)
  if (lo === hi) {
    lo -= Math.abs(lo) * 0.1 || 1
    hi += Math.abs(hi) * 0.1 || 1
  }
  const d = (hi - lo) * pad
  return [zero && lo === 0 ? 0 : lo - d, hi + d]
}

function Axes({ w, h, xs, ys, fx, fy, xLabel, yLabel, fmtX }: {
  w: number; h: number; xs: number[]; ys: number[]; fx: (v: number) => number; fy: (v: number) => number
  xLabel: string; yLabel: string; fmtX: (v: number) => string
}) {
  // 縦軸の目盛りは間隔に合わせた桁数で (範囲が狭いと 0.60, 0.60 のように同じ表示が並ぶため)
  const fmtY = tickFmt(ys)
  return (
    <g className="text-[11px]" fill="var(--ink-muted)">
      {ys.map((v) => (
        <g key={`y${v}`}>
          <line x1={M.left} x2={w - M.right} y1={fy(v)} y2={fy(v)} stroke="var(--chart-grid)" strokeWidth={1} />
          <text x={M.left - 6} y={fy(v)} dy="0.32em" textAnchor="end" className="tabular">
            {fmtY(v)}
          </text>
        </g>
      ))}
      {xs.map((v) => (
        <text key={`x${v}`} x={fx(v)} y={h - M.bottom + 16} textAnchor="middle" className="tabular">
          {fmtX(v)}
        </text>
      ))}
      <line x1={M.left} x2={w - M.right} y1={h - M.bottom} y2={h - M.bottom} stroke="var(--chart-axis)" strokeWidth={1} />
      <text x={(M.left + w - M.right) / 2} y={h - 6} textAnchor="middle" fill="var(--muted-foreground)">
        {xLabel}
      </text>
      <text transform={`translate(12 ${(M.top + h - M.bottom) / 2}) rotate(-90)`} textAnchor="middle" fill="var(--muted-foreground)">
        {yLabel}
      </text>
    </g>
  )
}

function Tip({ x, y, w, children }: { x: number; y: number; w: number; children: ReactNode }) {
  return (
    <div
      className="pointer-events-none absolute z-10 rounded-md bg-foreground px-2.5 py-1.5 text-xs text-background shadow"
      style={{ left: Math.min(x + 12, w - 220), top: Math.max(0, y - 10), maxWidth: 260 }}
    >
      {children}
    </div>
  )
}

export interface ScatterPoint {
  id: string
  x: number
  y: number
  color: string
  label: ReactNode
  detail?: ReactNode
}

export function Scatter({ points, xLabel, yLabel, fmtX = String, fmtY = String, height = 300 }: {
  points: ScatterPoint[]; xLabel: string; yLabel: string; fmtX?: (v: number) => string; fmtY?: (v: number) => string; height?: number
}) {
  const [ref, w] = useWidth<HTMLDivElement>()
  const [hover, setHover] = useState<ScatterPoint | null>(null)
  if (!points.length) return <p className="text-sm text-muted-foreground">値のある実行がない</p>
  const [x0, x1] = extent(points.map((p) => p.x), 0.08, true)
  const [y0, y1] = extent(points.map((p) => p.y), 0.1)
  const fx = (v: number) => M.left + ((v - x0) / (x1 - x0)) * (w - M.left - M.right)
  const fy = (v: number) => height - M.bottom - ((v - y0) / (y1 - y0)) * (height - M.top - M.bottom)
  return (
    <div ref={ref} className="relative w-full">
      <svg width={w} height={height} className="block" onMouseLeave={() => setHover(null)}>
        <Axes w={w} h={height} xs={ticks(x0, x1)} ys={ticks(y0, y1)} fx={fx} fy={fy} xLabel={xLabel} yLabel={yLabel} fmtX={fmtX} />
        {points.map((p) => (
          <g key={p.id} onMouseEnter={() => setHover(p)}>
            {/* 当たり判定は印より大きく */}
            <circle cx={fx(p.x)} cy={fy(p.y)} r={12} fill="transparent" />
            <circle cx={fx(p.x)} cy={fy(p.y)} r={hover?.id === p.id ? 7 : 5.5} fill={p.color} stroke="var(--background)" strokeWidth={2} />
          </g>
        ))}
      </svg>
      {hover && (
        <Tip x={fx(hover.x)} y={fy(hover.y)} w={w}>
          <div className="font-medium">{hover.label}</div>
          <div className="tabular">
            {yLabel}: {fmtY(hover.y)} · {xLabel}: {fmtX(hover.x)}
          </div>
          {hover.detail}
        </Tip>
      )}
    </div>
  )
}

export interface LineSeries {
  id: string
  color: string
  label: ReactNode
  points: { x: number; y: number; detail?: ReactNode }[]
}

/** 折れ線 (x = ステップ)。ホバーで最も近い x の縦線と、その x の全系列の値 */
export function LineChart({ series, xLabel, yLabel, fmtX = String, fmtY = String, height = 300 }: {
  series: LineSeries[]; xLabel: string; yLabel: string; fmtX?: (v: number) => string; fmtY?: (v: number) => string; height?: number
}) {
  const [ref, w] = useWidth<HTMLDivElement>()
  const [hx, setHx] = useState<number | null>(null)
  const all = series.flatMap((s) => s.points)
  if (!all.length) return <p className="text-sm text-muted-foreground">値のある実行がない</p>
  const [x0, x1] = extent(all.map((p) => p.x), 0.04)
  const [y0, y1] = extent(all.map((p) => p.y), 0.1)
  const fx = (v: number) => M.left + ((v - x0) / (x1 - x0)) * (w - M.left - M.right)
  const fy = (v: number) => height - M.bottom - ((v - y0) / (y1 - y0)) * (height - M.top - M.bottom)
  const xsAll = [...new Set(all.map((p) => p.x))].sort((a, b) => a - b)
  const onMove = (e: MouseEvent<SVGSVGElement>) => {
    const px = e.clientX - e.currentTarget.getBoundingClientRect().left
    let best = xsAll[0]
    for (const x of xsAll) if (Math.abs(fx(x) - px) < Math.abs(fx(best) - px)) best = x
    setHx(best)
  }
  return (
    <div ref={ref} className="relative w-full">
      <svg width={w} height={height} className="block" onMouseMove={onMove} onMouseLeave={() => setHx(null)}>
        <Axes w={w} h={height} xs={ticks(x0, x1, 6)} ys={ticks(y0, y1)} fx={fx} fy={fy} xLabel={xLabel} yLabel={yLabel} fmtX={fmtX} />
        {hx != null && <line x1={fx(hx)} x2={fx(hx)} y1={M.top} y2={height - M.bottom} stroke="var(--chart-axis)" strokeWidth={1} />}
        {series.map((s) => {
          const pts = [...s.points].sort((a, b) => a.x - b.x)
          return (
            <g key={s.id}>
              <polyline points={pts.map((p) => `${fx(p.x)},${fy(p.y)}`).join(' ')} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" />
              {pts.map((p) => (
                <circle key={p.x} cx={fx(p.x)} cy={fy(p.y)} r={p.x === hx ? 5.5 : 4} fill={s.color} stroke="var(--background)" strokeWidth={2} />
              ))}
            </g>
          )
        })}
      </svg>
      {hx != null && (
        <Tip x={fx(hx)} y={M.top} w={w}>
          <div className="mb-0.5 font-medium tabular">
            {xLabel} {fmtX(hx)}
          </div>
          {series.map((s) => {
            const p = s.points.find((q) => q.x === hx)
            return p ? (
              <div key={s.id} className="flex items-center gap-1.5 tabular">
                <span className="size-2 shrink-0 rounded-full" style={{ background: s.color }} />
                <span className="min-w-0 truncate">{s.label}</span>
                <span className="ml-auto pl-2">{fmtY(p.y)}</span>
              </div>
            ) : null
          })}
        </Tip>
      )}
    </div>
  )
}
