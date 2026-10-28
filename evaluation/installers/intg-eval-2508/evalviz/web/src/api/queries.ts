// サーバのデータは全部 TanStack Query で取る。キーと取得関数はここに集め、画面は queryOptions を使う。
import { keepPreviousData, queryOptions } from '@tanstack/react-query'
import type { FeaturedResponse, Meta, Run, RunAnalysis, RunDetail, SampleResponse, SamplesResponse } from './types'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function getJson<T>(path: string, params?: Record<string, string | undefined>): Promise<T> {
  const qs = new URLSearchParams()
  for (const [k, v] of Object.entries(params ?? {})) if (v != null && v !== '') qs.set(k, v)
  const url = qs.size ? `${path}?${qs}` : path
  const res = await fetch(url)
  if (!res.ok) {
    let msg = res.statusText
    try {
      msg = ((await res.json()) as { error?: string }).error ?? msg
    } catch {
      /* 本文が JSON でない */
    }
    throw new ApiError(res.status, msg)
  }
  return (await res.json()) as T
}

const seg = encodeURIComponent

// 一覧はサーバが 3 分ごとに再インデックスするので、それより短く取り直す
const LIST_STALE = 60_000
// サンプル分析は推論結果ファイルが変わらない限り同じ
const ANALYSIS_STALE = 10 * 60_000

export const metaQuery = () =>
  queryOptions({ queryKey: ['meta'], queryFn: () => getJson<Meta>('/api/meta'), staleTime: 30_000 })

export const featuredQuery = () =>
  queryOptions({ queryKey: ['featured'], queryFn: () => getJson<FeaturedResponse>('/api/featured'), staleTime: LIST_STALE })

export const runsQuery = () =>
  queryOptions({ queryKey: ['runs'], queryFn: () => getJson<Run[]>('/api/runs'), staleTime: LIST_STALE })

export const runQuery = (runId: string) =>
  queryOptions({ queryKey: ['run', runId], queryFn: () => getJson<RunDetail>(`/api/runs/${seg(runId)}`), staleTime: LIST_STALE })

export const analysisQuery = (runIds: string[], ver: string | undefined) =>
  queryOptions({
    queryKey: ['analysis', ver, [...runIds].sort()],
    queryFn: () => getJson<RunAnalysis[]>('/api/analysis', { runs: runIds.join(','), ver }),
    staleTime: ANALYSIS_STALE,
    enabled: runIds.length > 0 && !!ver,
    placeholderData: keepPreviousData,
  })

export const samplesQuery = (runIds: string[], ver: string | undefined, ds: string | undefined) =>
  queryOptions({
    // 列の順が応答の cells の順になるので、並べ替えずにキーにする
    queryKey: ['samples', ver, ds, runIds],
    queryFn: () => getJson<SamplesResponse>('/api/samples', { runs: runIds.join(','), ver, ds }),
    staleTime: ANALYSIS_STALE,
    enabled: runIds.length > 0 && !!ver && !!ds,
    placeholderData: keepPreviousData,
  })

export const sampleQuery = (runIds: string[], ver: string, ds: string, key: string) =>
  queryOptions({
    queryKey: ['sample', ver, ds, key, runIds],
    queryFn: () => getJson<SampleResponse>('/api/sample', { runs: runIds.join(','), ver, ds, key }),
    staleTime: ANALYSIS_STALE,
    enabled: runIds.length > 0,
  })

export const refreshIndex = () => getJson<{ ok: boolean }>('/api/refresh')
