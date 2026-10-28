import { Link } from '@tanstack/react-router'
import type { ReactNode } from 'react'

export function PageHeader({ title, description, children }: { title: ReactNode; description?: ReactNode; children?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-start gap-4">
      <div className="min-w-0 flex-1">
        <h1 className="text-lg font-semibold tracking-tight">{title}</h1>
        {description && <p className="mt-0.5 text-sm text-muted-foreground">{description}</p>}
      </div>
      {children}
    </div>
  )
}

/** 実行が未選択のときの案内 */
export function NoRunsSelected({ what }: { what: string }) {
  return (
    <div className="m-4 rounded-lg border border-dashed p-8 text-center text-sm text-muted-foreground">
      {what}を見るには、
      <Link to="/" className="mx-1 text-foreground underline">
        リーダーボード
      </Link>
      でモデルを選ぶか、
      <Link to="/runs" className="mx-1 text-foreground underline">
        実行の一覧
      </Link>
      で実行を選ぶ
    </div>
  )
}
