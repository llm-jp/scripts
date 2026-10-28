import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'

/** 比べる llm-jp-eval のバージョン。括弧内は結果のある選択中の実行の数 */
export function VersionSelect({ ver, versions, onChange }: { ver?: string; versions: { version: string; n: number }[]; onChange: (v: string) => void }) {
  if (!versions.length) return null
  return (
    <Select value={ver} onValueChange={onChange}>
      <SelectTrigger size="sm" className="w-52">
        <SelectValue placeholder="llm-jp-eval のバージョン" />
      </SelectTrigger>
      <SelectContent>
        {versions.map((v) => (
          <SelectItem key={v.version} value={v.version}>
            llm-jp-eval {v.version} <span className="text-muted-foreground">({v.n} 実行)</span>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}
