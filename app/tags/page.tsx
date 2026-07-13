import { slug } from 'github-slugger'
import tagData from 'app/tag-data.json'
import { genPageMetadata } from 'app/seo'
import { Chip, Eyebrow } from '@/components/journal/ui'

export const metadata = genPageMetadata({ title: 'Tags', description: '태그로 글 둘러보기' })

export default function Page() {
  const tagCounts = tagData as Record<string, number>
  const tagKeys = Object.keys(tagCounts)
  const sortedTags = tagKeys.sort((a, b) => tagCounts[b] - tagCounts[a])

  return (
    <div className="ds-prose-container py-8 pt-8">
      <Eyebrow>Tags</Eyebrow>
      <h1 className="mt-3 text-[3rem] font-extrabold leading-[1.05] tracking-[-0.03em] text-ink">
        태그
      </h1>
      <p className="mt-2.5 font-mono text-[0.8125rem] text-ink-muted">{tagKeys.length} tags</p>

      <div className="mt-6 flex flex-wrap gap-2.5">
        {tagKeys.length === 0 && <p className="text-ink-muted">태그가 없습니다.</p>}
        {sortedTags.map((t) => (
          <Chip key={t} href={`/tags/${slug(t)}`}>
            #{t}
            <span className="font-mono text-[0.85em] opacity-55">{tagCounts[t]}</span>
          </Chip>
        ))}
      </div>
    </div>
  )
}
