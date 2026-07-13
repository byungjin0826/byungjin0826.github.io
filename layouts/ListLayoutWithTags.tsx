'use client'

import { useState } from 'react'
import { usePathname } from 'next/navigation'
import type { CoreContent, Blog } from '@/lib/content-types'
import Link from '@/components/Link'
import { Chip, Divider, Eyebrow } from '@/components/journal/ui'
import { CATEGORIES, inCategory, postCategory, dotDate } from '@/components/journal/meta'

interface PaginationProps {
  totalPages: number
  currentPage: number
}
interface ListLayoutProps {
  posts: CoreContent<Blog>[]
  title: string
  initialDisplayPosts?: CoreContent<Blog>[]
  pagination?: PaginationProps
}

function ArchiveRow({ post }: { post: CoreContent<Blog> }) {
  return (
    <Link
      href={`/${post.path}`}
      className="group grid grid-cols-[1fr_auto] items-baseline gap-x-4 gap-y-1 border-t border-line py-4 sm:grid-cols-[96px_1fr_auto]"
    >
      <span className="order-2 font-mono text-xs text-ink-muted sm:order-none sm:pt-1">
        {dotDate(post.date)}
      </span>
      <div className="order-1 col-span-2 sm:order-none sm:col-span-1">
        <h3 className="text-[1.1875rem] font-extrabold tracking-[-0.02em] text-ink transition-colors group-hover:text-accent-text">
          {post.title}
        </h3>
        {post.summary && (
          <p className="mt-1.5 text-[0.8125rem] leading-relaxed text-ink-muted">{post.summary}</p>
        )}
      </div>
      <span className="order-3 sm:order-none">
        <Chip variant="solid">{postCategory(post.tags)}</Chip>
      </span>
    </Link>
  )
}

export default function ListLayoutWithTags({ posts, title }: ListLayoutProps) {
  const pathname = usePathname() || ''
  const showFilter = pathname.startsWith('/blog')
  const [cat, setCat] = useState<string>('전체')

  const filtered = showFilter ? posts.filter((p) => inCategory(p.tags, cat)) : posts

  // 연도별 그룹
  const byYear: Record<string, CoreContent<Blog>[]> = {}
  filtered.forEach((p) => {
    const y = new Date(p.date).getFullYear().toString()
    ;(byYear[y] = byYear[y] || []).push(p)
  })
  const years = Object.keys(byYear).sort((a, b) => Number(b) - Number(a))

  const heading = showFilter && cat !== '전체' ? `전체 글 · ${cat}` : title

  return (
    <div className="ds-prose-container pt-8 py-8">
      <Eyebrow>Archive</Eyebrow>
      <h1 className="mt-3 text-[3rem] font-extrabold leading-[1.05] tracking-[-0.03em] text-ink">
        {heading}
      </h1>
      <p className="mt-2.5 font-mono text-[0.8125rem] text-ink-muted">{filtered.length} posts</p>

      {showFilter && (
        <div className="mt-6 flex flex-wrap gap-2">
          {CATEGORIES.map((c) => (
            <Chip key={c} active={cat === c} onClick={() => setCat(c)}>
              {c}
            </Chip>
          ))}
        </div>
      )}

      {years.map((y) => (
        <section key={y} className="mt-12">
          <Divider label={y} />
          <div className="mt-4">
            {byYear[y].map((p) => (
              <ArchiveRow key={p.path} post={p} />
            ))}
          </div>
        </section>
      ))}

      {filtered.length === 0 && (
        <p className="mt-8 text-ink-muted">해당하는 글이 없습니다.</p>
      )}
    </div>
  )
}
