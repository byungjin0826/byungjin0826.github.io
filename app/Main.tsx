'use client'

import { useState } from 'react'
import { slug } from 'github-slugger'
import type { CoreContent, Blog } from '@/lib/content-types'
import Link from '@/components/Link'
import tagData from 'app/tag-data.json'
import { Chip, Avatar, Divider, Eyebrow } from '@/components/journal/ui'
import { Arrow } from '@/components/journal/icons'
import { CATEGORIES, inCategory, postCategory, readingLabel, dotDate, topicTags } from '@/components/journal/meta'

type Post = CoreContent<Blog>

function HomeCard({ post }: { post: Post }) {
  return (
    <Link
      href={`/blog/${post.slug}`}
      className="group flex flex-col gap-3 border-t-2 border-rule pt-4"
    >
      <div className="flex items-center gap-3">
        <Chip variant="solid">{postCategory(post.tags)}</Chip>
        <span className="font-mono text-xs text-ink-muted">
          {dotDate(post.date)} · {readingLabel(post.readingTime)}
        </span>
      </div>
      <h3 className="text-[1.375rem] font-extrabold leading-[1.2] tracking-[-0.02em] text-ink transition-colors group-hover:text-accent-text">
        {post.title}
      </h3>
      {post.summary && (
        <p className="text-[0.9375rem] leading-relaxed text-ink-muted">{post.summary}</p>
      )}
    </Link>
  )
}

export default function Main({ posts }: { posts: Post[] }) {
  const [cat, setCat] = useState<string>('전체')

  const filtered = posts.filter((p) => inCategory(p.tags, cat))
  const featured = filtered[0]
  const rest = filtered.slice(1, 5)

  const popularTags = Object.entries(tagData as Record<string, number>)
    .filter(([t]) => t !== 'tech' && t !== 'daily')
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([t]) => t)

  return (
    <div className="ds-container py-12">
      {/* 카테고리 탭 */}
      <div className="mb-8 flex flex-wrap gap-2">
        {CATEGORIES.map((c) => (
          <Chip key={c} active={cat === c} onClick={() => setCat(c)}>
            {c}
          </Chip>
        ))}
      </div>

      {/* 피처드 히어로 */}
      {featured && (
        <Link
          href={`/blog/${featured.slug}`}
          className="group block border-t-2 border-rule pt-8"
        >
          <div className="mb-5 flex flex-wrap items-center gap-3">
            <Chip variant="solid">{postCategory(featured.tags)}</Chip>
            <span className="font-mono text-xs font-semibold tracking-[0.12em] text-accent-text">
              FEATURED
            </span>
            <span className="font-mono text-xs text-ink-muted">
              {dotDate(featured.date)} · {readingLabel(featured.readingTime)}
            </span>
          </div>
          <h1 className="m-0 max-w-[14ch] text-[clamp(2.4rem,5.2vw,4.1rem)] font-extrabold leading-[1.04] tracking-[-0.035em] text-ink transition-colors group-hover:text-accent-text">
            {featured.title}
          </h1>
          {featured.summary && (
            <p className="mt-5 max-w-[58ch] text-[1.1875rem] leading-relaxed text-ink-muted">
              {featured.summary}
            </p>
          )}
          <div className="mt-5 flex flex-wrap gap-2">
            {topicTags(featured.tags)
              .slice(0, 4)
              .map((t) => (
                <span key={t} className="font-mono text-xs text-ink-subtle">
                  #{t}
                </span>
              ))}
          </div>
          <div className="mt-6">
            <span className="inline-flex items-center justify-center gap-2 rounded-md border-[1.5px] border-accent bg-accent px-5 py-2.5 text-[0.9375rem] font-semibold text-on-accent transition-colors group-hover:border-accent-hover group-hover:bg-accent-hover">
              읽어보기 <Arrow size={16} />
            </span>
          </div>
        </Link>
      )}

      <div className="my-8 h-0.5 bg-rule" />

      {/* 본문 그리드 */}
      <div className="grid grid-cols-1 items-start gap-8 lg:grid-cols-[minmax(0,1fr)_300px]">
        <div>
          <div className="mb-5 flex items-baseline justify-between">
            <Eyebrow>최근 글 {cat !== '전체' ? `· ${cat}` : ''}</Eyebrow>
            <Link
              href="/blog"
              className="inline-flex items-center gap-1.5 text-[0.8125rem] font-bold text-ink-muted transition-colors hover:text-ink"
            >
              전체 보기 <Arrow size={13} />
            </Link>
          </div>
          {rest.length > 0 ? (
            <div className="grid grid-cols-1 gap-x-8 gap-y-10 sm:grid-cols-2">
              {rest.map((p) => (
                <HomeCard key={p.slug} post={p} />
              ))}
            </div>
          ) : (
            <p className="text-ink-muted">이 카테고리의 다른 글은 곧 올라옵니다.</p>
          )}
        </div>

        {/* 사이드바 */}
        <aside className="flex flex-col gap-8 lg:sticky lg:top-[92px]">
          <div className="rounded-lg border border-line bg-surface p-6">
            <div className="flex items-center gap-3">
              <Avatar name="/steps" size="lg" />
              <div>
                <div className="font-extrabold tracking-[-0.01em] text-ink">/steps</div>
                <div className="text-[0.8125rem] text-ink-muted">Data Scientist</div>
              </div>
            </div>
          </div>

          <div>
            <Divider label="인기 태그" />
            <div className="mt-4 flex flex-wrap gap-2">
              {popularTags.map((t) => (
                <Chip key={t} href={`/tags/${slug(t)}`}>
                  #{t}
                </Chip>
              ))}
            </div>
          </div>
        </aside>
      </div>
    </div>
  )
}
