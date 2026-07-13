import { ReactNode } from 'react'
import type { CoreContent, Blog, Authors } from '@/lib/content-types'
import { slug } from 'github-slugger'
import Comments from '@/components/Comments'
import Link from '@/components/Link'
import siteMetadata from '@/data/siteMetadata'
import ScrollTopAndComment from '@/components/ScrollTopAndComment'
import { Chip, Avatar, Divider, Button, Eyebrow } from '@/components/journal/ui'
import { ArrowLeft, Clock, Github } from '@/components/journal/icons'
import CopyLinkButton from '@/components/journal/CopyLinkButton'
import { postCategory, readingLabel, dotDate, topicTags } from '@/components/journal/meta'

const editUrl = (path) => `${siteMetadata.siteRepo}/blob/main/data/${path}`

const AUTHOR_BIO =
  '데이터·AI 엔지니어. RAG·LLM 인프라와 시계열, 그리고 개발하며 겪은 것들을 기록합니다. 여행과 잡다한 호기심도 함께.'

interface LayoutProps {
  content: CoreContent<Blog>
  authorDetails: CoreContent<Authors>[]
  next?: { path: string; title: string; tags?: string[] }
  prev?: { path: string; title: string; tags?: string[] }
  children: ReactNode
}

export default function PostLayout({ content, next, prev, children }: LayoutProps) {
  const { filePath, path, slug: postSlug, date, title, tags, summary } = content
  const basePath = path.split('/')[0]
  const nextPost = next && next.path ? next : prev && prev.path ? prev : null

  return (
    <article className="ds-prose-container pb-4 pt-7">
      <ScrollTopAndComment />

      <Link
        href={`/${basePath}`}
        className="inline-flex items-center gap-1.5 text-[0.8125rem] font-bold text-ink-muted transition-colors hover:text-ink"
      >
        <ArrowLeft size={15} /> 목록으로
      </Link>

      {/* 헤더 */}
      <header className="pt-7">
        <div className="mb-4 flex items-center gap-3">
          <Chip variant="solid">{postCategory(tags)}</Chip>
          <span className="font-mono text-xs text-ink-muted">{dotDate(date)}</span>
        </div>
        <h1 className="m-0 text-[clamp(2rem,6vw,3rem)] font-extrabold leading-[1.08] tracking-[-0.03em] text-ink">
          {title}
        </h1>
        {summary && (
          <p className="mt-4 text-[1.1875rem] leading-relaxed text-ink-muted">{summary}</p>
        )}
        <div className="mt-6 flex items-center gap-3 border-b-2 border-rule pb-6">
          <Avatar name="/steps" size="md" />
          <div className="text-[0.8125rem]">
            <div className="font-bold text-ink">/steps</div>
            <div className="flex items-center gap-1.5 text-ink-muted">
              <Clock size={13} /> {readingLabel(content.readingTime)}
            </div>
          </div>
        </div>
      </header>

      {/* 본문 */}
      <div className="prose max-w-none pt-8 dark:prose-invert">{children}</div>

      {/* 토픽 태그 */}
      {tags && tags.length > 0 && (
        <div className="mt-10 flex flex-wrap gap-2">
          {topicTags(tags).map((t) => (
            <Chip key={t} href={`/tags/${slug(t)}`}>
              #{t}
            </Chip>
          ))}
        </div>
      )}

      {/* 액션 */}
      <div className="mt-6 flex flex-wrap gap-2.5">
        <CopyLinkButton />
        <a
          href={editUrl(filePath)}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center justify-center gap-1.5 rounded-md border-[1.5px] border-line-strong px-3.5 py-[7px] text-[0.8125rem] font-semibold text-ink transition-colors hover:border-ink hover:bg-surface-2"
        >
          <Github size={15} /> GitHub에서 보기
        </a>
      </div>

      {/* 저자 카드 */}
      <div className="mt-12 rounded-lg border border-line bg-surface p-6">
        <div className="flex gap-5">
          <Avatar name="/steps" size="xl" />
          <div>
            <div className="text-[1.1875rem] font-extrabold tracking-[-0.01em] text-ink">
              /steps{' '}
              <span className="font-mono text-xs font-normal text-ink-muted">STEP BY STEP</span>
            </div>
            <p className="mb-3.5 mt-2 text-[0.9375rem] leading-relaxed text-ink-muted">{AUTHOR_BIO}</p>
            <div className="flex gap-2.5">
              <Button href="/about" variant="secondary" size="sm">
                소개 보기
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* 다음 글 */}
      {nextPost && (
        <div className="mt-10">
          <Divider label="다음 글" />
          <Link href={`/${nextPost.path}`} className="group mt-4 block">
            {nextPost.tags && <Eyebrow className="mb-1.5">{postCategory(nextPost.tags)}</Eyebrow>}
            <h3 className="text-[1.75rem] font-extrabold tracking-[-0.02em] text-ink transition-colors group-hover:text-accent-text">
              {nextPost.title} →
            </h3>
          </Link>
        </div>
      )}

      {siteMetadata.comments && (
        <div className="mt-12 pt-6" id="comment">
          <Comments slug={postSlug} />
        </div>
      )}
    </article>
  )
}
