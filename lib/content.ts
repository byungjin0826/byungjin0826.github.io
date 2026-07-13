// 콘텐츠 계층 (contentlayer 대체)
// 실제 파일 읽기는 scripts/build-content.mjs 가 빌드 전에 수행하고 JSON으로 emit한다.
// 여기서는 그 JSON을 정적 import 해 사용 — 런타임 fs 없음(Turbopack/SSG 안전).
import { slug as slugTag } from 'github-slugger'
import blogJson from './generated/blog.json'
import authorsJson from './generated/authors.json'
import type { Blog, Authors } from './content-types'

export type { Blog, Authors, CoreContent, TocItem, ReadingTime } from './content-types'

const isProduction = process.env.NODE_ENV === 'production'

export const allBlogs: Blog[] = (blogJson as unknown as Blog[]).filter(
  (p) => !isProduction || p.draft !== true
)

export const allAuthors: Authors[] = authorsJson as unknown as Authors[]

/** 날짜 내림차순 정렬 (초안 제외) */
export function sortPosts<T extends { date: string; draft?: boolean }>(posts: T[]): T[] {
  return [...posts]
    .filter((p) => !isProduction || p.draft !== true)
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
}

export function coreContent<T extends { body?: unknown }>(content: T): Omit<T, 'body'> {
  const { body, ...rest } = content
  return rest
}

export function allCoreContent<T extends { body?: unknown }>(contents: T[]): Omit<T, 'body'>[] {
  return contents.map(coreContent)
}

/** 태그 슬러그 → 개수 */
export function getAllTags(): Record<string, number> {
  const counts: Record<string, number> = {}
  for (const p of allBlogs) {
    for (const t of p.tags || []) {
      const s = slugTag(t)
      counts[s] = (counts[s] || 0) + 1
    }
  }
  return counts
}
