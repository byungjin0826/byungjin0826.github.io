/* 콘텐츠 타입 — fs 코드가 없어 클라이언트 컴포넌트에서도 안전하게 type import 가능 */

export interface ReadingTime {
  text: string
  minutes: number
  time: number
  words: number
}

export interface TocItem {
  value: string
  url: string
  depth: number
}

export interface Blog {
  title: string
  date: string
  tags: string[]
  lastmod?: string
  draft?: boolean
  summary?: string
  images?: string | string[]
  authors?: string[]
  layout?: string
  bibliography?: string
  canonicalUrl?: string
  // computed
  slug: string
  path: string
  filePath: string
  readingTime: ReadingTime
  toc: TocItem[]
  structuredData: Record<string, unknown>
  body: { raw: string }
}

export interface Authors {
  name: string
  avatar?: string
  occupation?: string
  company?: string
  email?: string
  twitter?: string
  linkedin?: string
  github?: string
  layout?: string
  // computed
  slug: string
  path: string
  body: { raw: string }
}

/** 본문(body)을 제외한 목록/전달용 축약 타입 */
export type CoreContent<T> = Omit<T, 'body'>
