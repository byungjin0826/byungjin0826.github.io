/* 포스트 메타 헬퍼 — 태그 기반 카테고리, 읽는 시간, 날짜 포맷 */

export const CATEGORIES = ['전체', '기술', '일상'] as const

/** 첫 태그(tech/daily)로 한글 카테고리 산출 */
export function postCategory(tags?: string[]): string {
  if (!tags) return '기타'
  if (tags.includes('tech')) return '기술'
  if (tags.includes('daily')) return '일상'
  return '기타'
}

/** 카테고리 필터에 걸러줄 게시글 판단 */
export function inCategory(tags: string[] | undefined, category: string): boolean {
  if (category === '전체') return true
  return postCategory(tags) === category
}

/** readingTime(분) → "N분 읽기" */
export function readingLabel(readingTime?: { minutes?: number }): string {
  const m = Math.max(1, Math.round(readingTime?.minutes ?? 0))
  return `${m}분 읽기`
}

/** ISO date → "YYYY.MM.DD" */
export function dotDate(date: string): string {
  const d = new Date(date)
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}.${p(d.getMonth() + 1)}.${p(d.getDate())}`
}

/** 본문에 노출할 토픽 태그(카테고리 태그 제외) */
export function topicTags(tags?: string[]): string[] {
  if (!tags) return []
  return tags.filter((t) => t !== 'tech' && t !== 'daily')
}
