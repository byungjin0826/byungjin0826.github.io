'use client'

/* 포스트 목차 — 우측 고정 레일(scroll-spy) + 좁은 화면 접이식.
   시안: ui_kits/byeongjin-journal/JournalPost.jsx */
import { useEffect, useMemo, useState } from 'react'
import type { TocItem } from '@/lib/content-types'

function useActiveHeading(ids: string[]) {
  const [active, setActive] = useState<string | null>(ids[0] || null)
  useEffect(() => {
    if (!ids.length || typeof IntersectionObserver === 'undefined') return
    const seen = new Set<string>()
    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) seen.add(e.target.id)
          else seen.delete(e.target.id)
        })
        const firstVisible = ids.find((id) => seen.has(id))
        if (firstVisible) setActive(firstVisible)
      },
      { rootMargin: '-90px 0px -65% 0px', threshold: 0 }
    )
    ids.forEach((id) => {
      const el = document.getElementById(id)
      if (el) obs.observe(el)
    })
    return () => obs.disconnect()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ids.join('|')])
  return active
}

function TocList({ items, active }: { items: TocItem[]; active: string | null }) {
  return (
    <ul className="m-0 list-none border-l-2 border-line p-0">
      {items.map((it) => {
        const id = it.url.slice(1)
        const on = active === id
        return (
          <li key={it.url}>
            <a
              href={it.url}
              className={`-ml-0.5 block border-l-2 py-[5px] text-[0.8125rem] leading-snug transition-colors ${
                it.depth >= 3 ? 'pl-[26px]' : 'pl-3.5'
              } ${
                on
                  ? 'border-accent font-bold text-accent-text'
                  : 'border-transparent font-medium text-ink-muted hover:text-ink'
              }`}
            >
              {it.value}
            </a>
          </li>
        )
      })}
    </ul>
  )
}

export default function Toc({ toc }: { toc: TocItem[] }) {
  const items = useMemo(() => toc.filter((t) => t.depth <= 3), [toc])
  const active = useActiveHeading(items.map((t) => t.url.slice(1)))
  if (items.length < 3) return null
  return <TocList items={items} active={active} />
}
